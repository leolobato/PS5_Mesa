/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_winsys.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "addrlib/inc/addrtypes.h"
#include "addrlib/src/amdgpu_asic_addr.h"
#include "tools/radv_debug.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "ac_linux_drm.h"
#include "sid.h"

/* ------------------------------------------------------------------ queue */

static struct radv_ps5_queue radv_ps5_queue_storage;
static struct radv_ps5_queue *radv_ps5_queue_instance;
static once_flag radv_ps5_queue_once = ONCE_FLAG_INIT;

static void
radv_ps5_queue_create(void)
{
   struct radv_ps5_queue *const queue = &radv_ps5_queue_storage;
   if (!radv_ps5_platform_init())
      return;
   /* The words the GPU runs and the marker it writes live in the window, where
    * ps5vk's submission buffer has always been. */
   if (!radv_ps5_memory_alloc(RADV_PS5_RING_BYTES, RADV_PS5_LARGE_BYTES, true, &queue->ring))
      return;
   if (!radv_ps5_memory_alloc(RADV_PS5_PAGE_BYTES, RADV_PS5_PAGE_BYTES, true, &queue->marker_memory)) {
      radv_ps5_memory_free(&queue->ring);
      return;
   }
   queue->marker = (volatile uint32_t *)queue->marker_memory.cpu;
   *queue->marker = 0;
   radv_ps5_cpu_flush((const void *)queue->marker, sizeof(*queue->marker));
   simple_mtx_init(&queue->submit_lock, mtx_plain);
   mtx_init(&queue->sync_lock, mtx_plain);
   cnd_init(&queue->sync_cond);
   radv_ps5_queue_instance = queue;
}

struct radv_ps5_queue *
radv_ps5_queue_get(void)
{
   call_once(&radv_ps5_queue_once, radv_ps5_queue_create);
   return radv_ps5_queue_instance;
}

uint64_t
radv_ps5_queue_poll(struct radv_ps5_queue *queue)
{
   const uint64_t submitted = p_atomic_read(&queue->submitted_seq);
   /* The marker holds the low 32 bits of the newest sequence the GPU has run.
    * Its CPU cache line may hold an older value, so it is evicted before the
    * read, as ps5vk's marker poll does. */
   radv_ps5_cpu_flush((const void *)queue->marker, sizeof(*queue->marker));
   const uint32_t marker = *queue->marker;
   /* The GPU never lags the CPU by 2^32 submissions, so the newest sequence
    * with these low bits that is not past the submitted one is the one. */
   const uint64_t completed = submitted - (uint32_t)((uint32_t)submitted - marker);
   uint64_t known = p_atomic_read(&queue->completed_seq);
   while (completed > known) {
      const uint64_t seen = p_atomic_cmpxchg(&queue->completed_seq, known, completed);
      if (seen == known) {
         known = completed;
         break;
      }
      known = seen;
   }
   return known;
}

/* A marker that makes no progress for this long while work is outstanding
 * means the GPU is not running it: the device is lost (ps5vk gives up after 2 s;
 * a conformance run's heaviest cases need more). */
#define RADV_PS5_STALL_NS (UINT64_C(10) * 1000000000)
/* Before sleeping, the marker is checked in a loop for this long: most waits
 * end well inside a millisecond (PS5_Vulkan R36). */
#define RADV_PS5_SPIN_NS UINT64_C(1500000)

/* How many submissions still in flight a stall report describes, how many of
 * the stuck one's waits it lists, how many it reads, and how many words of the
 * work that did not finish it prints. */
#define RADV_PS5_STALL_SUBMISSIONS 4
#define RADV_PS5_STALL_WAITS 8
#define RADV_PS5_STALL_MAX_WAITS 512
#define RADV_PS5_STALL_DUMP_WORDS 1024

static const char *
radv_ps5_wait_function(uint32_t function)
{
   static const char *const names[8] = {"always", "<", "<=", "==", "!=", ">=", ">", "(reserved)"};
   return names[function & 7];
}

bool
radv_ps5_trace_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *const value = getenv("RADV_PS5_TRACE");
      enabled = value && *value && strcmp(value, "0") != 0;
   }
   return enabled;
}

uint32_t
radv_ps5_packet_length(const uint32_t *words, uint32_t at, uint32_t count)
{
   const uint32_t header = words[at];
   const unsigned type = PKT_TYPE_G(header);
   if (type == 2)
      return 1;
   if (type == 1)
      return 0;
   /* PKT3_NOP_PAD: a NOP of the header alone. */
   const uint32_t length =
      type == 3 && PKT3_IT_OPCODE_G(header) == PKT3_NOP && PKT_COUNT_G(header) == 0x3fff ? 1 : PKT_COUNT_G(header) + 2;
   return length <= count - at ? length : 0;
}

bool
radv_ps5_packet_is_draw(unsigned opcode)
{
   switch (opcode) {
   case PKT3_DRAW_INDIRECT:
   case PKT3_DRAW_INDEX_INDIRECT:
   case PKT3_DRAW_INDEX_2:
   case PKT3_DRAW_INDIRECT_MULTI:
   case PKT3_DRAW_INDEX_AUTO:
   case PKT3_DRAW_INDEX_MULTI_AUTO:
   case PKT3_DRAW_INDEX_OFFSET_2:
   case PKT3_DRAW_INDEX_INDIRECT_MULTI:
   case PKT3_DISPATCH_MESH_INDIRECT_MULTI:
   case PKT3_DISPATCH_TASKMESH_GFX:
      return true;
   default:
      return false;
   }
}

bool
radv_ps5_packet_is_dispatch(unsigned opcode)
{
   return opcode == PKT3_DISPATCH_DIRECT || opcode == PKT3_DISPATCH_INDIRECT;
}

/* The shader program registers RADV sets on GFX10.3 (radv_shader.c): pixel,
 * legacy vertex, NGG or geometry (ES), hull (LS) and compute. Each holds the
 * program address >> 8, with bits 40 and up in the next register. */
static const struct {
   uint32_t reg;
   const char *stage;
} radv_ps5_program_regs[] = {
   {R_00B020_SPI_SHADER_PGM_LO_PS, "PS"}, {R_00B120_SPI_SHADER_PGM_LO_VS, "VS"},
   {R_00B320_SPI_SHADER_PGM_LO_ES, "GS"}, {R_00B520_SPI_SHADER_PGM_LO_LS, "HS"},
   {R_00B830_COMPUTE_PGM_LO, "CS"},
};

/* The program addresses (>> 8) the packets before word end set, per entry of
 * radv_ps5_program_regs. */
static void
radv_ps5_programs_before(const uint32_t *words, uint32_t count, uint32_t end, uint32_t *program_lo,
                         uint32_t *program_hi)
{
   for (uint32_t at = 0; at < end;) {
      const uint32_t length = radv_ps5_packet_length(words, at, count);
      if (!length)
         break;
      if (PKT_TYPE_G(words[at]) == 3 && PKT3_IT_OPCODE_G(words[at]) == PKT3_SET_SH_REG && length >= 3) {
         const uint32_t first = SI_SH_REG_OFFSET + words[at + 1] * 4;
         for (uint32_t v = 0; v < length - 2; v++) {
            const uint32_t reg = first + v * 4;
            for (unsigned r = 0; r < ARRAY_SIZE(radv_ps5_program_regs); r++) {
               if (reg == radv_ps5_program_regs[r].reg)
                  program_lo[r] = words[at + 2 + v];
               else if (reg == radv_ps5_program_regs[r].reg + 4)
                  program_hi[r] = words[at + 2 + v];
            }
         }
      }
      at += length;
   }
}

/* Words as lines of eight, after a prefix: for decoding offline (ac_parse_ib
 * for packets, llvm-mc for shader code). */
static void
radv_ps5_report_words(const char *prefix, const uint32_t *words, uint32_t first, uint32_t end)
{
   for (uint32_t line = first; line < end; line += 8) {
      char text[8 * 9 + 1];
      unsigned used = 0;
      for (uint32_t w = line; w < end && w < line + 8; w++)
         used += snprintf(text + used, sizeof(text) - used, " %08x", words[w]);
      fprintf(stderr, "radv/ps5: GPU stall: %s @%u:%s\n", prefix, line, text);
   }
}

/* A shader program's code, up to its first s_endpgm: shaders live in the
 * window, which the CPU sees at the GPU's addresses. */
#define RADV_PS5_STALL_CODE_WORDS 2048
#define RADV_PS5_S_ENDPGM 0xbf810000u

static void
radv_ps5_report_program(const char *stage, uint64_t address)
{
   if (address < RADV_PS5_WINDOW_BASE || address >= RADV_PS5_WINDOW_BASE + RADV_PS5_WINDOW_BYTES - 4) {
      fprintf(stderr, "radv/ps5: GPU stall: %s program 0x%" PRIx64 " is outside the window; not read\n", stage, address);
      return;
   }
   const uint32_t *const code = (const uint32_t *)(uintptr_t)address;
   const uint64_t room = (RADV_PS5_WINDOW_BASE + RADV_PS5_WINDOW_BYTES - address) / 4;
   const uint32_t limit = (uint32_t)MIN2(room, RADV_PS5_STALL_CODE_WORDS);
   uint32_t words = 0;
   bool ended = false;
   while (words < limit && !ended)
      ended = code[words++] == RADV_PS5_S_ENDPGM;
   fprintf(stderr, "radv/ps5: GPU stall: %s program 0x%" PRIx64 ": %u words%s\n", stage, address, words,
           ended ? " to s_endpgm" : ", no s_endpgm");
   char prefix[32];
   snprintf(prefix, sizeof(prefix), "%s code", stage);
   radv_ps5_report_words(prefix, code, 0, words);
}

/* Whether the CPU can read the byte at address: written to a pipe, an address
 * nothing maps fails with EFAULT where reading it would end the process. Every
 * buffer of the window and the device-memory region is mapped for the CPU at
 * the GPU's address, so an unreadable one is memory the GPU cannot reach
 * either. A pipe that cannot be made says readable. */
static bool
radv_ps5_readable(uint64_t address)
{
   static int probe[2] = {-1, -1};
   if (probe[0] < 0 && pipe(probe) != 0) {
      probe[0] = probe[1] = -1;
      return true;
   }
   if (write(probe[1], (const void *)(uintptr_t)address, 1) != 1)
      return false;
   char byte;
   return read(probe[0], &byte, 1) == 1;
}

/* A shader stage's user SGPRs as the packets before word end set them. */
#define RADV_PS5_PS_USER_DATA 16
#define RADV_PS5_USER_DATA 32

static void
radv_ps5_user_data_before(const uint32_t *words, uint32_t count, uint32_t end, uint32_t base_reg, unsigned regs,
                          uint32_t *user_data, bool *set)
{
   for (uint32_t at = 0; at < end;) {
      const uint32_t length = radv_ps5_packet_length(words, at, count);
      if (!length)
         break;
      if (PKT_TYPE_G(words[at]) == 3 && PKT3_IT_OPCODE_G(words[at]) == PKT3_SET_SH_REG && length >= 3) {
         const uint32_t first = SI_SH_REG_OFFSET + words[at + 1] * 4;
         for (uint32_t v = 0; v < length - 2; v++) {
            const uint32_t reg = first + v * 4;
            if (reg >= base_reg && reg < base_reg + regs * 4) {
               const unsigned index = (reg - base_reg) / 4;
               user_data[index] = words[at + 2 + v];
               set[index] = true;
            }
         }
      }
      at += length;
   }
}

/* What a stuck draw's pixel shader reads through its user SGPRs: each one that
 * holds an address in the window (a descriptor set or a push constant block)
 * is printed with the 16 words there, and every buffer descriptor among them
 * with the start of its buffer, as the GPU sees them now. Pointers are
 * followed only into memory a buffer descriptor or the window places in the
 * GPU's range; this runs last, after everything else is printed. */
#define RADV_PS5_STALL_BUFFER_WORDS 128

static void
radv_ps5_report_buffer(unsigned index, unsigned slot, const uint32_t *descriptor)
{
   const uint64_t base = descriptor[0] | ((uint64_t)(descriptor[1] & 0xffff) << 32);
   const uint32_t records = descriptor[2];
   /* A buffer descriptor's word 3 has TYPE 0 in bits 31:30 (an image's is 8
    * to 15 in bits 31:28) and names its components, so it is never 0; its
    * base and size must also be sane before the base is read. */
   if (!descriptor[3] || (descriptor[3] >> 30) || !base || base >= RADV_PS5_GPU_ADDRESS_LIMIT || (base & 3) || !records ||
       records > (1u << 28))
      return;
   if (!radv_ps5_readable(base)) {
      fprintf(stderr, "radv/ps5: GPU stall: PS user data %u descriptor %u: buffer 0x%" PRIx64 " is NOT MAPPED\n", index,
              slot, base);
      return;
   }
   const uint32_t words = MIN2(records / 4, RADV_PS5_STALL_BUFFER_WORDS);
   fprintf(stderr,
           "radv/ps5: GPU stall: PS user data %u descriptor %u: buffer 0x%" PRIx64 ", %u bytes (%08x %08x %08x %08x)\n",
           index, slot, base, records, descriptor[0], descriptor[1], descriptor[2], descriptor[3]);
   char prefix[48];
   snprintf(prefix, sizeof(prefix), "PS user data %u buffer %u", index, slot);
   radv_ps5_report_words(prefix, (const uint32_t *)(uintptr_t)base, 0, words);
}

static void
radv_ps5_report_ps_buffers(const uint32_t *words, uint32_t count, uint32_t end)
{
   uint32_t user_data[RADV_PS5_PS_USER_DATA] = {0};
   bool set[RADV_PS5_PS_USER_DATA] = {false};
   radv_ps5_user_data_before(words, count, end, R_00B030_SPI_SHADER_USER_DATA_PS_0, RADV_PS5_PS_USER_DATA, user_data,
                             set);
   /* All of them first: following a pointer reads memory. */
   for (unsigned i = 0; i < RADV_PS5_PS_USER_DATA; i++) {
      if (set[i])
         fprintf(stderr, "radv/ps5: GPU stall: PS user data %u = 0x%08x\n", i, user_data[i]);
   }
   for (unsigned i = 0; i < RADV_PS5_PS_USER_DATA; i++) {
      if (!set[i])
         continue;
      /* 32-bit pointers carry the window's high word (RADV_PS5_ADDRESS32_HI). */
      if (!user_data[i] || (user_data[i] & 15) || user_data[i] > 0xffffffffu - 64)
         continue;
      const uint32_t *const table = (const uint32_t *)(uintptr_t)(RADV_PS5_WINDOW_BASE + user_data[i]);
      if (!radv_ps5_readable((uint64_t)(uintptr_t)table)) {
         fprintf(stderr, "radv/ps5: GPU stall: PS user data %u table is NOT MAPPED\n", i);
         continue;
      }
      char prefix[32];
      snprintf(prefix, sizeof(prefix), "PS user data %u table", i);
      radv_ps5_report_words(prefix, table, 0, 16);
      for (unsigned slot = 0; slot < 4; slot++)
         radv_ps5_report_buffer(i, slot, table + slot * 4);
   }
}

/* Whether every descriptor a stuck draw can reach points at mapped memory: the
 * descriptor tables the user SGPRs of its pixel and geometry stages point to
 * (their first RADV_PS5_STALL_TABLE_WORDS words, which hold every descriptor
 * set of a DXVK draw), every image and buffer descriptor in them, and the
 * index buffer. A GPU that waits on a fault forever stalls like a shader that
 * never ends. Only what is not mapped is printed, then a count. */
#define RADV_PS5_STALL_TABLE_WORDS 512

static void
radv_ps5_check_tables(const char *stage, uint32_t base_reg, const uint32_t *words, uint32_t count, uint32_t end)
{
   uint32_t user_data[RADV_PS5_USER_DATA] = {0};
   bool set[RADV_PS5_USER_DATA] = {false};
   radv_ps5_user_data_before(words, count, end, base_reg, RADV_PS5_USER_DATA, user_data, set);
   unsigned tables = 0, images = 0, buffers = 0, unmapped = 0;
   for (unsigned i = 0; i < RADV_PS5_USER_DATA; i++) {
      if (!set[i] || !user_data[i] || (user_data[i] & 15) || user_data[i] > 0xffffffffu - RADV_PS5_STALL_TABLE_WORDS * 4)
         continue;
      const uint64_t table_va = RADV_PS5_WINDOW_BASE + user_data[i];
      if (!radv_ps5_readable(table_va))
         continue; /* not a pointer, or a table nothing maps: printed below if so */
      tables++;
      const uint32_t *const table = (const uint32_t *)(uintptr_t)table_va;
      for (uint32_t w = 0; w + 4 <= RADV_PS5_STALL_TABLE_WORDS; w += 4) {
         if (!radv_ps5_readable(table_va + w * 4))
            break;
         const uint32_t *const d = table + w;
         uint64_t base = 0;
         const char *kind = NULL;
         if ((d[3] >> 28) >= 8 && !(w & 7) && w + 8 <= RADV_PS5_STALL_TABLE_WORDS) {
            base = ((uint64_t)d[0] | ((uint64_t)(d[1] & 0xff) << 32)) << 8;
            kind = "image";
            images++;
            w += 4;
         } else if (d[3] && !(d[3] >> 30) && d[2] && d[2] <= (1u << 28)) {
            base = d[0] | ((uint64_t)(d[1] & 0xffff) << 32);
            if (base < RADV_PS5_WINDOW_BASE || base >= RADV_PS5_GPU_ADDRESS_LIMIT)
               continue; /* a sampler or data that only looks like a buffer */
            kind = "buffer";
            buffers++;
         }
         if (kind && base && !radv_ps5_readable(base)) {
            unmapped++;
            fprintf(stderr,
                    "radv/ps5: GPU stall: %s user data %u word %u: %s 0x%" PRIx64 " is NOT MAPPED (%08x %08x %08x %08x)\n",
                    stage, i, w, kind, base, d[0], d[1], d[2], d[3]);
         }
      }
   }
   fprintf(stderr, "radv/ps5: GPU stall: %s tables %u: %u images, %u buffers, %u not mapped\n", stage, tables, images,
           buffers, unmapped);
}

/* With RADV_PS5_TRACE, the breadcrumb word names the last draw of the stuck
 * submission that finished: the next one is the draw that did not, and its
 * packets and shaders are printed. */
static void
radv_ps5_report_breadcrumb(uint64_t seq, const uint32_t *words, uint32_t count, uint64_t trace_address)
{
   volatile const uint32_t *const trace = (volatile const uint32_t *)(uintptr_t)trace_address;
   radv_ps5_cpu_flush((const void *)trace, sizeof(*trace));
   const uint32_t value = *trace;
   const uint32_t finished = (value >> 16) == (uint32_t)(seq & 0xffff) ? (value & 0xffff) : 0;
   uint32_t last_work = UINT32_MAX, window = 0, culprit = UINT32_MAX, total = 0;
   for (uint32_t at = 0; at < count;) {
      const uint32_t length = radv_ps5_packet_length(words, at, count);
      if (!length)
         break;
      if (PKT_TYPE_G(words[at]) == 3) {
         const unsigned opcode = PKT3_IT_OPCODE_G(words[at]);
         if (radv_ps5_packet_is_draw(opcode) || radv_ps5_packet_is_dispatch(opcode)) {
            last_work = at;
         } else if (opcode == PKT3_RELEASE_MEM && length >= 6 &&
                    (words[at + 3] | ((uint64_t)words[at + 4] << 32)) == trace_address) {
            const uint32_t crumb = words[at + 5] & 0xffff;
            total = crumb;
            if (crumb == finished)
               window = at + length;
            else if (crumb == finished + 1 && culprit == UINT32_MAX)
               culprit = last_work;
         }
      }
      at += length;
   }
   fprintf(stderr,
           "radv/ps5: GPU stall: breadcrumb 0x%08x: submission %" PRIu64 " finished %u of its %u draws and dispatches\n",
           value, seq, finished, total);
   if (culprit == UINT32_MAX) {
      fprintf(stderr, "radv/ps5: GPU stall: no draw after breadcrumb %u in submission %" PRIu64 "\n", finished, seq);
      return;
   }
   const uint32_t culprit_end = culprit + radv_ps5_packet_length(words, culprit, count);
   fprintf(stderr, "radv/ps5: GPU stall: submission %" PRIu64 " draw %u at word %u (opcode 0x%02x) did not finish\n", seq,
           finished + 1, culprit, PKT3_IT_OPCODE_G(words[culprit]));
   const uint32_t first = culprit_end - window > RADV_PS5_STALL_DUMP_WORDS ? culprit_end - RADV_PS5_STALL_DUMP_WORDS
                                                                            : window;
   radv_ps5_report_words("draw window", words, first, culprit_end);
   uint32_t program_lo[ARRAY_SIZE(radv_ps5_program_regs)] = {0};
   uint32_t program_hi[ARRAY_SIZE(radv_ps5_program_regs)] = {0};
   radv_ps5_programs_before(words, count, culprit, program_lo, program_hi);
   const bool dispatch = radv_ps5_packet_is_dispatch(PKT3_IT_OPCODE_G(words[culprit]));
   for (unsigned r = 0; r < ARRAY_SIZE(radv_ps5_program_regs); r++) {
      const bool compute = radv_ps5_program_regs[r].reg == R_00B830_COMPUTE_PGM_LO;
      if (program_lo[r] && compute == dispatch)
         radv_ps5_report_program(radv_ps5_program_regs[r].stage,
                                 ((uint64_t)program_hi[r] << 40) | ((uint64_t)program_lo[r] << 8));
   }
   if (!dispatch) {
      if (PKT3_IT_OPCODE_G(words[culprit]) == PKT3_DRAW_INDEX_2 && culprit_end - culprit >= 5) {
         const uint64_t index_va = words[culprit + 2] | ((uint64_t)words[culprit + 3] << 32);
         fprintf(stderr, "radv/ps5: GPU stall: index buffer 0x%" PRIx64 " %s\n", index_va,
                 radv_ps5_readable(index_va) ? "is mapped" : "is NOT MAPPED");
      }
      radv_ps5_check_tables("GS", R_00B230_SPI_SHADER_USER_DATA_GS_0, words, count, culprit);
      radv_ps5_check_tables("PS", R_00B030_SPI_SHADER_USER_DATA_PS_0, words, count, culprit);
      radv_ps5_report_ps_buffers(words, count, culprit);
   }
}

struct radv_ps5_stall_wait {
   uint32_t at;
   uint32_t control;
   uint64_t address;
   uint32_t reference;
   uint32_t mask;
};

/* What the GPU was given in one submission: its packets by kind and each
 * WAIT_REG_MEM, then where in it the GPU stopped. RADV ends work it has to wait
 * for with a barrier (radv_cs_emit_cache_flush): a RELEASE_MEM writes the next
 * value of the command buffer's counter once the work before it has finished,
 * and the CP waits until the counter equals it. The counter lives in the
 * command buffer's upload buffer, which the CPU sees at the GPU's address while
 * the submission is pending. Its value tells which barrier the GPU passed last:
 * the work after that one is what did not finish, and the report prints it. */
static void
radv_ps5_report_packets(uint64_t seq, const uint32_t *words, uint32_t count, uint64_t trace_address)
{
   static struct radv_ps5_stall_wait waits_seen[RADV_PS5_STALL_MAX_WAITS];
   unsigned packets = 0, draws = 0, dispatches = 0, waits = 0, dma = 0, syncs = 0, kept = 0;
   uint32_t at = 0;
   while (at < count) {
      const uint32_t length = radv_ps5_packet_length(words, at, count);
      if (!length) {
         fprintf(stderr, "radv/ps5: GPU stall: submission %" PRIu64 " word %u is not a whole packet (0x%08x)\n", seq,
                 at, words[at]);
         count = at;
         break;
      }
      if (PKT_TYPE_G(words[at]) != 3) {
         at += length;
         continue;
      }
      packets++;
      const unsigned opcode = PKT3_IT_OPCODE_G(words[at]);
      if (radv_ps5_packet_is_draw(opcode))
         draws++;
      else if (radv_ps5_packet_is_dispatch(opcode))
         dispatches++;
      switch (opcode) {
      case PKT3_DMA_DATA:
         dma++;
         break;
      case PKT3_EVENT_WRITE:
      case PKT3_RELEASE_MEM:
      case PKT3_ACQUIRE_MEM:
      case PKT3_PFP_SYNC_ME:
         syncs++;
         break;
      case PKT3_WAIT_REG_MEM:
         if (length < 7)
            break;
         if (kept < RADV_PS5_STALL_MAX_WAITS)
            waits_seen[kept++] = (struct radv_ps5_stall_wait){
               .at = at,
               .control = words[at + 1],
               .address = words[at + 2] | ((uint64_t)words[at + 3] << 32),
               .reference = words[at + 4],
               .mask = words[at + 5],
            };
         if (waits++ < RADV_PS5_STALL_WAITS) {
            const uint32_t control = words[at + 1];
            if (control & 0x10)
               fprintf(stderr,
                       "radv/ps5: GPU stall: submission %" PRIu64 " word %u waits until memory 0x%" PRIx64
                       " & 0x%08x %s 0x%08x (%s)\n",
                       seq, at, (uint64_t)words[at + 2] | ((uint64_t)words[at + 3] << 32), words[at + 5],
                       radv_ps5_wait_function(control), words[at + 4], (control >> 8) & 3 ? "PFP" : "ME");
            else
               fprintf(stderr,
                       "radv/ps5: GPU stall: submission %" PRIu64 " word %u waits until register 0x%x & 0x%08x %s "
                       "0x%08x (%s)\n",
                       seq, at, words[at + 2], words[at + 5], radv_ps5_wait_function(control), words[at + 4],
                       (control >> 8) & 3 ? "PFP" : "ME");
         }
         break;
      default:
         break;
      }
      at += length;
   }
   fprintf(stderr,
           "radv/ps5: GPU stall: submission %" PRIu64 " has %u packets: %u draws, %u dispatches, %u waits, %u DMA, "
           "%u syncs\n",
           seq, packets, draws, dispatches, waits, dma, syncs);

   /* The first barrier not yet reached. Only an equality wait on memory in the
    * window is a barrier; its counter is read once per address. */
   uint32_t from = 0, to = count;
   uint64_t read_address = 0;
   uint32_t value = 0;
   for (unsigned i = 0; i < kept; i++) {
      const struct radv_ps5_stall_wait *const wait = &waits_seen[i];
      if (!(wait->control & 0x10) || (wait->control & 7) != 3 || wait->mask != 0xffffffff ||
          wait->address < RADV_PS5_WINDOW_BASE || wait->address + 4 > RADV_PS5_WINDOW_BASE + RADV_PS5_WINDOW_BYTES)
         continue;
      if (wait->address != read_address) {
         volatile const uint32_t *const counter = (volatile const uint32_t *)(uintptr_t)wait->address;
         radv_ps5_cpu_flush((const void *)counter, sizeof(*counter));
         value = *counter;
         read_address = wait->address;
      }
      if (wait->reference <= value) {
         from = wait->at + 7;
         continue;
      }
      to = wait->at;
      fprintf(stderr,
              "radv/ps5: GPU stall: submission %" PRIu64 " stopped before the barrier at word %u: the counter at 0x%" PRIx64
              " holds 0x%08x, the barrier waits for 0x%08x\n",
              seq, wait->at, wait->address, value, wait->reference);
      break;
   }
   if (to == count)
      fprintf(stderr, "radv/ps5: GPU stall: submission %" PRIu64 " passed every barrier; it stopped after word %u\n", seq,
              from);

   /* The shaders bound when that work started, and what the work is. */
   uint32_t program_lo[ARRAY_SIZE(radv_ps5_program_regs)] = {0};
   uint32_t program_hi[ARRAY_SIZE(radv_ps5_program_regs)] = {0};
   radv_ps5_programs_before(words, count, to, program_lo, program_hi);
   unsigned work_draws = 0, work_dispatches = 0, work_dma = 0;
   for (at = from; at < to;) {
      const uint32_t length = radv_ps5_packet_length(words, at, count);
      if (!length)
         break;
      const unsigned opcode = PKT3_IT_OPCODE_G(words[at]);
      if (PKT_TYPE_G(words[at]) == 3) {
         work_draws += radv_ps5_packet_is_draw(opcode);
         work_dispatches += radv_ps5_packet_is_dispatch(opcode);
         work_dma += opcode == PKT3_DMA_DATA;
      }
      at += length;
   }
   fprintf(stderr,
           "radv/ps5: GPU stall: submission %" PRIu64 " words %u-%u did not finish: %u draws, %u dispatches, %u DMA\n",
           seq, from, to, work_draws, work_dispatches, work_dma);
   for (unsigned r = 0; r < ARRAY_SIZE(radv_ps5_program_regs); r++) {
      if (program_lo[r])
         fprintf(stderr, "radv/ps5: GPU stall: submission %" PRIu64 " %s program 0x%" PRIx64 " bound by word %u\n", seq,
                 radv_ps5_program_regs[r].stage, ((uint64_t)program_hi[r] << 40) | ((uint64_t)program_lo[r] << 8), to);
   }

   /* The words themselves, for decoding (ac_parse_ib); the last ones before the
    * barrier if there are more. */
   const uint32_t first = to - from > RADV_PS5_STALL_DUMP_WORDS ? to - RADV_PS5_STALL_DUMP_WORDS : from;
   char prefix[48];
   snprintf(prefix, sizeof(prefix), "submission %" PRIu64, seq);
   radv_ps5_report_words(prefix, words, first, to);

   if (radv_ps5_trace_enabled())
      radv_ps5_report_breadcrumb(seq, words, count, trace_address);
}

void
radv_ps5_queue_report_stall(struct radv_ps5_queue *queue, uint64_t seq, const char *waiter, uint64_t stalled_ns)
{
   const uint64_t completed = radv_ps5_queue_poll(queue);
   if (p_atomic_xchg(&queue->stall_reported, 1)) {
      fprintf(stderr, "radv/ps5: the GPU has not reached submission %" PRIu64 " (at %" PRIu64 ") in %u s (%s)\n", seq,
              completed, (unsigned)(stalled_ns / 1000000000), waiter);
      return;
   }
   /* Read without the submit lock. A record the GPU has not finished is neither
    * retired nor reused, so its fields and words stay as they were; first and
    * count are read once, as another thread may still submit or retire. */
   const uint64_t now = os_time_get_nano();
   const uint32_t first = p_atomic_read(&queue->in_flight_first);
   const uint32_t count = MIN2(p_atomic_read(&queue->in_flight_count), RADV_PS5_MAX_IN_FLIGHT);
   fprintf(stderr,
           "radv/ps5: GPU stall: %s waited for submission %" PRIu64 "; the GPU has run up to %" PRIu64 " of %" PRIu64
           ", %u in flight, no progress for %" PRIu64 " ms\n",
           waiter, seq, completed, p_atomic_read(&queue->submitted_seq), count, stalled_ns / 1000000);
   const uint64_t agc_seq = p_atomic_read(&queue->agc_submit_seq);
   const uint64_t agc_ns = p_atomic_read(&queue->agc_submit_ns);
   if (agc_seq)
      fprintf(stderr, "radv/ps5: GPU stall: AGC has not returned from submitting %" PRIu64 " for %" PRIu64 " ms\n",
              agc_seq, (now - agc_ns) / 1000000);
   if (queue->tess_factor_ring_va)
      fprintf(stderr, "radv/ps5: GPU stall: tessellation factor ring 0x%" PRIx64 " (%u bytes) since submission %" PRIu64
              "\n", queue->tess_factor_ring_va, queue->tess_factor_ring_size, queue->tess_factor_ring_seq);
   unsigned shown = 0, pending = 0;
   for (uint32_t i = 0; i < count; i++) {
      const struct radv_ps5_in_flight *const record = &queue->in_flight[(first + i) % RADV_PS5_MAX_IN_FLIGHT];
      if (record->seq <= completed)
         continue;
      pending++;
      if (shown == RADV_PS5_STALL_SUBMISSIONS)
         continue;
      const uint32_t *const words =
         record->dedicated.cpu ? (const uint32_t *)record->dedicated.cpu : (const uint32_t *)queue->ring.cpu + record->start;
      if (record->dedicated.cpu)
         fprintf(stderr,
                 "radv/ps5: GPU stall: submission %" PRIu64 ": %u words in %u AGC submissions, in a buffer of its own, "
                 "submitted %" PRIu64 " ms ago\n",
                 record->seq, record->words, record->parts, (now - record->submitted_ns) / 1000000);
      else
         fprintf(stderr,
                 "radv/ps5: GPU stall: submission %" PRIu64 ": %u words in %u AGC submissions, ring words %u-%u, "
                 "submitted %" PRIu64 " ms ago\n",
                 record->seq, record->words, record->parts, record->start, record->end,
                 (now - record->submitted_ns) / 1000000);
      /* The oldest one is where the GPU is. */
      if (shown == 0 && record->words)
         radv_ps5_report_packets(record->seq, words, record->words,
                                 (uint64_t)(uintptr_t)queue->marker_memory.cpu + RADV_PS5_TRACE_OFFSET);
      shown++;
   }
   if (pending > shown)
      fprintf(stderr, "radv/ps5: GPU stall: %u more submissions wait behind them\n", pending - shown);
}

bool
radv_ps5_queue_wait_seq(struct radv_ps5_queue *queue, uint64_t seq, uint64_t abs_timeout_ns, const char *waiter)
{
   uint64_t completed = radv_ps5_queue_poll(queue);
   if (completed >= seq)
      return true;
   uint64_t now = os_time_get_nano();
   const uint64_t spin_until = now + RADV_PS5_SPIN_NS;
   uint64_t progress_at = now;
   uint64_t last = completed;
   for (;;) {
      completed = radv_ps5_queue_poll(queue);
      if (completed >= seq)
         return true;
      now = os_time_get_nano();
      if (now >= abs_timeout_ns)
         return false;
      if (completed != last) {
         last = completed;
         progress_at = now;
      } else if (now - progress_at > RADV_PS5_STALL_NS) {
         radv_ps5_queue_report_stall(queue, seq, waiter, now - progress_at);
         return false;
      }
      if (now < spin_until)
         __builtin_ia32_pause();
      else
         radv_ps5_sleep_us(1000);
   }
}

/* ----------------------------------------------------------- the GPU itself */

/* The console's GPU as the winsys describes it to RADV, in the form the amdgpu
 * kernel reports a GPU (Mesa's recorded devices, src/amd/common/amdgpu_devices.c,
 * use the same form for AddrLib's tests), then completed by ac_gpu_info's own
 * derivations.
 *
 * Measured on my console (PS5_Vulkan):
 * - shaders compiled by ACO for GFX10.3 (CHIP_NAVI21) run: every psbc stage
 *   since M5 A, and compute in waves of 32;
 * - tiled images follow AddrLib's GFX10 swizzles with sixteen pipes, a 256-byte
 *   pipe interleave and no RB+ -- the Navi10 configuration -- texel for texel in
 *   both 64 KiB modes (docs/HARDWARE_FINDINGS.md, tools/mip-layout-oracle.cpp);
 * - the GPU timestamp counts at 100 MHz (V0-query, pid 143);
 * - a full-screen draw's occlusion count is a sixteenth of its samples, which
 *   is one counter per render backend if there are sixteen (to be proved by S6).
 *
 * Not yet measured, and marked where they are set: the shader engine and
 * compute-unit counts (the public PS5 figures: two shader engines, 36 active
 * compute units of 40) and the render-backend layout. Probe S3 measures them;
 * until then they only size scratch and a few hardware limits. */
#define RADV_PS5_EXTERNAL_REV 0x28 /* the NAVI21 range: GFX10.3 */
/* The revision of the GC 10.1.3 model (radv_ps5_take_gc_10_1_3_traits): the
 * start of GFX1013's range. */
#define RADV_PS5_GFX1013_REV 0x82
#define RADV_PS5_GB_ADDR_CONFIG 0x00100044

/* A GFX10 GPU with the console's units, clocks and memory, as the amdgpu
 * kernel would report it and ac_gpu_info completes it; the IP version and the
 * revision decide which GFX10 it is. */
static void
radv_ps5_fill_info(struct radeon_info *info, bool compiler_compat_mode, uint32_t ip_minor,
                   uint32_t ip_discovery_version, uint32_t external_rev)
{
   const struct drm_amdgpu_info_hw_ip gfx_ip = {
      .hw_ip_version_major = 10,
      .hw_ip_version_minor = ip_minor,
      .ib_start_alignment = 32,
      .ib_size_alignment = 32,
      .available_rings = 0x1,
      .ip_discovery_version = ip_discovery_version,
   };
   struct drm_amdgpu_info_device dev = {
      .device_id = 0,
      .chip_rev = 0,
      .external_rev = external_rev,
      .family = FAMILY_NV,
      /* Provisional until S3. */
      .num_shader_engines = 2,
      .num_shader_arrays_per_engine = 2,
      .cu_active_number = 36,
      .cu_ao_mask = 0x1ff,
      .cu_bitmap = {{0x1ff, 0x1ff}, {0x1ff, 0x1ff}},
      .cu_ao_bitmap = {{0x1ff, 0x1ff}, {0x1ff, 0x1ff}},
      .num_cu_per_sh = 10,
      .enabled_rb_pipes_mask = 0xffff,
      .num_rb_pipes = 16,
      .num_tcc_blocks = 16,
      /* The 100 MHz timestamp clock, in kHz. */
      .gpu_counter_freq = 100000,
      .max_engine_clock = UINT64_C(2230000),
      .min_engine_clock = UINT64_C(500000),
      .max_memory_clock = UINT64_C(1750000),
      .min_memory_clock = UINT64_C(96000),
      .num_hw_gfx_contexts = 8,
      .ids_flags = 0,
      .virtual_address_offset = UINT64_C(0x200000),
      .virtual_address_max = RADV_PS5_GPU_ADDRESS_LIMIT,
      .virtual_address_alignment = RADV_PS5_PAGE_BYTES,
      .pte_fragment_size = 2097152,
      .gart_page_size = RADV_PS5_PAGE_BYTES,
      .vram_type = AMDGPU_VRAM_TYPE_GDDR6,
      .vram_bit_width = 256,
      .gc_double_offchip_lds_buf = 1,
      .wave_front_size = 32,
      .num_shader_visible_vgprs = 1024,
      .gs_vgt_table_depth = 32,
      .gs_prim_buffer_depth = 1792,
      .max_gs_waves_per_vgt = 32,
   };
   const uint64_t pool = radv_ps5_memory_pool_bytes();
   const struct drm_amdgpu_memory_info memory = {
      .vram = {.total_heap_size = pool},
      .cpu_accessible_vram = {.total_heap_size = pool},
      .gtt = {0},
   };
   const struct amdgpu_gpu_info gpu_info = {
      .gb_addr_cfg = RADV_PS5_GB_ADDR_CONFIG,
   };

   memset(info, 0, sizeof(*info));
   /* No kernel stands behind this description; the version is the lowest
    * RADV's amdgpu path accepts, so no feature is assumed from a newer kernel. */
   info->drm_major = 3;
   info->drm_minor = 54;
   info->is_amdgpu = false;

   ac_fill_hw_ip_info(info, &dev, AMD_IP_GFX, &gfx_ip);
   info->ip[AMD_IP_GFX].num_instances = 1;
   ac_identify_chip(info, &dev);
   ac_fill_memory_info(info, &dev, &memory);
   ac_fill_hw_info(info, &dev);
   ac_fill_tiling_info(info, &gpu_info);
   ac_fill_feature_info(info, &dev);
   ac_fill_bug_info(info);
   ac_fill_tess_info(info);
   ac_fill_compiler_info(info, &dev, compiler_compat_mode);
}

/* The GPU's fixed-function blocks are GC 10.1.3's, the chip of the AMD BC-250
 * (Mesa's CHIP_GFX1013, "GFX10 plus ray tracing instructions"; upstream amdgpu
 * programs it as GC 10.1.3 with the GB_ADDR_CONFIG above), while register
 * programming and the shader core behave as GFX10.3's, the identity RADV gets.
 * So these traits come from Mesa's GFX1013 model, derived from the same inputs.
 * Each was first found on the console as a symptom (PS5_Vulkan RADV_PHASE.md,
 * 2026-09-27, has the audit), and the model gives every one of them; the
 * assertion keeps the model and the measurements in step across Mesa updates.
 *
 * - No accelerated dot products: v_dot4_u32_u8 does not compute here. NGG's
 *   workgroup repack sums its counts with it, and a geometry shader whose
 *   vertex count is not constant, or a tessellated patch NGG culling repacked,
 *   drew nothing; with the v_msad_u8 fallback both draw every texel (the RADV
 *   smoke title's geometry and tessellation checks).
 * - No GFX10.3 wave ID in TG_SIZE: a 128-invocation workgroup of wave64
 *   reported one subgroup where two ran
 *   (dEQP-VK.subgroups.multiple_dispatches.uniform_subgroup_size); bits 20-24
 *   read 0, and the ordered wave ID serves instead.
 * - No per-primitive NGG parameters: a fragment shader reading gl_PrimitiveID
 *   that no earlier stage writes read 0 behind an NGG vertex shader
 *   (dEQP-VK.pipeline.*.misc.implicit_primitive_id; the smoke title's
 *   primitive ID check).
 * - No strict vertex order: the second of two triangles read its per-vertex
 *   inputs rotated with or without ROTATE_PC_PTR, so
 *   VK_KHR_fragment_shader_barycentric is not reported (the smoke title's
 *   barycentric pair; dEQP-VK.fragment_shading_barycentric.data).
 * - No variable-rate shading: a 2x2 pipeline rate ran the fragment shader once
 *   a texel and read gl_ShadingRateEXT as 1x1 (the smoke title's shading-rate
 *   check; dEQP-VK.fragment_shading_rate), so VK_KHR_fragment_shading_rate is
 *   not reported. No RB+ either.
 * - No E5B9G9R9 colour target: every blit into it with a non-zero colour read
 *   back wrong (dEQP-VK.api.copy_and_blit.*.blit_image.all_formats.color.*.
 *   e5b9g9r9_ufloat_pack32, 500 cases).
 * - The TC-compatible HTILE clear bug: a depth-only image cleared to 0 read back
 *   1 after a draw whose fragments were all discarded
 *   (dEQP-VK.dynamic_state.*.discard.depth); its workaround sets
 *   ZRANGE_PRECISION 0 after a clear to 0.
 * - GFX1013's AddrLib revision: non-RB+ GFX10 swizzles with the depth/stencil
 *   mipmap fix Navi10 lacks. With a Navi10 revision a 256x256 D16 image with
 *   mips was laid out in three 64 KiB blocks, its 128x128 level in the mip tail,
 *   while the depth block wrote level 0 into a fourth
 *   (dEQP-VK.glsl.texture_functions.texture.sampler2dshadow_*).
 *
 * GFX1013's model differs from GFX10.3 in more than these; the rest is not
 * taken until measured here (the audit lists them). */
static void
radv_ps5_take_gc_10_1_3_traits(struct radeon_info *info, bool compiler_compat_mode)
{
   struct radeon_info model;
   radv_ps5_fill_info(&model, compiler_compat_mode, 1, 0xa0103, RADV_PS5_GFX1013_REV);
   assert(model.family == CHIP_GFX1013);

   info->compiler_info.has_accelerated_dot_product = model.compiler_info.has_accelerated_dot_product;
   info->compiler_info.has_cs_wave_id = model.compiler_info.has_cs_wave_id;
   info->compiler_info.has_ngg_per_prim_params = model.compiler_info.has_ngg_per_prim_params;
   info->has_ps_strict_vertex_order = model.has_ps_strict_vertex_order;
   info->has_vrs = model.has_vrs;
   info->rbplus_allowed = model.rbplus_allowed;
   info->has_rgb9e5_color_target = model.has_rgb9e5_color_target;
   info->has_htile_tc_z_clear_bug_without_stencil = model.has_htile_tc_z_clear_bug_without_stencil;
   info->has_htile_tc_z_clear_bug_with_stencil = model.has_htile_tc_z_clear_bug_with_stencil;
   info->chip_external_rev = model.chip_external_rev;

   assert(!info->compiler_info.has_accelerated_dot_product && !info->compiler_info.has_cs_wave_id &&
          !info->compiler_info.has_ngg_per_prim_params && !info->has_ps_strict_vertex_order && !info->has_vrs &&
          !info->rbplus_allowed && !info->has_rgb9e5_color_target && info->has_htile_tc_z_clear_bug_without_stencil &&
          info->has_htile_tc_z_clear_bug_with_stencil && info->chip_external_rev == RADV_PS5_GFX1013_REV);
}

static void
radv_ps5_describe_gpu(struct radeon_info *info, bool compiler_compat_mode)
{
   radv_ps5_fill_info(info, compiler_compat_mode, 3, 0xa0300, RADV_PS5_EXTERNAL_REV);
   snprintf(info->marketing_name, sizeof(info->marketing_name), "PlayStation 5 GPU");
   radv_ps5_take_gc_10_1_3_traits(info, compiler_compat_mode);

   /* What the platform, not the chip, rules out. */
   /* Every wave of a shader using scratch_* instructions faulted (MEMVIOL) a
    * few instructions in, where it had set FLAT_SCRATCH with s_setreg; the
    * same shaders through buffer instructions, as on GFX8, run (the smoke
    * title's scratch checks, graphicsfuzz's large private arrays). */
   info->compiler_info.has_flat_scratch = false;
   /* The BVH instruction is GFX10.3's (and GC 10.1.3's): acceleration
    * structures build and ray queries traverse them (dEQP-VK.ray_query and
    * the ray query cases elsewhere, run rq-full-1 in PS5_Vulkan), so the
    * description keeps it. */

   /* A legacy GS hung the GPU every time (dEQP-VK.geometry with
    * RADV_DEBUG=nongg: 31 of 33 cases). Why is not established: RADV sets the
    * GS ring sizes with a UCONFIG write, as it does everywhere, while the
    * tessellation rings are AGC's (sceAgcDriverSetTFRing); whether the GS
    * ring sizes are too is unmeasured. */
   info->has_legacy_gs = false;
   /* An INDIRECT_BUFFER into title memory faulted the GPU (PS5_Vulkan B8), so
    * the command buffers device-generated commands write cannot run. */
   info->has_gpu_written_ibs = false;
   /* VideoOut scans out 64 KiB R_X tiles (tiling 0) with no pipe or bank
    * swizzle: ps5vk renders its framebuffers in that mode, and the demo
    * renderer's CPU-written tiles showed (PS5_Vulkan src/demo_renderer.cpp,
    * ps5platform/videoout.h). */
   info->has_fixed_display_swizzle = true;
   info->display_swizzle_mode = ADDR_SW_64KB_R_X;
   /* Every indirect mesh draw stopped the CP with a bad opcode at
    * DISPATCH_MESH_INDIRECT_MULTI (dEQP-VK.mesh_shader.ext.api.draw_indirect*,
    * run mesh-exp-1 in PS5_Vulkan); direct mesh draws, DRAW_INDEX_AUTO in
    * fast launch, run. Indirect ones go through ordinary indirect draws
    * (radv_mesh_draw_records_enabled). */
   info->has_dispatch_mesh_indirect = false;
   /* No exported function holds the GPU at a stable power state
    * (radv_ps5_ctx_set_pstate refuses every state but none), so the profiling
    * lock VK_KHR_performance_query takes always failed
    * (dEQP-VK.query_pool.performance_query.*: VK_ERROR_UNKNOWN). */
   info->has_perf_counters = false;
   /* One pool of memory the CPU and the GPU share, which RADV presents as an
    * integrated GPU's: a device-local heap and a host heap over it, with
    * host-cached types. The CPU's mapping of that memory is cached and the GPU
    * reads and writes it coherently with the CPU's caches (the RADV smoke
    * title's coherence probe). */
   info->has_dedicated_vram = false;
   info->all_vram_visible = true;
   info->address32_hi = RADV_PS5_ADDRESS32_HI;
   /* Timeline semaphores: the runtime builds them over the binary sync type
    * (vk_sync_timeline), with threaded submission for waits before signals. */
   info->has_timeline_syncobj = true;
   info->kernel_has_modifiers = false;
   /* Sparse resources are ranges of the device-memory region whose pages the
    * winsys maps to buffers as they are bound, and to a shared zero block
    * while they are not (radv_ps5_vrange_*). No exported function sets the
    * page table's PRT bits, so an unbound page is not known to read zero or
    * to report itself non-resident; and a bind is a mapping change made at
    * once, after the submission's waits. */
   info->has_sparse = true;
   /* A compute and transfer queue family, as upstream offers on this family
    * of GPUs, whose queues submit to the graphics ring through the same
    * sceAgcDriverSubmitDcb: their work runs in turn with graphics rather than
    * alongside it, which Vulkan allows. AGC's asynchronous compute queues
    * would run it alongside; they are not used yet. */
   info->num_compute_queues_on_gfx = 4;
   info->has_sparse_prt = false;
   info->has_sparse_cpu_binds = true;
   info->max_sparse_va_bytes = radv_ps5_vrange_space_bytes();
   /* A captured buffer can be placed at its address again
    * (radv_ps5_memory_alloc_replayable); before that, every replay case of
    * dEQP-VK.binding_model.buffer_device_address (723) failed. Outside the
    * shaders' window captures keep clear of every other buffer; in it the
    * kernel places buffers, so replayable ones (the ray tracing pipelines'
    * shader arenas, whose group handles are captured) go in a range at its
    * top the port keeps for them (radv_ps5_window_replayable). */
   info->has_replayable_va = true;
   info->has_replayable_va_32bit = radv_ps5_window_replayable();
   info->has_vm_always_valid = true;
   /* buffer_from_ptr gives the application's host memory GPU access, which the
    * GPU reaches at the CPU's address (HARDWARE_FINDINGS.md, 2026-09-27). */
   info->has_userptr = true;
   info->max_submitted_ibs[AMD_IP_GFX] = 1;

   /* ac_query_gpu_info's own derivations, which have no kernel input. */
   info->scratch_wavesize_granularity_shift = 10;
   info->scratch_wavesize_granularity = BITFIELD_BIT(info->scratch_wavesize_granularity_shift);
   const unsigned max_waves_per_tg = 32;
   info->max_scratch_waves =
      MAX2(32 * info->max_good_cu_per_sa * info->max_sa_per_se * info->num_se, max_waves_per_tg);
   info->has_scratch_base_registers = false;
   info->max_gflops = 128 * info->num_cu * info->max_gpu_freq_mhz / 1000;
   info->memory_bandwidth_gbps =
      DIV_ROUND_UP(info->memory_freq_mhz_effective * info->memory_bus_width / 8, 1000);
   info->instr_prefetch_distance = 3;
   info->se_tile_repeat = 32 * info->max_se;
}

void
radv_ps5_winsys_heap_usage(uint64_t *pool_bytes, uint64_t *available_bytes, uint64_t *driver_bytes)
{
   *pool_bytes = radv_ps5_memory_pool_bytes();
   *available_bytes = radv_ps5_memory_available_bytes();
   *driver_bytes = p_atomic_read(&radv_ps5_allocated_bytes);
}

VkResult
radv_ps5_winsys_query_info(uint64_t debug_flags, struct radeon_winsys_info *info)
{
   memset(info, 0, sizeof(*info));
   if (!radv_ps5_queue_get())
      return VK_ERROR_INITIALIZATION_FAILED;
   radv_ps5_describe_gpu(&info->base, !(debug_flags & RADV_DEBUG_NO_CACHE_COMPAT));
   info->syncobj_sync_type = radv_ps5_sync_type;
   /* One queue, which a title's submissions all reach: low and medium, the
    * priorities an application may always have, both run there. A priority is
    * a scheduling hint, and with one queue there is nothing to order. */
   info->global_priority_mask = BITFIELD_BIT(RADEON_CTX_PRIORITY_LOW) | BITFIELD_BIT(RADEON_CTX_PRIORITY_MEDIUM);
   /* GPU memory and the queue's syncs have no file descriptor to share. */
   info->has_external_fd = false;
   return VK_SUCCESS;
}

/* -------------------------------------------------------------- the winsys */

static uint64_t
radv_ps5_winsys_query_value(struct radeon_winsys *rws, enum radeon_value_id value)
{
   struct radv_ps5_winsys *ws = radv_ps5_winsys(rws);
   switch (value) {
   case RADEON_ALLOCATED_VRAM:
   case RADEON_ALLOCATED_VRAM_VIS:
   case RADEON_VRAM_USAGE:
   case RADEON_VRAM_VIS_USAGE:
      return p_atomic_read(&ws->allocated_vram);
   case RADEON_ALLOCATED_GTT:
   case RADEON_GTT_USAGE:
      return p_atomic_read(&ws->allocated_gtt);
   case RADEON_TIMESTAMP:
      return radv_ps5_read_gpu_clock();
   default:
      /* The console's other clocks and counters are not the title's to read. */
      return 0;
   }
}

static bool
radv_ps5_winsys_read_registers(struct radeon_winsys *rws, unsigned reg_offset, unsigned num_registers, uint32_t *out)
{
   (void)rws;
   (void)reg_offset;
   (void)num_registers;
   (void)out;
   return false;
}

static bool
radv_ps5_winsys_query_gpuvm_fault(struct radeon_winsys *rws, struct radv_winsys_gpuvm_fault_info *fault_info)
{
   (void)rws;
   (void)fault_info;
   return false;
}

static VkResult
radv_ps5_ctx_create(struct radeon_winsys *rws, enum radeon_ctx_priority priority, struct radeon_winsys_ctx **rctx)
{
   /* The priorities reported (global_priority_mask) are the ones a context is
    * made with; a higher one is denied, as the extension allows
    * (dEQP-VK.api.device_init.create_device_global_priority*). */
   if (priority != RADEON_CTX_PRIORITY_MEDIUM && priority != RADEON_CTX_PRIORITY_LOW)
      return VK_ERROR_NOT_PERMITTED;
   struct radv_ps5_ctx *const ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   ctx->ws = radv_ps5_winsys(rws);
   ctx->priority = priority;
   *rctx = (struct radeon_winsys_ctx *)ctx;
   return VK_SUCCESS;
}

static void
radv_ps5_ctx_set_tess_factor_ring(struct radeon_winsys_ctx *rctx, uint64_t va, uint32_t size)
{
   struct radv_ps5_ctx *const ctx = (struct radv_ps5_ctx *)rctx;
   ctx->tess_factor_ring_va = va;
   ctx->tess_factor_ring_size = size;
}

static void
radv_ps5_ctx_destroy(struct radeon_winsys_ctx *rctx)
{
   free(rctx);
}

static bool
radv_ps5_ctx_wait_idle(struct radeon_winsys_ctx *rctx, enum amd_ip_type ip_type, int ring_index)
{
   (void)ip_type;
   (void)ring_index;
   struct radv_ps5_queue *const queue = ((struct radv_ps5_ctx *)rctx)->ws->queue;
   return radv_ps5_queue_wait_seq(queue, p_atomic_read(&queue->submitted_seq), OS_TIMEOUT_INFINITE, "ctx_wait_idle");
}

static int
radv_ps5_ctx_set_pstate(struct radeon_winsys_ctx *rctx, uint32_t pstate)
{
   (void)rctx;
   return pstate == RADEON_CTX_PSTATE_NONE ? 0 : -1;
}

static void
radv_ps5_winsys_destroy(struct radeon_winsys *rws)
{
   free(rws);
}

static int
radv_ps5_winsys_get_fd(struct radeon_winsys *rws)
{
   (void)rws;
   return -1;
}

static struct util_sync_provider *
radv_ps5_winsys_get_sync_provider(struct radeon_winsys *rws)
{
   (void)rws;
   return NULL;
}

static int
radv_ps5_winsys_reserve_vmid(struct radeon_winsys *rws)
{
   (void)rws;
   return -1;
}

static void
radv_ps5_winsys_unreserve_vmid(struct radeon_winsys *rws)
{
   (void)rws;
}

VkResult
radv_ps5_winsys_create(const struct radeon_info *info, uint64_t debug_flags, uint64_t perftest_flags,
                       struct radeon_winsys **winsys)
{
   struct radv_ps5_queue *const queue = radv_ps5_queue_get();
   if (!queue)
      return VK_ERROR_INITIALIZATION_FAILED;
   struct radv_ps5_winsys *const ws = calloc(1, sizeof(*ws));
   if (!ws)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy(&ws->info, info, sizeof(ws->info));
   ws->debug_flags = debug_flags;
   ws->perftest_flags = perftest_flags;
   ws->queue = queue;

   ws->base.destroy = radv_ps5_winsys_destroy;
   ws->base.query_value = radv_ps5_winsys_query_value;
   ws->base.read_registers = radv_ps5_winsys_read_registers;
   ws->base.query_gpuvm_fault = radv_ps5_winsys_query_gpuvm_fault;
   ws->base.ctx_create = radv_ps5_ctx_create;
   ws->base.ctx_destroy = radv_ps5_ctx_destroy;
   ws->base.ctx_set_tess_factor_ring = radv_ps5_ctx_set_tess_factor_ring;
   ws->base.ctx_wait_idle = radv_ps5_ctx_wait_idle;
   ws->base.ctx_set_pstate = radv_ps5_ctx_set_pstate;
   ws->base.get_fd = radv_ps5_winsys_get_fd;
   ws->base.get_sync_provider = radv_ps5_winsys_get_sync_provider;
   ws->base.copy_sync_payloads = radv_ps5_sync_copy_payloads;
   ws->base.reserve_vmid = radv_ps5_winsys_reserve_vmid;
   ws->base.unreserve_vmid = radv_ps5_winsys_unreserve_vmid;
   radv_ps5_bo_init_functions(ws);
   radv_ps5_cs_init_functions(ws);

   *winsys = &ws->base;
   return VK_SUCCESS;
}
