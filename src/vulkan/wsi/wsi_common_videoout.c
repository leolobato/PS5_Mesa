/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * VK_KHR_display on the PlayStation 5's one display, VideoOut, in place of
 * the DRM backend (wsi_common_display.c), which the console does not have.
 *
 * The display has one plane and one 3840x2160 mode at 59.94 Hz, and a 119.88
 * Hz mode first where the title asks for it and the display refreshes at it
 * (measured, as PS5_Vulkan's ps5vk measured both). The title asks through
 * PS5_VIDEOOUT_REFRESH (60 or 120, or wsi_videoout_set_refresh before the
 * output opens) or, without it, by declaring high-frame-rate output in its
 * param.json (/app0's, or the one PS5_VIDEOOUT_PARAM_JSON names); VideoOut
 * itself grants the mode or refuses it.
 *
 * VideoOut's framebuffers are the process's: sets of five buffers of
 * direct memory, one set per swapchain size, each registered once, in the
 * 64 KiB R_X tiles VideoOut scans out (ps5platform/videoout.h). VideoOut
 * scaled a 1920x1080 set to the whole 3840x2160 output (Remote Play capture
 * of DXVK's D3D11 and D3D9 through Wine; ps5vk's 1080p buffers likewise), but
 * it takes framebuffers of a few sizes only (wsi_common_videoout_scale.h):
 * 1440x1080, 1440x960 and 1280x720 sets were refused with 0x80290005, and a
 * fourth set in one process with 0x80290001 (mihawk-99/PS5_Vulkan
 * HARDWARE_FINDINGS.md, 2026-10-04). A swapchain of a size it takes takes the
 * first buffers of its size's set as its images, imported into its device as
 * host memory (VK_EXT_external_memory_host) and laid out as the display takes them (the
 * device's display swizzle, radeon_info), so a device made again, as
 * RetroArch makes one for each core, presents where the last one did and the
 * screen keeps its last frame in between. A swapchain of any other size up to
 * the mode's renders into images of its own, and each present blits its image
 * on the presenting queue into a buffer of the smallest taken size that holds
 * it, scaled to fit with its aspect ratio kept and centred, the bars black.
 *
 * A present waits for the frame on a thread and flips it with
 * sceVideoOutSubmitFlip at the next vblank; FIFO is the only present mode. A
 * title may ask VideoOut to show each flip for two or three vblanks
 * (wsi_videoout_set_flip_rate), or name a frame rate (PS5_VIDEOOUT_FRAME_RATE,
 * wsi_videoout_set_frame_rate) that the output's rate, once known, turns into
 * vblanks per flip: at 119.88 Hz, 60 fps is two and 40 fps three. An
 * acquire takes a buffer that is neither on screen, nor waiting for its flip
 * to show, nor held by an application, the one flipped longest ago, and waits
 * for a flip or a vblank when none is.
 *
 * One swapchain presents at a time. A new one replaces it through
 * oldSwapchain, or takes the display from one that has never presented: a
 * window that only made a context, as wined3d's hidden 10x10 window probing
 * OpenGL does, must not keep the screen from the window made after it. The
 * one it takes the display from is retired, as an oldSwapchain is. A
 * swapchain that has presented keeps the display until it is destroyed or
 * replaced through oldSwapchain.
 */

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c11/threads.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_math.h"
#include "vk_device.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_util.h"
#include "wsi_common_entrypoints.h"
#include "wsi_common_private.h"
#include "wsi_common_videoout_scale.h"

#if defined(__PROSPERO__)
#include <immintrin.h>
#include <ps5platform/kernel.h>
#include <ps5platform/videoout.h>
#else
#include <sys/mman.h>
#endif

#define VIDEOOUT_WIDTH 3840
#define VIDEOOUT_HEIGHT 2160
#define VIDEOOUT_BUFFERS 5
/* Framebuffer sets one process may register: VideoOut refused a fourth
 * (0x80290001, buffers 15-19; mihawk-99/PS5_Vulkan HARDWARE_FINDINGS.md,
 * 2026-10-04). CPU frames take one, swapchains at the taken sizes the others. */
#define VIDEOOUT_SETS 3
/* Sizes VideoOut refused, not asked for again in the process. */
#define VIDEOOUT_REFUSED 8
/* A 64 KiB tile holds 128x128 four-byte pixels: a 3840x2160 image is 30x17
 * tiles, 31.9 MiB, a 1920x1080 one 15x9 tiles, 8.4 MiB. */
#define VIDEOOUT_TILE_PIXELS 128
#define VIDEOOUT_TILE_BYTES (UINT64_C(64) << 10)
#define VIDEOOUT_BUFFER_ALIGNMENT (UINT64_C(2) << 20)
#define VIDEOOUT_REFRESH_MILLIHERTZ 59940
#define VIDEOOUT_HIGH_REFRESH_MILLIHERTZ 119880
/* A period above this after the high-frame-rate mode was taken: the display
 * went on refreshing at 60 Hz, as a tester's did (PS5_Vulkan, R31). */
#define VIDEOOUT_HIGH_REFRESH_LIMIT_NS UINT64_C(12500000)
/* A period below this (or none, a vblank wait failing) was not measured: no
 * display refreshes faster than 119.88 Hz, every 8.342 ms. */
#define VIDEOOUT_HIGH_REFRESH_FLOOR_NS UINT64_C(7500000)
#define VIDEOOUT_PERIOD_INTERVALS 6
/* param.json's attribute3 bit for 120 Hz output. PS5_Vulkan R31/R92 declared it with 0x80000
 * (a 120 Hz mode that requires VRR), which turns the system's VRR off; retail titles with VRR
 * declare 0x40 with 0x40000. VideoOut itself grants mode 15 or refuses it. */
#define VIDEOOUT_ATTRIBUTE3_HIGH_FRAME_RATE 0x40u

#if !defined(__PROSPERO__)
/* The host model: flips show at once and a vblank comes every millisecond,
 * enough for the host builds to run the swapchain's logic. */
#define PS5_VIDEO_OUT_USER_SYSTEM 0xff
#define PS5_VIDEO_OUT_PIXEL_FORMAT_B8G8R8A8_SDR UINT64_C(0x8000000000000000)
#define PS5_VIDEO_OUT_TILING_64KB_R_X 0u
#define PS5_VIDEO_OUT_ATTRIBUTE_BYTES 80
#define PS5_VIDEO_OUT_FLIP_STATUS_WORDS 16
#define PS5_VIDEO_OUT_FLIP_STATUS_SHOWN_ARGUMENT 3
#define PS5_VIDEO_OUT_FLIP_VSYNC 1
#define PS5_VIDEO_OUT_MODE_HIGH_FRAME_RATE 15u
#define PS5_VIDEO_OUT_MODE_RESTORE 1u

struct ps5_video_out_buffer {
   void *data;
   void *metadata;
   void *reserved[2];
};

static int64_t host_shown_argument;

static int
sceVideoOutOpen(int32_t user, int32_t bus, int32_t index, const void *parameter)
{
   return 1;
}

static int
sceVideoOutSetFlipRate(int32_t handle, int32_t rate)
{
   return 0;
}

static uint32_t host_attribute_width, host_attribute_height;

static void
sceVideoOutSetBufferAttribute2(void *attribute, uint64_t pixel_format, uint32_t tiling, uint32_t width,
                               uint32_t height, uint64_t reserved1, uint32_t reserved2, uint64_t reserved3)
{
   host_attribute_width = width;
   host_attribute_height = height;
}

/* As measured on the console: framebuffers of the sizes VideoOut takes
 * register, others are refused as a resolution error. MESA_VK_VIDEOOUT_HOST_LOG
 * set prints each registration, for a host test to follow. */
static int
sceVideoOutRegisterBuffers2(int32_t handle, int32_t set, int32_t start, struct ps5_video_out_buffer *buffers,
                            int32_t count, void *attribute, int32_t reserved, void *option)
{
   const VkExtent2D extent = {host_attribute_width, host_attribute_height};
   const int result = wsi_videoout_size_taken(extent) ? 0 : (int)0x80290005u;
   if (getenv("MESA_VK_VIDEOOUT_HOST_LOG"))
      fprintf(stderr, "videoout-host: register set %d buffers %d-%d %ux%u: 0x%08x\n", set, start, start + count - 1,
              extent.width, extent.height, (unsigned)result);
   return result;
}

static int
sceVideoOutSubmitFlip(int32_t handle, int32_t buffer_index, uint32_t mode, int64_t argument)
{
   if (getenv("MESA_VK_VIDEOOUT_HOST_LOG"))
      fprintf(stderr, "videoout-host: flip buffer %d argument %" PRId64 "\n", buffer_index, argument);
   host_shown_argument = argument;
   return 0;
}

static int
sceVideoOutGetFlipStatus(int32_t handle, uint64_t status[PS5_VIDEO_OUT_FLIP_STATUS_WORDS])
{
   memset(status, 0, PS5_VIDEO_OUT_FLIP_STATUS_WORDS * sizeof(uint64_t));
   status[PS5_VIDEO_OUT_FLIP_STATUS_SHOWN_ARGUMENT] = host_shown_argument;
   return 0;
}

/* MESA_VK_VIDEOOUT_HOST_HIGH_VBLANK_US set: the output takes mode 15, and a
 * vblank then comes every that many microseconds, for a host test of the
 * 119.88 Hz checks (8342 a display refreshing at it, 16683 one left at 60 Hz,
 * 1000 one too fast to be a measurement). Unset, mode 15 is not supported. */
static unsigned host_vblank_us = 1000;

static unsigned
host_high_vblank_us(void)
{
   const char *const value = getenv("MESA_VK_VIDEOOUT_HOST_HIGH_VBLANK_US");
   return value != NULL ? (unsigned)strtoul(value, NULL, 10) : 0;
}

static int
sceVideoOutWaitVblank(int32_t handle)
{
   os_time_sleep(host_vblank_us);
   return 0;
}

static int
sceVideoOutIsOutputSupported(int32_t handle, uint32_t mode, const void *a, const void *b, const void *c)
{
   return host_high_vblank_us() != 0 ? 1 : 0;
}

static int
sceVideoOutConfigureOutput(int32_t handle, uint32_t mode, const void *a, const void *b, const void *c)
{
   if (host_high_vblank_us() == 0)
      return -1;
   host_vblank_us = mode == PS5_VIDEO_OUT_MODE_HIGH_FRAME_RATE ? host_high_vblank_us() : 1000;
   return 0;
}
#endif

/* The display and its modes: one each per process, as VideoOut is. */
struct wsi_videoout_mode {
   uint32_t refresh_millihertz;
   bool high_frame_rate;
};

struct wsi_videoout_display {
   int dummy;
};

static struct wsi_videoout_display videoout_display;
static struct wsi_videoout_mode videoout_mode = {.refresh_millihertz = VIDEOOUT_REFRESH_MILLIHERTZ};
static struct wsi_videoout_mode videoout_high_mode = {
   .refresh_millihertz = VIDEOOUT_HIGH_REFRESH_MILLIHERTZ,
   .high_frame_rate = true,
};

static VkDisplayKHR
videoout_display_handle(void)
{
   return (VkDisplayKHR)(uintptr_t)&videoout_display;
}

static VkDisplayModeKHR
videoout_mode_handle(struct wsi_videoout_mode *mode)
{
   return (VkDisplayModeKHR)(uintptr_t)mode;
}

static struct wsi_videoout_mode *
videoout_mode_from_handle(VkDisplayModeKHR mode)
{
   return (struct wsi_videoout_mode *)(uintptr_t)mode;
}

static const VkExtent2D videoout_extent = {VIDEOOUT_WIDTH, VIDEOOUT_HEIGHT};

struct wsi_videoout_swapchain;

/* A present waiting for its frame before its flip. */
struct wsi_videoout_present {
   struct wsi_videoout_swapchain *chain;
   uint32_t buffer;
   VkFence fence;
};

/* VideoOut, its framebuffers and what each holds, under lock. */
struct wsi_videoout_output {
   mtx_t lock;
   cnd_t changed;
   bool opened;
   int handle;
   bool high_frame_rate;
   bool modes_settled;
   bool high_mode_offered;
   /* The registered sets, one per swapchain size, set i as buffer indices
    * i * VIDEOOUT_BUFFERS onwards. */
   struct {
      VkExtent2D extent;
      /* CPU frames' own set, never a swapchain's (wsi_videoout_show_tiled). */
      bool cpu;
      uint64_t buffer_bytes;
      uint8_t *memory;
   } set[VIDEOOUT_SETS];
   uint32_t set_count;
   /* Sizes VideoOut refused: asked again, each would only be refused again. */
   VkExtent2D refused[VIDEOOUT_REFUSED];
   uint32_t refused_count;
   /* The argument of the latest flip submitted; each flip's is one more. */
   int64_t flip_argument;
   /* When a swapchain last flipped; CPU frames show only once it is idle. */
   uint64_t swapchain_flip_ns;
   /* The set CPU frames are copied into (wsi_videoout_show_tiled), and the
    * buffer of it written last; -1 before the first. */
   int32_t cpu_set;
   uint32_t cpu_buffer;
   struct {
      /* The argument this buffer was last flipped with, 0 before its first flip. */
      int64_t argument;
      /* An application holds it, acquired and not yet presented. */
      bool held;
      /* Presented, its flip waiting for the frame. */
      bool pending;
   } buffer[VIDEOOUT_SETS * VIDEOOUT_BUFFERS];
   /* The swapchain presenting now; older ones are retired. */
   struct wsi_videoout_swapchain *current;

   /* The flip thread and its queue. */
   bool thread_started;
   thrd_t thread;
   struct wsi_videoout_present queue[VIDEOOUT_BUFFERS * 2];
   uint32_t queue_head, queue_count;
};

static struct wsi_videoout_output videoout_output = {
   .handle = -1,
   .cpu_set = -1,
};
static once_flag videoout_once = ONCE_FLAG_INIT;
/* sceVideoOutSetFlipRate's rate for the output: a flip shows for at least
 * rate + 1 vblanks. Set with set 0's registration. Under the output's lock. */
static int videoout_flip_rate;
/* What wsi_videoout_set_refresh and wsi_videoout_set_frame_rate asked for
 * before the output opened, in place of PS5_VIDEOOUT_REFRESH and
 * PS5_VIDEOOUT_FRAME_RATE; -1 for nothing. Under the output's lock. */
static int videoout_refresh_asked = -1;
static int videoout_frame_rate_asked = -1;

static void
videoout_output_init_once(void)
{
   mtx_init(&videoout_output.lock, mtx_plain);
   cnd_init(&videoout_output.changed);
}

struct wsi_videoout_swapchain {
   struct wsi_swapchain base;
   /* Its set: its images are, or are blitted into, buffers first to
    * first + image_count - 1. */
   uint32_t set;
   uint32_t first;
   /* Its images are its own, each present blitted into rect of its buffer,
    * whose size is target (wsi_common_videoout_scale.h). */
   bool scaled;
   VkExtent2D target;
   VkRect2D rect;
   bool retired;
   /* It queued a present: the display is its until it is destroyed or
    * replaced through oldSwapchain. */
   bool presented;
   /* Presents queued and not yet flipped. */
   uint32_t in_flight;
   struct wsi_image images[];
};

struct wsi_videoout {
   struct wsi_interface base;
};

/* --- VideoOut ------------------------------------------------------------ */

/* The title's param.json: /app0's, or the one PS5_VIDEOOUT_PARAM_JSON names, for a title
 * without /app0 (an elevated process does not see it). */
static bool
videoout_title_declares_high_frame_rate(void)
{
   const char *const path = getenv("PS5_VIDEOOUT_PARAM_JSON");
   FILE *const file = fopen(path != NULL && path[0] != '\0' ? path : "/app0/sce_sys/param.json", "rb");
   if (file == NULL)
      return false;
   char text[16384];
   const size_t length = fread(text, 1, sizeof(text) - 1, file);
   fclose(file);
   text[length] = '\0';
   const char *at = strstr(text, "\"attribute3\"");
   if (at == NULL || (at = strchr(at, ':')) == NULL)
      return false;
   const unsigned long value = strtoul(at + 1, NULL, 0);
   return (value & VIDEOOUT_ATTRIBUTE3_HIGH_FRAME_RATE) == VIDEOOUT_ATTRIBUTE3_HIGH_FRAME_RATE;
}

/* The refresh period: the median of a few intervals between vblank waits,
 * the first wait finding the phase. 0 when a wait fails. */
static uint64_t
videoout_vblank_period_ns(int handle)
{
   uint64_t intervals[VIDEOOUT_PERIOD_INTERVALS];
   if (sceVideoOutWaitVblank(handle) != 0)
      return 0;
   uint64_t previous = os_time_get_nano();
   for (unsigned at = 0; at < VIDEOOUT_PERIOD_INTERVALS; at++) {
      if (sceVideoOutWaitVblank(handle) != 0)
         return 0;
      const uint64_t now = os_time_get_nano();
      intervals[at] = now - previous;
      previous = now;
   }
   for (unsigned i = 1; i < VIDEOOUT_PERIOD_INTERVALS; i++)
      for (unsigned j = i; j > 0 && intervals[j - 1] > intervals[j]; j--) {
         const uint64_t swap = intervals[j];
         intervals[j] = intervals[j - 1];
         intervals[j - 1] = swap;
      }
   return intervals[VIDEOOUT_PERIOD_INTERVALS / 2];
}

/* A frame rate the output can be paced to: 0 (no cap), 120, 60, 40 or 30. */
static bool
videoout_frame_rate_valid(int fps)
{
   return fps == 0 || fps == 120 || fps == 60 || fps == 40 || fps == 30;
}

/* A switch from the environment as a number: -1 unset or empty, -2 not a
 * number. */
static int
videoout_env_number(const char *name)
{
   const char *const value = getenv(name);
   if (value == NULL || value[0] == '\0')
      return -1;
   char *end;
   const long number = strtol(value, &end, 10);
   return *end == '\0' && number >= 0 && number <= 1000 ? (int)number : -2;
}

/* The flip rate that paces fps frames a second at the output's rate: each
 * flip shows for output / fps vblanks where that is one, two or three
 * (119.88 Hz: 120, 60 and 40 fps; 59.94 Hz: 60 and 30 fps). Any other frame
 * rate is not capped, so a display that refused 119.88 Hz never turns 60 fps
 * into 30 or 40 into 20. 0 fps is no cap. Under the output's lock. */
static int
videoout_frame_rate_flip_rate_locked(unsigned fps)
{
   const unsigned hz = videoout_output.high_frame_rate ? 120 : 60;
   const char *const rate = videoout_output.high_frame_rate ? "119.88" : "59.94";
   if (fps == 0)
      return 0;
   if (hz % fps == 0 && hz / fps <= 3) {
      const int vblanks = (int)(hz / fps);
      fprintf(stderr, "wsi/videoout: %u fps paced as each flip for %d vblanks at %s Hz\n", fps, vblanks, rate);
      return vblanks - 1;
   }
   fprintf(stderr, "wsi/videoout: %u fps cannot be paced at %s Hz; not capped\n", fps, rate);
   return 0;
}

/* Opens VideoOut once per process, in the high-frame-rate mode where the
 * title asks for it, the output takes it and the display then refreshes at
 * it, and paces it to the title's frame rate. The title asks with
 * PS5_VIDEOOUT_REFRESH (or wsi_videoout_set_refresh): 60 never, 120 without
 * reading param.json, unset where param.json declares it. Under lock. */
static bool
videoout_open_locked(void)
{
   struct wsi_videoout_output *out = &videoout_output;
   if (out->opened)
      return out->handle >= 0;
   out->opened = true;
   out->handle = sceVideoOutOpen(PS5_VIDEO_OUT_USER_SYSTEM, 0, 0, NULL);
   if (out->handle < 0) {
      fprintf(stderr, "wsi/videoout: sceVideoOutOpen failed: 0x%08x\n", (unsigned)out->handle);
      return false;
   }
   int refresh = videoout_refresh_asked;
   if (refresh < 0) {
      refresh = videoout_env_number("PS5_VIDEOOUT_REFRESH");
      if (refresh == -2 || (refresh >= 0 && refresh != 60 && refresh != 120)) {
         fprintf(stderr, "wsi/videoout: PS5_VIDEOOUT_REFRESH=%s ignored (60 or 120)\n",
                 getenv("PS5_VIDEOOUT_REFRESH"));
         refresh = -1;
      }
   }
   if (refresh == 60) {
      fprintf(stderr, "wsi/videoout: 119.88 Hz not asked for (PS5_VIDEOOUT_REFRESH=60)\n");
   } else if (refresh == 120 || videoout_title_declares_high_frame_rate()) {
      const int supported =
         sceVideoOutIsOutputSupported(out->handle, PS5_VIDEO_OUT_MODE_HIGH_FRAME_RATE, NULL, NULL, NULL);
      const int configured =
         supported > 0
            ? sceVideoOutConfigureOutput(out->handle, PS5_VIDEO_OUT_MODE_HIGH_FRAME_RATE, NULL, NULL, NULL)
            : supported;
      /* Not supported (0) configures nothing: it is not an acceptance. */
      out->high_frame_rate = supported > 0 && configured == 0;
      if (out->high_frame_rate) {
         /* A period of 0 (a vblank wait failed) or one no display has is no
          * measurement: the mode is not kept unmeasured. */
         const uint64_t period = videoout_vblank_period_ns(out->handle);
         if (period < VIDEOOUT_HIGH_REFRESH_FLOOR_NS || period > VIDEOOUT_HIGH_REFRESH_LIMIT_NS) {
            sceVideoOutConfigureOutput(out->handle, PS5_VIDEO_OUT_MODE_RESTORE, NULL, NULL, NULL);
            out->high_frame_rate = false;
            if (period > VIDEOOUT_HIGH_REFRESH_LIMIT_NS)
               fprintf(stderr,
                       "wsi/videoout: 119.88 Hz accepted, but a vblank comes every %.3f ms; 59.94 Hz restored\n",
                       (double)period / 1e6);
            else
               fprintf(stderr, "wsi/videoout: 119.88 Hz accepted, but the vblank period could not be measured; "
                               "59.94 Hz restored\n");
         } else {
            fprintf(stderr, "wsi/videoout: 119.88 Hz output, a vblank every %.3f ms\n", (double)period / 1e6);
         }
      } else {
         fprintf(stderr, "wsi/videoout: 119.88 Hz %s (%d); presenting at 59.94 Hz\n",
                 supported > 0 ? "refused" : "not supported", configured);
      }
   }

   /* The frame rate, now that the output's rate is known: set 0's
    * registration sets its flip rate. */
   int frame_rate = videoout_frame_rate_asked;
   if (frame_rate < 0) {
      frame_rate = videoout_env_number("PS5_VIDEOOUT_FRAME_RATE");
      if (frame_rate == -2 || (frame_rate >= 0 && !videoout_frame_rate_valid(frame_rate))) {
         fprintf(stderr, "wsi/videoout: PS5_VIDEOOUT_FRAME_RATE=%s ignored (0, 120, 60, 40 or 30)\n",
                 getenv("PS5_VIDEOOUT_FRAME_RATE"));
         frame_rate = -1;
      }
   }
   if (frame_rate >= 0)
      videoout_flip_rate = videoout_frame_rate_flip_rate_locked((unsigned)frame_rate);
   return true;
}

PUBLIC int wsi_videoout_set_refresh(int hz);
PUBLIC int wsi_videoout_set_flip_rate(int rate);
PUBLIC int wsi_videoout_set_frame_rate(int fps);

/* The title's switch in place of PS5_VIDEOOUT_REFRESH, for a title whose
 * environment the driver does not see: 60 never offers 119.88 Hz, 120 offers
 * it where VideoOut grants it without reading param.json, 0 follows
 * param.json. Only before the output opens, as listing the display's modes
 * opens it. Returns 0, -1 for a rate it does not take, -2 once the output is
 * open. */
PUBLIC int
wsi_videoout_set_refresh(int hz)
{
   if (hz != 0 && hz != 60 && hz != 120)
      return -1;
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&videoout_output.lock);
   const bool opened = videoout_output.opened;
   if (!opened)
      videoout_refresh_asked = hz;
   mtx_unlock(&videoout_output.lock);
   return opened ? -2 : 0;
}

/* Frame pacing for the title: each flip shows for at least rate + 1 vblanks
 * (0 every vblank, 1 every second, 2 every third), on the open output at once
 * and on one opened later. A frame rate asked for (PS5_VIDEOOUT_FRAME_RATE,
 * wsi_videoout_set_frame_rate) replaces it when the output opens; after
 * that, the later call wins. Returns VideoOut's answer, 0 before the output's
 * framebuffers are registered, -1 for a rate it does not take. */
PUBLIC int
wsi_videoout_set_flip_rate(int rate)
{
   if (rate < 0 || rate > 2)
      return -1;
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&videoout_output.lock);
   videoout_flip_rate = rate;
   const int result = videoout_output.handle >= 0 && videoout_output.set_count > 0
                         ? sceVideoOutSetFlipRate(videoout_output.handle, rate)
                         : 0;
   mtx_unlock(&videoout_output.lock);
   return result;
}

/* Frame pacing by a frame rate in place of PS5_VIDEOOUT_FRAME_RATE, for a
 * title whose environment the driver does not see: 0 (no cap), 120, 60, 40
 * or 30, turned into a flip rate by the output's rate
 * (videoout_frame_rate_flip_rate_locked) when the output opens, or at once on
 * an open one. Returns VideoOut's answer, 0 before the output's framebuffers
 * are registered, -1 for a frame rate it does not take. */
PUBLIC int
wsi_videoout_set_frame_rate(int fps)
{
   if (!videoout_frame_rate_valid(fps))
      return -1;
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&videoout_output.lock);
   int result = 0;
   if (videoout_output.handle >= 0) {
      videoout_flip_rate = videoout_frame_rate_flip_rate_locked((unsigned)fps);
      if (videoout_output.set_count > 0)
         result = sceVideoOutSetFlipRate(videoout_output.handle, videoout_flip_rate);
   } else {
      videoout_frame_rate_asked = fps;
   }
   mtx_unlock(&videoout_output.lock);
   return result;
}

static void
videoout_settle_modes(void)
{
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&videoout_output.lock);
   if (!videoout_output.modes_settled) {
      videoout_output.modes_settled = true;
      videoout_output.high_mode_offered = videoout_open_locked() && videoout_output.high_frame_rate;
   }
   mtx_unlock(&videoout_output.lock);
}

/* The bytes of one buffer of the given size: whole tiles, 2 MiB aligned. */
static uint64_t
videoout_buffer_bytes(VkExtent2D extent)
{
   const uint64_t tiles = (uint64_t)DIV_ROUND_UP(extent.width, VIDEOOUT_TILE_PIXELS) *
                          DIV_ROUND_UP(extent.height, VIDEOOUT_TILE_PIXELS);
   return align64(tiles * VIDEOOUT_TILE_BYTES, VIDEOOUT_BUFFER_ALIGNMENT);
}

/* The set of buffers of a swapchain size, cleared and registered the first
 * time a swapchain takes that size; a set stays registered for the process,
 * as a buffer of it may be on screen. Under lock. */
static VkResult
videoout_register_kind_locked(VkExtent2D extent, bool cpu, uint32_t *set_out)
{
   struct wsi_videoout_output *out = &videoout_output;
   for (uint32_t i = 0; i < out->set_count; i++) {
      if (out->set[i].extent.width == extent.width && out->set[i].extent.height == extent.height &&
          out->set[i].cpu == cpu) {
         *set_out = i;
         return VK_SUCCESS;
      }
   }
   for (uint32_t i = 0; i < out->refused_count; i++)
      if (wsi_videoout_extent_equal(out->refused[i], extent))
         return VK_ERROR_INITIALIZATION_FAILED;
   if (out->set_count == VIDEOOUT_SETS) {
      fprintf(stderr, "wsi/videoout: no set left for a %ux%u swapchain (%u sizes registered)\n", extent.width,
              extent.height, out->set_count);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   if (!videoout_open_locked())
      return VK_ERROR_INITIALIZATION_FAILED;

   const uint32_t set = out->set_count;
   const uint64_t buffer_bytes = videoout_buffer_bytes(extent);
   const size_t bytes = VIDEOOUT_BUFFERS * buffer_bytes;
   uint8_t *buffers = NULL;
#if defined(__PROSPERO__)
   int64_t physical = -1;
   if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, VIDEOOUT_BUFFER_ALIGNMENT,
                                     PS5_KERNEL_DIRECT_TYPE_CPU, &physical) != 0)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   void *address = NULL;
   if (sceKernelMapDirectMemory(&address, bytes,
                                PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE | PS5_KERNEL_PROT_GPU_READ |
                                   PS5_KERNEL_PROT_GPU_WRITE,
                                0, physical, VIDEOOUT_BUFFER_ALIGNMENT) != 0) {
      sceKernelReleaseDirectMemory(physical, bytes);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   buffers = address;
   /* The display reads memory, not the CPU's cache. */
   memset(buffers, 0, bytes);
   for (size_t at = 0; at < bytes; at += 64)
      _mm_clflush(buffers + at);
   _mm_mfence();
#else
   buffers = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (buffers == MAP_FAILED)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
#endif

   int result = set == 0 ? sceVideoOutSetFlipRate(out->handle, videoout_flip_rate) : 0;
   struct ps5_video_out_buffer registered[VIDEOOUT_BUFFERS];
   for (uint32_t i = 0; i < VIDEOOUT_BUFFERS; i++)
      registered[i] = (struct ps5_video_out_buffer){.data = buffers + i * buffer_bytes};
   uint8_t attribute[PS5_VIDEO_OUT_ATTRIBUTE_BYTES] = {0};
   sceVideoOutSetBufferAttribute2(attribute, PS5_VIDEO_OUT_PIXEL_FORMAT_B8G8R8A8_SDR,
                                  PS5_VIDEO_OUT_TILING_64KB_R_X, extent.width, extent.height, 0, 0, 0);
   if (result == 0)
      result = sceVideoOutRegisterBuffers2(out->handle, (int32_t)set, (int32_t)(set * VIDEOOUT_BUFFERS), registered,
                                           VIDEOOUT_BUFFERS, attribute, 0, NULL);
   if (result != 0) {
      fprintf(stderr, "wsi/videoout: the %ux%u framebuffers (set %u) were not registered: 0x%08x\n", extent.width,
              extent.height, set, (unsigned)result);
      /* Nothing was registered, so nothing can flip these buffers: their
       * memory goes back. Kept, each refusal leaked a set's worth, and a
       * game asking again every frame (DXVK does) took all of the process's
       * direct memory in seconds. The size is not asked for again. */
#if defined(__PROSPERO__)
      sceKernelMunmap(buffers, bytes);
      sceKernelReleaseDirectMemory(physical, bytes);
#else
      munmap(buffers, bytes);
#endif
      if (out->refused_count < VIDEOOUT_REFUSED)
         out->refused[out->refused_count++] = extent;
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   out->set[set].extent = extent;
   out->set[set].cpu = cpu;
   out->set[set].buffer_bytes = buffer_bytes;
   out->set[set].memory = buffers;
   out->set_count++;
   *set_out = set;
   return VK_SUCCESS;
}

static VkResult
videoout_register_locked(VkExtent2D extent, uint32_t *set_out)
{
   return videoout_register_kind_locked(extent, false, set_out);
}

static int64_t
videoout_shown_argument_locked(void)
{
   uint64_t status[PS5_VIDEO_OUT_FLIP_STATUS_WORDS] = {0};
   if (videoout_output.flip_argument == 0 || sceVideoOutGetFlipStatus(videoout_output.handle, status) != 0)
      return 0;
   return (int64_t)status[PS5_VIDEO_OUT_FLIP_STATUS_SHOWN_ARGUMENT];
}

/* The flip thread: each present's flip once its frame is done, in order. */
static int
videoout_flip_thread(void *data)
{
   struct wsi_videoout_output *out = &videoout_output;
   mtx_lock(&out->lock);
   for (;;) {
      while (out->queue_count == 0)
         cnd_wait(&out->changed, &out->lock);
      const struct wsi_videoout_present present = out->queue[out->queue_head];
      mtx_unlock(&out->lock);

      struct wsi_videoout_swapchain *chain = present.chain;
      const struct wsi_device *wsi = chain->base.wsi;
      wsi->WaitForFences(chain->base.device, 1, &present.fence, VK_TRUE, UINT64_MAX);

      mtx_lock(&out->lock);
      const int64_t argument = out->flip_argument + 1;
      const int result = sceVideoOutSubmitFlip(out->handle, (int32_t)present.buffer, PS5_VIDEO_OUT_FLIP_VSYNC,
                                               argument);
      if (result == 0) {
         out->flip_argument = argument;
         out->buffer[present.buffer].argument = argument;
         out->swapchain_flip_ns = os_time_get_nano();
      } else {
         fprintf(stderr, "wsi/videoout: sceVideoOutSubmitFlip(%u) failed: 0x%08x\n", present.buffer,
                 (unsigned)result);
      }
      out->buffer[present.buffer].pending = false;
      out->queue_head = (out->queue_head + 1) % ARRAY_SIZE(out->queue);
      out->queue_count--;
      chain->in_flight--;
      cnd_broadcast(&out->changed);
   }
   return 0;
}

/* --- CPU frames ------------------------------------------------------------ */

/* Whether a CPU frame would show now: the display is open and no swapchain
 * has a flip queued or has flipped within VIDEOOUT_CPU_IDLE_NS. */
#define VIDEOOUT_CPU_IDLE_NS UINT64_C(150000000)

static bool
videoout_cpu_idle_locked(void)
{
   const struct wsi_videoout_output *out = &videoout_output;
   return out->opened && out->handle >= 0 && out->queue_count == 0 &&
          os_time_get_nano() - out->swapchain_flip_ns >= VIDEOOUT_CPU_IDLE_NS;
}

bool
wsi_videoout_idle(void)
{
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&videoout_output.lock);
   const bool idle = videoout_cpu_idle_locked();
   mtx_unlock(&videoout_output.lock);
   return idle;
}

/* Where the presenting swapchain's images show: *frame is the size of the
 * framebuffers flipped, *rect the part of them an image fills (all of it but
 * for a scaled swapchain). For the hardware cursor, whose position is in the
 * flipped framebuffer's pixels. false while no swapchain presents. */
bool
wsi_videoout_present_rect(VkRect2D *rect, VkExtent2D *frame)
{
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&videoout_output.lock);
   const struct wsi_videoout_swapchain *chain = videoout_output.current;
   if (chain) {
      *rect = chain->rect;
      *frame = chain->target;
   }
   mtx_unlock(&videoout_output.lock);
   return chain != NULL;
}

/* Shows a frame the CPU drew, already in VideoOut's 64 KiB tiling (as a
 * title's own presenter writes it) and in B8G8R8A8 as the swapchains are:
 * VideoOut scales a smaller set of that format to the mode, not one of the
 * title's R8G8B8A8 format, which it shows unscaled at the top left
 * (measured). While no swapchain is presenting: a
 * game's GDI content, such as a DirectShow movie drawn outside Direct3D, in
 * between its Vulkan frames. The next swapchain present takes the screen
 * back. Returns 0 when flipped, 1 when a swapchain is presenting, <0 on
 * failure. */
int
wsi_videoout_show_tiled(const void *tiled, uint64_t bytes, uint32_t width, uint32_t height)
{
   struct wsi_videoout_output *out = &videoout_output;
   if (!tiled || !width || !height)
      return -1;
   call_once(&videoout_once, videoout_output_init_once);
   mtx_lock(&out->lock);
   if (!videoout_cpu_idle_locked()) {
      mtx_unlock(&out->lock);
      return 1;
   }
   uint32_t set;
   if (out->cpu_set >= 0 && out->set[out->cpu_set].extent.width == width &&
       out->set[out->cpu_set].extent.height == height) {
      set = (uint32_t)out->cpu_set;
   } else if (videoout_register_kind_locked((VkExtent2D){width, height}, true, &set) == VK_SUCCESS) {
      out->cpu_set = (int32_t)set;
   } else {
      mtx_unlock(&out->lock);
      return -2;
   }
   /* Round robin over the set: the buffer written was last on screen
    * VIDEOOUT_BUFFERS - 1 flips ago. */
   out->cpu_buffer = (out->cpu_buffer + 1) % VIDEOOUT_BUFFERS;
   const uint32_t buffer = set * VIDEOOUT_BUFFERS + out->cpu_buffer;
   uint8_t *target = out->set[set].memory + (uint64_t)out->cpu_buffer * out->set[set].buffer_bytes;
   const uint64_t copied = MIN2(bytes, out->set[set].buffer_bytes);
   memcpy(target, tiled, copied);
#if defined(__PROSPERO__)
   /* The display reads memory, not the CPU's cache. */
   for (uint64_t at = 0; at < copied; at += 64)
      _mm_clflush(target + at);
   _mm_mfence();
#endif
   const int64_t argument = out->flip_argument + 1;
   const int result = sceVideoOutSubmitFlip(out->handle, (int32_t)buffer, PS5_VIDEO_OUT_FLIP_VSYNC, argument);
   if (result == 0) {
      out->flip_argument = argument;
      out->buffer[buffer].argument = argument;
   }
   mtx_unlock(&out->lock);
   return result == 0 ? 0 : -3;
}

/* --- VK_KHR_display -------------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetPhysicalDeviceDisplayPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                          VkDisplayPropertiesKHR *pProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkDisplayPropertiesKHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayPropertiesKHR, &out, prop)
   {
      *prop = (VkDisplayPropertiesKHR){
         .display = videoout_display_handle(),
         .displayName = "PS5 VideoOut",
         /* The size of the screen VideoOut drives is not known. */
         .physicalDimensions = {0, 0},
         .physicalResolution = videoout_extent,
         .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
         .planeReorderPossible = VK_FALSE,
         .persistentContent = VK_FALSE,
      };
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetPhysicalDeviceDisplayProperties2KHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                           VkDisplayProperties2KHR *pProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkDisplayProperties2KHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayProperties2KHR, &out, prop)
   {
      uint32_t count = 1;
      wsi_GetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &count, &prop->displayProperties);
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetPhysicalDeviceDisplayPlanePropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                               VkDisplayPlanePropertiesKHR *pProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkDisplayPlanePropertiesKHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayPlanePropertiesKHR, &out, prop)
   {
      *prop = (VkDisplayPlanePropertiesKHR){
         .currentDisplay = videoout_display_handle(),
         .currentStackIndex = 0,
      };
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetPhysicalDeviceDisplayPlaneProperties2KHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                VkDisplayPlaneProperties2KHR *pProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkDisplayPlaneProperties2KHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayPlaneProperties2KHR, &out, prop)
   {
      uint32_t count = 1;
      wsi_GetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &count, &prop->displayPlaneProperties);
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetDisplayPlaneSupportedDisplaysKHR(VkPhysicalDevice physicalDevice, uint32_t planeIndex,
                                        uint32_t *pDisplayCount, VkDisplayKHR *pDisplays)
{
   VK_OUTARRAY_MAKE_TYPED(VkDisplayKHR, out, pDisplays, pDisplayCount);
   if (planeIndex == 0) {
      vk_outarray_append_typed(VkDisplayKHR, &out, display)
         *display = videoout_display_handle();
   }
   return vk_outarray_status(&out);
}

/* The fastest mode first: an application taking the first mode gets 119.88
 * Hz where it is offered. */
static uint32_t
videoout_modes(struct wsi_videoout_mode *modes[2])
{
   videoout_settle_modes();
   uint32_t count = 0;
   if (videoout_output.high_mode_offered)
      modes[count++] = &videoout_high_mode;
   modes[count++] = &videoout_mode;
   return count;
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetDisplayModePropertiesKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display, uint32_t *pPropertyCount,
                                VkDisplayModePropertiesKHR *pProperties)
{
   struct wsi_videoout_mode *modes[2];
   const uint32_t mode_count = videoout_modes(modes);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayModePropertiesKHR, out, pProperties, pPropertyCount);
   for (uint32_t i = 0; i < mode_count; i++) {
      vk_outarray_append_typed(VkDisplayModePropertiesKHR, &out, prop)
      {
         *prop = (VkDisplayModePropertiesKHR){
            .displayMode = videoout_mode_handle(modes[i]),
            .parameters = {.visibleRegion = videoout_extent, .refreshRate = modes[i]->refresh_millihertz},
         };
      }
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetDisplayModeProperties2KHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display, uint32_t *pPropertyCount,
                                 VkDisplayModeProperties2KHR *pProperties)
{
   struct wsi_videoout_mode *modes[2];
   const uint32_t mode_count = videoout_modes(modes);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayModeProperties2KHR, out, pProperties, pPropertyCount);
   for (uint32_t i = 0; i < mode_count; i++) {
      vk_outarray_append_typed(VkDisplayModeProperties2KHR, &out, prop)
      {
         prop->displayModeProperties = (VkDisplayModePropertiesKHR){
            .displayMode = videoout_mode_handle(modes[i]),
            .parameters = {.visibleRegion = videoout_extent, .refreshRate = modes[i]->refresh_millihertz},
         };
      }
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_CreateDisplayModeKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                         const VkDisplayModeCreateInfoKHR *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                         VkDisplayModeKHR *pMode)
{
   /* VideoOut's modes are the ones reported and it makes no others: one
    * asked for with a reported mode's parameters is that mode. */
   struct wsi_videoout_mode *modes[2];
   const uint32_t mode_count = videoout_modes(modes);
   const VkDisplayModeParametersKHR *parameters = &pCreateInfo->parameters;
   for (uint32_t i = 0; i < mode_count; i++) {
      if (parameters->visibleRegion.width == videoout_extent.width &&
          parameters->visibleRegion.height == videoout_extent.height &&
          parameters->refreshRate == modes[i]->refresh_millihertz) {
         *pMode = videoout_mode_handle(modes[i]);
         return VK_SUCCESS;
      }
   }
   return VK_ERROR_INITIALIZATION_FAILED;
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetDisplayPlaneCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode, uint32_t planeIndex,
                                   VkDisplayPlaneCapabilitiesKHR *pCapabilities)
{
   *pCapabilities = (VkDisplayPlaneCapabilitiesKHR){
      .supportedAlpha = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
      .minSrcPosition = {0, 0},
      .maxSrcPosition = {0, 0},
      /* VideoOut scales a smaller source to the whole mode. */
      .minSrcExtent = {1, 1},
      .maxSrcExtent = videoout_extent,
      .minDstPosition = {0, 0},
      .maxDstPosition = {0, 0},
      .minDstExtent = videoout_extent,
      .maxDstExtent = videoout_extent,
   };
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_GetDisplayPlaneCapabilities2KHR(VkPhysicalDevice physicalDevice,
                                    const VkDisplayPlaneInfo2KHR *pDisplayPlaneInfo,
                                    VkDisplayPlaneCapabilities2KHR *pCapabilities)
{
   return wsi_GetDisplayPlaneCapabilitiesKHR(physicalDevice, pDisplayPlaneInfo->mode, pDisplayPlaneInfo->planeIndex,
                                             &pCapabilities->capabilities);
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_CreateDisplayPlaneSurfaceKHR(VkInstance _instance, const VkDisplaySurfaceCreateInfoKHR *pCreateInfo,
                                 const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR);

   VkIcdSurfaceDisplay *surface =
      vk_zalloc2(&instance->alloc, pAllocator, sizeof(*surface), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (surface == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   surface->base.platform = VK_ICD_WSI_PLATFORM_DISPLAY;
   surface->displayMode = pCreateInfo->displayMode;
   surface->planeIndex = pCreateInfo->planeIndex;
   surface->planeStackIndex = pCreateInfo->planeStackIndex;
   surface->transform = pCreateInfo->transform;
   surface->globalAlpha = pCreateInfo->globalAlpha;
   surface->alphaMode = pCreateInfo->alphaMode;
   surface->imageExtent = pCreateInfo->imageExtent;

   *pSurface = VkIcdSurfaceBase_to_handle(&surface->base);
   return VK_SUCCESS;
}

/* --- Surfaces ------------------------------------------------------------ */

static VkResult
videoout_surface_get_support(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device, uint32_t queueFamilyIndex,
                             VkBool32 *pSupported)
{
   /* Flips come from the CPU once a frame is done, from any queue's work. */
   *pSupported = VK_TRUE;
   return VK_SUCCESS;
}

static VkResult
videoout_surface_get_capabilities2(VkIcdSurfaceBase *icd_surface, struct wsi_device *wsi_device,
                                   const void *info_next, VkSurfaceCapabilities2KHR *caps)
{
   VkSurfaceCapabilitiesKHR *const c = &caps->surfaceCapabilities;
   const VkSurfacePresentModeKHR *present_mode = vk_find_struct_const(info_next, SURFACE_PRESENT_MODE_KHR);

   c->minImageCount = 2;
   c->maxImageCount = VIDEOOUT_BUFFERS;
   /* The mode's size, and any smaller one, which VideoOut stretches to it. */
   c->currentExtent = videoout_extent;
   c->minImageExtent = (VkExtent2D){1, 1};
   c->maxImageExtent = videoout_extent;
   c->maxImageArrayLayers = 1;
   c->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   c->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   c->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

   VkImageUsageFlags image_usage = wsi_caps_get_image_usage();
   VkImageUsageFlags2CreateInfoKHR *usage2 = vk_find_struct(caps->pNext, IMAGE_USAGE_FLAGS_2_CREATE_INFO_KHR);
   if (usage2)
      usage2->usage = image_usage;
   else
      c->supportedUsageFlags = image_usage;

   vk_foreach_struct (ext, caps->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR: {
         VkSurfaceProtectedCapabilitiesKHR *protected = (void *)ext;
         protected->supportsProtected = VK_FALSE;
         break;
      }
      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_KHR: {
         VkSurfacePresentScalingCapabilitiesKHR *scaling = (void *)ext;
         /* An image of a size VideoOut takes fills the output; any other
          * is scaled to fit it with its aspect ratio kept, centred: the
          * window Wine stretches a display mode's image to has the mode's
          * shape. */
         scaling->supportedPresentScaling =
            VK_PRESENT_SCALING_STRETCH_BIT_KHR | VK_PRESENT_SCALING_ASPECT_RATIO_STRETCH_BIT_KHR;
         scaling->supportedPresentGravityX = VK_PRESENT_GRAVITY_CENTERED_BIT_KHR;
         scaling->supportedPresentGravityY = VK_PRESENT_GRAVITY_CENTERED_BIT_KHR;
         scaling->minScaledImageExtent = c->minImageExtent;
         scaling->maxScaledImageExtent = c->maxImageExtent;
         break;
      }
      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_KHR: {
         /* FIFO only. */
         VkSurfacePresentModeCompatibilityKHR *compat = (void *)ext;
         if (compat->pPresentModes) {
            if (compat->presentModeCount) {
               assert(present_mode);
               compat->pPresentModes[0] = present_mode->presentMode;
               compat->presentModeCount = 1;
            }
         } else {
            compat->presentModeCount = 1;
         }
         break;
      }
      default:
         break;
      }
   }
   return VK_SUCCESS;
}

static const VkFormat videoout_formats[] = {
   VK_FORMAT_B8G8R8A8_UNORM,
   VK_FORMAT_B8G8R8A8_SRGB,
};

static VkResult
videoout_surface_get_formats(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device, uint32_t *pSurfaceFormatCount,
                             VkSurfaceFormatKHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (unsigned i = 0; i < ARRAY_SIZE(videoout_formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f)
      {
         f->format = videoout_formats[i];
         f->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   return vk_outarray_status(&out);
}

static VkResult
videoout_surface_get_formats2(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device, const void *info_next,
                              uint32_t *pSurfaceFormatCount, VkSurfaceFormat2KHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (unsigned i = 0; i < ARRAY_SIZE(videoout_formats); i++) {
      vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, f)
      {
         assert(f->sType == VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR);
         f->surfaceFormat.format = videoout_formats[i];
         f->surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   return vk_outarray_status(&out);
}

static VkResult
videoout_surface_get_present_modes(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                                   uint32_t *pPresentModeCount, VkPresentModeKHR *pPresentModes)
{
   VK_OUTARRAY_MAKE_TYPED(VkPresentModeKHR, out, pPresentModes, pPresentModeCount);
   vk_outarray_append_typed(VkPresentModeKHR, &out, mode)
      *mode = VK_PRESENT_MODE_FIFO_KHR;
   return vk_outarray_status(&out);
}

static VkResult
videoout_surface_get_present_rectangles(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                                        uint32_t *pRectCount, VkRect2D *pRects)
{
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);
   vk_outarray_append_typed(VkRect2D, &out, rect)
      *rect = (VkRect2D){.offset = {0, 0}, .extent = videoout_extent};
   return vk_outarray_status(&out);
}

/* --- Images -------------------------------------------------------------- */

/* Framebuffer index of a set, imported into the swapchain's device as host
 * memory for an image laid out as the display scans out. */
static VkResult
videoout_import_framebuffer(const struct wsi_swapchain *wsi_chain, VkImage vk_image, uint32_t set, uint32_t index,
                            VkDeviceMemory *memory)
{
   const struct wsi_device *wsi = wsi_chain->wsi;
   const uint64_t buffer_bytes = videoout_output.set[set].buffer_bytes;
   void *const pointer = videoout_output.set[set].memory + index * buffer_bytes;

   VkMemoryRequirements reqs;
   wsi->GetImageMemoryRequirements(wsi_chain->device, vk_image, &reqs);
   if (reqs.size > buffer_bytes || reqs.alignment > VIDEOOUT_BUFFER_ALIGNMENT) {
      fprintf(stderr, "wsi/videoout: a %" PRIu64 "-byte image does not fit a %" PRIu64 "-byte framebuffer\n",
              (uint64_t)reqs.size, buffer_bytes);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   VkMemoryHostPointerPropertiesEXT host_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
   };
   VkResult result = wsi->GetMemoryHostPointerPropertiesEXT(
      wsi_chain->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, pointer, &host_props);
   if (result != VK_SUCCESS)
      return result;

   const uint32_t type_bits = reqs.memoryTypeBits & host_props.memoryTypeBits;
   if (!type_bits)
      return VK_ERROR_INITIALIZATION_FAILED;

   const VkImportMemoryHostPointerInfoEXT import = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
      .pHostPointer = pointer,
   };
   const VkMemoryAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import,
      .allocationSize = buffer_bytes,
      .memoryTypeIndex = wsi_select_memory_type(wsi, 0, 0, type_bits),
   };
   return wsi->AllocateMemory(wsi_chain->device, &allocate, &wsi_chain->alloc, memory);
}

/* A swapchain image: the framebuffer of its index, imported as host memory. */
static VkResult
videoout_create_image_mem(const struct wsi_swapchain *wsi_chain, const struct wsi_image_info *info,
                          struct wsi_image *image)
{
   const struct wsi_videoout_swapchain *chain = (const struct wsi_videoout_swapchain *)wsi_chain;
   return videoout_import_framebuffer(wsi_chain, image->image, chain->set, image - chain->images, &image->memory);
}

/* A scaled swapchain's image: the application's own, in device memory. */
static VkResult
videoout_create_scaled_image_mem(const struct wsi_swapchain *wsi_chain, const struct wsi_image_info *info,
                                 struct wsi_image *image)
{
   const struct wsi_device *wsi = wsi_chain->wsi;
   VkMemoryRequirements reqs;
   wsi->GetImageMemoryRequirements(wsi_chain->device, image->image, &reqs);
   const VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image->image,
   };
   const VkMemoryAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &dedicated,
      .allocationSize = reqs.size,
      .memoryTypeIndex = wsi_select_device_memory_type(wsi, reqs.memoryTypeBits),
   };
   return wsi->AllocateMemory(wsi_chain->device, &allocate, &wsi_chain->alloc, &image->memory);
}

/* The present's blit of a scaled swapchain's image into its framebuffer: the
 * bars cleared to black where the image does not cover the framebuffer, the
 * image scaled into its rectangle, and both back in the layout a present
 * leaves them in. */
static void
videoout_record_scaled_blit(const struct wsi_videoout_swapchain *chain, const struct wsi_image *image,
                            VkCommandBuffer cmd)
{
   const struct wsi_device *wsi = chain->base.wsi;
   const VkExtent3D source = chain->base.image_info.create.extent;
   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
   };
   VkImageMemoryBarrier barriers[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = image->image,
         .subresourceRange = range,
      },
      {
         /* The framebuffer's last contents are not needed: all of it is
          * written. */
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = image->blit.image,
         .subresourceRange = range,
      },
   };
   wsi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0,
                           NULL, 2, barriers);

   if (!wsi_videoout_rect_covers(chain->rect, chain->target)) {
      const VkClearColorValue black = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
      wsi->CmdClearColorImage(cmd, image->blit.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
      const VkImageMemoryBarrier cleared = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = image->blit.image,
         .subresourceRange = range,
      };
      wsi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0,
                              NULL, 1, &cleared);
   }

   const VkImageBlit region = {
      .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .srcOffsets = {{0, 0, 0}, {(int32_t)source.width, (int32_t)source.height, 1}},
      .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .dstOffsets = {{chain->rect.offset.x, chain->rect.offset.y, 0},
                     {chain->rect.offset.x + (int32_t)chain->rect.extent.width,
                      chain->rect.offset.y + (int32_t)chain->rect.extent.height, 1}},
   };
   /* A copy where the image keeps its size, filtered where it is scaled. */
   const bool same_size = source.width == chain->rect.extent.width && source.height == chain->rect.extent.height;
   wsi->CmdBlitImage(cmd, image->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image->blit.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                     same_size ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);

   barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   barriers[0].dstAccessMask = 0;
   barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   barriers[0].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
   barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   barriers[1].dstAccessMask = 0;
   barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   barriers[1].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
   wsi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0,
                           NULL, 2, barriers);
}

/* After a scaled swapchain's image is made: the framebuffer of its index as
 * the blit's destination (image->blit.image and .memory, which
 * wsi_destroy_image frees), and the blit recorded for each queue family that
 * may present. A blit scales only on a graphics queue; a family without one
 * presents nothing new (RADV on the PS5 has only the graphics family). */
static VkResult
videoout_finish_create_scaled(const struct wsi_swapchain *wsi_chain, const struct wsi_image_info *info,
                              struct wsi_image *image)
{
   const struct wsi_videoout_swapchain *chain = (const struct wsi_videoout_swapchain *)wsi_chain;
   const struct wsi_device *wsi = wsi_chain->wsi;
   const uint32_t index = image - chain->images;
   VkResult result;

   /* A separate blit queue would be a transfer engine, which cannot scale. */
   if (wsi_chain->blit.queue != NULL)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
   };
   const struct wsi_image_create_info scanout = {
      .sType = VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA,
      .pNext = &external,
      .scanout = true,
   };
   const VkImageCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &scanout,
      .flags = VK_IMAGE_CREATE_ALIAS_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = info->create.format,
      .extent = {chain->target.width, chain->target.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   result = wsi->CreateImage(wsi_chain->device, &create, &wsi_chain->alloc, &image->blit.image);
   if (result != VK_SUCCESS)
      return result;
   result = videoout_import_framebuffer(wsi_chain, image->blit.image, chain->set, index, &image->blit.memory);
   if (result != VK_SUCCESS)
      return result;
   result = wsi->BindImageMemory(wsi_chain->device, image->blit.image, image->blit.memory, 0);
   if (result != VK_SUCCESS)
      return result;

   VkQueueFamilyProperties families[64];
   uint32_t family_count = ARRAY_SIZE(families);
   wsi->GetPhysicalDeviceQueueFamilyProperties(wsi->pdevice, &family_count, families);

   image->blit.cmd_buffers =
      vk_zalloc(&wsi_chain->alloc, sizeof(VkCommandBuffer) * wsi->queue_family_count, 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image->blit.cmd_buffers)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   for (uint32_t i = 0; i < wsi->queue_family_count; i++) {
      if (!wsi_chain->cmd_pools[i])
         continue;
      const VkCommandBufferAllocateInfo allocate = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = wsi_chain->cmd_pools[i],
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      result = wsi->AllocateCommandBuffers(wsi_chain->device, &allocate, &image->blit.cmd_buffers[i]);
      if (result != VK_SUCCESS)
         return result;
      const VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      };
      wsi->BeginCommandBuffer(image->blit.cmd_buffers[i], &begin);
      if (i < family_count && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
         videoout_record_scaled_blit(chain, image, image->blit.cmd_buffers[i]);
      result = wsi->EndCommandBuffer(image->blit.cmd_buffers[i]);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VkResult
wsi_videoout_configure_image(const struct wsi_swapchain *chain, const VkSwapchainCreateInfoKHR *pCreateInfo,
                             const struct wsi_videoout_image_params *params, struct wsi_image_info *info)
{
   if (params->scaled) {
      /* The application's own images, read by the present's blit. */
      VkResult result = wsi_configure_image(chain, pCreateInfo, 0, info);
      if (result != VK_SUCCESS)
         return result;
      info->create.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      info->usage2.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      info->create_mem = videoout_create_scaled_image_mem;
      info->finish_create = videoout_finish_create_scaled;
      return VK_SUCCESS;
   }

   VkResult result =
      wsi_configure_image(chain, pCreateInfo, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, info);
   if (result != VK_SUCCESS)
      return result;

   /* Laid out as the display scans out. */
   info->wsi.scanout = true;
   info->create_mem = videoout_create_image_mem;
   return VK_SUCCESS;
}

/* --- Swapchains ---------------------------------------------------------- */

static struct wsi_image *
videoout_swapchain_get_wsi_image(struct wsi_swapchain *wsi_chain, uint32_t image_index)
{
   struct wsi_videoout_swapchain *chain = (struct wsi_videoout_swapchain *)wsi_chain;
   return &chain->images[image_index];
}

/* A buffer of first to first + count - 1 an application may render into, under lock:
 * neither held, nor waiting for its flip, nor flipped past the flip shown
 * last, nor on screen; the one flipped longest ago. UINT32_MAX when none is,
 * with *vblank set when only a flip still to show holds one. */
static uint32_t
videoout_free_buffer_locked(uint32_t first, uint32_t count, bool *vblank)
{
   struct wsi_videoout_output *out = &videoout_output;
   const int64_t shown = videoout_shown_argument_locked();

   /* On screen: the buffer of the latest flip shown. */
   uint32_t on_screen = UINT32_MAX;
   for (uint32_t i = 0; i < ARRAY_SIZE(out->buffer); i++)
      if (out->buffer[i].argument != 0 && out->buffer[i].argument <= shown &&
          (on_screen == UINT32_MAX || out->buffer[i].argument > out->buffer[on_screen].argument))
         on_screen = i;

   uint32_t best = UINT32_MAX;
   *vblank = false;
   for (uint32_t i = first; i < first + count; i++) {
      if (out->buffer[i].held || out->buffer[i].pending || i == on_screen)
         continue;
      if (out->buffer[i].argument > shown) {
         *vblank = true;
         continue;
      }
      if (best == UINT32_MAX || out->buffer[i].argument < out->buffer[best].argument)
         best = i;
   }
   return best;
}

static VkResult
videoout_swapchain_acquire_next_image(struct wsi_swapchain *wsi_chain, const VkAcquireNextImageInfoKHR *info,
                                      uint32_t *image_index)
{
   struct wsi_videoout_swapchain *chain = (struct wsi_videoout_swapchain *)wsi_chain;
   struct wsi_videoout_output *out = &videoout_output;
   const uint64_t deadline = os_time_get_absolute_timeout(info->timeout);

   mtx_lock(&out->lock);
   for (;;) {
      if (chain->retired) {
         mtx_unlock(&out->lock);
         return VK_ERROR_OUT_OF_DATE_KHR;
      }
      bool vblank;
      const uint32_t index = videoout_free_buffer_locked(chain->first, chain->base.image_count, &vblank);
      if (index != UINT32_MAX) {
         out->buffer[index].held = true;
         mtx_unlock(&out->lock);
         *image_index = index - chain->first;
         return VK_SUCCESS;
      }
      if (info->timeout == 0) {
         mtx_unlock(&out->lock);
         return VK_NOT_READY;
      }
      if (os_time_get_nano() >= deadline) {
         mtx_unlock(&out->lock);
         return VK_TIMEOUT;
      }
      if (vblank) {
         /* A flip still to show frees one at a vblank. */
         const int handle = out->handle;
         mtx_unlock(&out->lock);
         sceVideoOutWaitVblank(handle);
         mtx_lock(&out->lock);
      } else {
         /* A flip waiting for its frame, or an image held: wait for either. */
         struct timespec until;
         const uint64_t wake = MIN2(deadline, os_time_get_nano() + UINT64_C(20000000));
         until.tv_sec = wake / 1000000000;
         until.tv_nsec = wake % 1000000000;
         cnd_timedwait(&out->changed, &out->lock, &until);
      }
   }
}

static VkResult
videoout_swapchain_release_images(struct wsi_swapchain *wsi_chain, uint32_t count, const uint32_t *indices)
{
   struct wsi_videoout_swapchain *chain = (struct wsi_videoout_swapchain *)wsi_chain;
   struct wsi_videoout_output *out = &videoout_output;
   mtx_lock(&out->lock);
   for (uint32_t i = 0; i < count; i++)
      out->buffer[chain->first + indices[i]].held = false;
   cnd_broadcast(&out->changed);
   mtx_unlock(&out->lock);
   return VK_SUCCESS;
}

static VkResult
videoout_swapchain_queue_present(struct wsi_swapchain *wsi_chain, uint32_t image_index, uint64_t present_id,
                                 const VkPresentRegionKHR *damage)
{
   struct wsi_videoout_swapchain *chain = (struct wsi_videoout_swapchain *)wsi_chain;
   struct wsi_videoout_output *out = &videoout_output;

   const uint32_t buffer = chain->first + image_index;
   mtx_lock(&out->lock);
   out->buffer[buffer].held = false;
   if (chain->retired) {
      cnd_broadcast(&out->changed);
      mtx_unlock(&out->lock);
      return VK_ERROR_OUT_OF_DATE_KHR;
   }
   if (!out->thread_started) {
      if (thrd_create(&out->thread, videoout_flip_thread, NULL) != thrd_success) {
         mtx_unlock(&out->lock);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      out->thread_started = true;
   }
   assert(out->queue_count < ARRAY_SIZE(out->queue));
   out->queue[(out->queue_head + out->queue_count) % ARRAY_SIZE(out->queue)] = (struct wsi_videoout_present){
      .chain = chain,
      .buffer = buffer,
      .fence = chain->base.fences[image_index],
   };
   out->queue_count++;
   out->buffer[buffer].pending = true;
   chain->in_flight++;
   chain->presented = true;
   cnd_broadcast(&out->changed);
   mtx_unlock(&out->lock);
   return VK_SUCCESS;
}

static VkResult
videoout_swapchain_wait_for_present(struct wsi_swapchain *wsi_chain, uint64_t present_id, uint64_t timeout)
{
   return wsi_swapchain_wait_for_present_semaphore(wsi_chain, present_id, timeout);
}

static VkResult
videoout_swapchain_destroy(struct wsi_swapchain *wsi_chain, const VkAllocationCallbacks *pAllocator)
{
   struct wsi_videoout_swapchain *chain = (struct wsi_videoout_swapchain *)wsi_chain;
   struct wsi_videoout_output *out = &videoout_output;

   /* Its flips use its fences: they go first. */
   mtx_lock(&out->lock);
   while (chain->in_flight)
      cnd_wait(&out->changed, &out->lock);
   for (uint32_t i = 0; i < chain->base.image_count; i++)
      if (chain->images[i].acquired)
         out->buffer[chain->first + i].held = false;
   if (out->current == chain)
      out->current = NULL;
   cnd_broadcast(&out->changed);
   mtx_unlock(&out->lock);

   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->images[i].image != VK_NULL_HANDLE)
         wsi_destroy_image(&chain->base, &chain->images[i]);
   }
   wsi_swapchain_finish(&chain->base);
   vk_free(pAllocator, chain);
   return VK_SUCCESS;
}

/* Whether a new swapchain replacing old may take the display: nothing has it,
 * old has it, or what has it never presented. */
static bool
videoout_display_free_locked(const struct wsi_videoout_swapchain *old)
{
   const struct wsi_videoout_swapchain *current = videoout_output.current;
   return !current || current == old || !current->presented;
}

static VkResult
videoout_surface_create_swapchain(VkIcdSurfaceBase *icd_surface, VkDevice device, struct wsi_device *wsi_device,
                                  const VkSwapchainCreateInfoKHR *pCreateInfo,
                                  const VkAllocationCallbacks *pAllocator, struct wsi_swapchain **swapchain_out)
{
   VkIcdSurfaceDisplay *surface = (VkIcdSurfaceDisplay *)icd_surface;
   struct wsi_videoout_output *out = &videoout_output;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
   const VkExtent2D extent = pCreateInfo->imageExtent;
   if (extent.width == 0 || extent.height == 0 || extent.width > VIDEOOUT_WIDTH || extent.height > VIDEOOUT_HEIGHT ||
       pCreateInfo->minImageCount > VIDEOOUT_BUFFERS || !wsi_device->GetMemoryHostPointerPropertiesEXT)
      return VK_ERROR_INITIALIZATION_FAILED;

   /* The output in the surface's mode: a mode other than the output's was not
    * offered. */
   videoout_settle_modes();
   const struct wsi_videoout_mode *mode = videoout_mode_from_handle(surface->displayMode);
   if (mode->high_frame_rate && !out->high_mode_offered)
      return VK_ERROR_INITIALIZATION_FAILED;

   /* A size VideoOut takes is its framebuffers; any other is blitted into
    * framebuffers of the smallest taken size that holds it. */
   const VkExtent2D target = wsi_videoout_target_extent(extent);
   const bool scaled = !wsi_videoout_extent_equal(target, extent);

   uint32_t set = 0;
   mtx_lock(&out->lock);
   struct wsi_videoout_swapchain *old =
      (struct wsi_videoout_swapchain *)wsi_swapchain_from_handle(pCreateInfo->oldSwapchain);
   VkResult result = videoout_register_locked(target, &set);
   if (result == VK_SUCCESS && !videoout_display_free_locked(old))
      result = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   mtx_unlock(&out->lock);
   if (result != VK_SUCCESS)
      return result;

   const uint32_t num_images = MAX2(pCreateInfo->minImageCount, 3);
   const size_t size = sizeof(struct wsi_videoout_swapchain) + num_images * sizeof(struct wsi_image);
   struct wsi_videoout_swapchain *chain = vk_zalloc(pAllocator, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (chain == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct wsi_videoout_image_params image_params = {
      .base.image_type = WSI_IMAGE_TYPE_VIDEOOUT,
      .scaled = scaled,
   };
   result = wsi_swapchain_init(wsi_device, &chain->base, device, pCreateInfo, &image_params.base, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, chain);
      return result;
   }

   chain->base.destroy = videoout_swapchain_destroy;
   chain->base.get_wsi_image = videoout_swapchain_get_wsi_image;
   chain->base.acquire_next_image = videoout_swapchain_acquire_next_image;
   chain->base.release_images = videoout_swapchain_release_images;
   chain->base.queue_present = videoout_swapchain_queue_present;
   chain->base.wait_for_present = videoout_swapchain_wait_for_present;
   chain->base.present_mode = VK_PRESENT_MODE_FIFO_KHR;
   chain->base.image_count = num_images;
   chain->set = set;
   chain->first = set * VIDEOOUT_BUFFERS;
   chain->scaled = scaled;
   chain->target = target;
   chain->rect = scaled ? wsi_videoout_fit_rect(extent, target) : (VkRect2D){.extent = extent};
   if (scaled)
      fprintf(stderr,
              "wsi/videoout: a %ux%u swapchain presents scaled to %ux%u at (%d, %d) of %ux%u framebuffers (set %u)\n",
              extent.width, extent.height, chain->rect.extent.width, chain->rect.extent.height, chain->rect.offset.x,
              chain->rect.offset.y, target.width, target.height, set);

   for (uint32_t i = 0; i < num_images; i++) {
      result = wsi_create_image(&chain->base, &chain->base.image_info, &chain->images[i]);
      if (result != VK_SUCCESS) {
         /* wsi_create_image destroyed the one that failed. */
         for (uint32_t j = 0; j < i; j++)
            wsi_destroy_image(&chain->base, &chain->images[j]);
         wsi_swapchain_finish(&chain->base);
         vk_free(pAllocator, chain);
         return result;
      }
   }

   /* What had the display is retired: its acquires and presents are out of
    * date. It may have presented since the check above. */
   mtx_lock(&out->lock);
   if (!videoout_display_free_locked(old)) {
      mtx_unlock(&out->lock);
      for (uint32_t i = 0; i < num_images; i++)
         wsi_destroy_image(&chain->base, &chain->images[i]);
      wsi_swapchain_finish(&chain->base);
      vk_free(pAllocator, chain);
      return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   }
   if (out->current) {
      if (out->current != old)
         fprintf(stderr, "wsi/videoout: a swapchain that never presented gives the display to a new one\n");
      out->current->retired = true;
   }
   out->current = chain;
   mtx_unlock(&out->lock);

   *swapchain_out = &chain->base;
   return VK_SUCCESS;
}

/* --- The interface ------------------------------------------------------- */

VkResult
wsi_display_init_wsi(struct wsi_device *wsi_device, const VkAllocationCallbacks *alloc, int display_fd)
{
   struct wsi_videoout *wsi = vk_zalloc(alloc, sizeof(*wsi), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!wsi) {
      wsi_device->wsi[VK_ICD_WSI_PLATFORM_DISPLAY] = NULL;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   call_once(&videoout_once, videoout_output_init_once);

   wsi->base.get_support = videoout_surface_get_support;
   wsi->base.get_capabilities2 = videoout_surface_get_capabilities2;
   wsi->base.get_formats = videoout_surface_get_formats;
   wsi->base.get_formats2 = videoout_surface_get_formats2;
   wsi->base.get_present_modes = videoout_surface_get_present_modes;
   wsi->base.get_present_rectangles = videoout_surface_get_present_rectangles;
   wsi->base.create_swapchain = videoout_surface_create_swapchain;

   wsi_device->wsi[VK_ICD_WSI_PLATFORM_DISPLAY] = &wsi->base;
   return VK_SUCCESS;
}

void
wsi_display_finish_wsi(struct wsi_device *wsi_device, const VkAllocationCallbacks *alloc)
{
   struct wsi_videoout *wsi = (struct wsi_videoout *)wsi_device->wsi[VK_ICD_WSI_PLATFORM_DISPLAY];
   if (wsi)
      vk_free(alloc, wsi);
}

void
wsi_display_setup_syncobj_fd(struct wsi_device *wsi_device, int fd)
{
}
