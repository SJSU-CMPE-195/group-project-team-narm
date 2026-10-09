#ifndef H264_MP4_H
#define H264_MP4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H264_MP4_MAX_PARAMETER_SET_SIZE 256u
#define H264_MP4_MAX_ANNEXB_SIZE (4u * 1024u * 1024u)
#define H264_MP4_MAX_NALS 1024u
#define H264_MP4_TIMESCALE 90000u
#define H264_MP4_FRAGMENT_OVERHEAD 108u

typedef struct {
  uint8_t sps[H264_MP4_MAX_PARAMETER_SET_SIZE];
  uint8_t pps[H264_MP4_MAX_PARAMETER_SET_SIZE];
  size_t sps_len;
  size_t pps_len;
  uint16_t width;
  uint16_t height;
  uint32_t timescale;
  char codec[12];
} h264_mp4_config_t;

/* SPS/PPS include their NAL headers, without start codes. Sets timescale to
 * 90000 and codec to avc1.PPCCLL. Failure leaves config unchanged. Accepts
 * repeated identical parameter sets; differing sets require reconfiguration
 * using a buffer containing a single pair. Validates framing/NAL headers,
 * not the complete H.264 bitstream syntax. Dimensions come from the caller. */
bool h264_mp4_configure(h264_mp4_config_t *config, const uint8_t *annexb,
                        size_t len, uint16_t width, uint16_t height);

/* False for malformed Annex-B, including malformed NALs after an IDR. */
bool h264_mp4_is_keyframe(const uint8_t *data, size_t len);

/* Return bytes written or zero on failure. Output is untouched on failure.
 * Input/config and output must not overlap; NULL output is not a size query. */
size_t h264_mp4_init_segment(const h264_mp4_config_t *config, uint8_t *out,
                             size_t capacity);

/* One complete access unit per call, with no B frames. DTS and positive
 * duration use the configured timescale (normally 90000); DTS equals PTS.
 * Strips SPS/PPS/AUD, retains all other supported AVC NALs including SEI.
 * A VCL NAL is required. Leading/trailing Annex-B zero bytes are discarded.
 * Required capacity is 108 + sum(4 + retained NAL length). No allocation. */
size_t h264_mp4_fragment(const uint8_t *annexb, size_t len, uint32_t sequence,
                         uint64_t dts, uint32_t duration, uint8_t *out,
                         size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
