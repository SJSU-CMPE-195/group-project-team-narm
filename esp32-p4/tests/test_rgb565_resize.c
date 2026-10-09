#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rgb565_resize.h"

static void test_padded_rows(void) {
  uint16_t input[6][8];
  uint16_t output[18];
  uint32_t columns[4];
  size_t row_offsets[4];
  const uint32_t sampled[4] = {0, 2, 3, 5};

  for (uint32_t y = 0; y < 6; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      input[y][x] = x < 6 ? y * 100 + x : 0xffff;
    }
  }
  for (size_t i = 0; i < 18; ++i) {
    output[i] = 0xbeef;
  }
  assert(rgb565_resize_prepare(6, 6, sizeof(input[0]), 4, 4, columns,
                               row_offsets));
  rgb565_resize_frame((const uint8_t *)input, output + 1, 6, 4, 4, columns,
                      row_offsets);
  assert(output[0] == 0xbeef && output[17] == 0xbeef);
  for (uint32_t y = 0; y < 4; ++y) {
    for (uint32_t x = 0; x < 4; ++x) {
      assert(output[1 + y * 4 + x] == sampled[y] * 100 + sampled[x]);
    }
  }
}

static void test_identity_and_colors(void) {
  /* Red, green, blue, white, black, and an arbitrary RGB565 value. */
  const uint16_t input[] = {0xf800, 0x07e0, 0x001f, 0xffff, 0x0000, 0x1234};
  uint16_t output[6];
  uint32_t columns[2];
  size_t row_offsets[3];
  assert(rgb565_resize_prepare(2, 3, 4, 2, 3, columns, row_offsets));
  rgb565_resize_frame((const uint8_t *)input, output, 2, 2, 3, columns,
                      row_offsets);
  assert(memcmp(input, output, sizeof(input)) == 0);
}

static void test_full_720p_frame(void) {
  uint16_t *input = malloc(1920 * 1080 * sizeof(uint16_t));
  uint16_t *output = malloc((1280 * 720 + 2) * sizeof(uint16_t));
  uint32_t *columns = malloc(1280 * sizeof(uint32_t));
  size_t *row_offsets = malloc(720 * sizeof(size_t));
  assert(input && output && columns && row_offsets);

  for (uint32_t y = 0; y < 1080; ++y) {
    for (uint32_t x = 0; x < 1920; ++x) {
      input[y * 1920 + x] = (uint16_t)(x ^ (y << 4));
    }
  }
  output[0] = output[1280 * 720 + 1] = 0xbeef;
  assert(rgb565_resize_prepare(1920, 1080, 1920 * 2, 1280, 720, columns,
                               row_offsets));
  rgb565_resize_frame((const uint8_t *)input, output + 1, 1920, 1280, 720,
                      columns, row_offsets);
  assert(output[0] == 0xbeef && output[1280 * 720 + 1] == 0xbeef);
  /* A 3:2 resize keeps the first and last pixels of each three-pixel group. */
  for (uint32_t y = 0; y < 720; ++y) {
    uint32_t source_y = 3 * (y / 2) + (y % 2 ? 2 : 0);
    for (uint32_t x = 0; x < 1280; ++x) {
      uint32_t source_x = 3 * (x / 2) + (x % 2 ? 2 : 0);
      assert(output[1 + y * 1280 + x] ==
             (uint16_t)(source_x ^ (source_y << 4)));
    }
  }
  assert(columns[1279] == 1919 && row_offsets[719] == 1079 * 1920 * 2);
  free(input);
  free(output);
  free(columns);
  free(row_offsets);
}

static void test_small_ratios(void) {
  /* Compare both paths with the center-sampling formula, including odd
   * widths, the two-pixel fast-path tail, padded rows, and output guards. */
  uint16_t input[7][25];
  uint16_t output[24 * 3 + 2];
  uint32_t columns[24];
  size_t row_offsets[3];
  for (uint32_t y = 0; y < 7; ++y) {
    for (uint32_t x = 0; x < 25; ++x) {
      input[y][x] = (uint16_t)(100 * y + x);
    }
  }
  for (uint32_t width = 1; width <= 24; ++width) {
    for (uint32_t out_width = 1; out_width <= width; ++out_width) {
      for (size_t i = 0; i < sizeof(output) / sizeof(output[0]); ++i) {
        output[i] = 0xbeef;
      }
      assert(rgb565_resize_prepare(width, 7, sizeof(input[0]), out_width, 3,
                                   columns, row_offsets));
      rgb565_resize_frame((const uint8_t *)input, output + 1, width, out_width,
                          3, columns, row_offsets);
      assert(output[0] == 0xbeef && output[1 + out_width * 3] == 0xbeef);
      for (uint32_t y = 0; y < 3; ++y) {
        uint32_t source_y = ((2 * y + 1) * 7) / 6;
        for (uint32_t x = 0; x < out_width; ++x) {
          uint32_t source_x = ((2 * x + 1) * width) / (2 * out_width);
          assert(output[1 + y * out_width + x] == input[source_y][source_x]);
        }
      }
    }
  }
}

static void test_invalid_dimensions(void) {
  uint32_t columns[4];
  size_t row_offsets[4];
  assert(!rgb565_resize_prepare(0, 6, 12, 4, 4, columns, row_offsets));
  assert(!rgb565_resize_prepare(6, 6, 12, 0, 4, columns, row_offsets));
  assert(!rgb565_resize_prepare(6, 6, 12, 7, 4, columns, row_offsets));
  assert(!rgb565_resize_prepare(6, 6, 12, 4, 7, columns, row_offsets));
  assert(!rgb565_resize_prepare(6, 6, 10, 4, 4, columns, row_offsets));
  assert(!rgb565_resize_prepare(6, 6, 13, 4, 4, columns, row_offsets));
  assert(
      !rgb565_resize_prepare(6, 6, SIZE_MAX - 1, 4, 4, columns, row_offsets));
  assert(!rgb565_resize_prepare(6, 6, 12, 4, 4, NULL, row_offsets));
}

int main(void) {
  test_padded_rows();
  test_identity_and_colors();
  test_full_720p_frame();
  test_small_ratios();
  test_invalid_dimensions();
  puts("RGB565 resize checks passed (including 1920x1080 -> 1280x720).");
  return 0;
}
