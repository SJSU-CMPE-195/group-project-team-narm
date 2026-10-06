#include "rgb565_resize.h"

bool rgb565_resize_prepare(uint32_t input_width, uint32_t input_height,
                           size_t input_stride, uint32_t output_width,
                           uint32_t output_height, uint32_t *columns,
                           size_t *row_offsets)
{
    if (!input_width || !input_height || !output_width || !output_height ||
        !columns || !row_offsets || output_width > input_width ||
        output_height > input_height || input_stride % sizeof(uint16_t) ||
        input_width > SIZE_MAX / sizeof(uint16_t) ||
        input_stride < (size_t)input_width * sizeof(uint16_t) ||
        input_stride > SIZE_MAX / input_height) {
        return false;
    }

    /* Sample at pixel centers over the whole image, without cropping.
     * Precompute the mapping so the frame loop needs no division. */
    for (uint32_t x = 0; x < output_width; ++x) {
        columns[x] = ((2ULL * x + 1) * input_width) / (2ULL * output_width);
    }
    for (uint32_t y = 0; y < output_height; ++y) {
        uint32_t input_y = ((2ULL * y + 1) * input_height) / (2ULL * output_height);
        row_offsets[y] = (size_t)input_y * input_stride;
    }
    return true;
}

void rgb565_resize_frame(const uint8_t *input, uint16_t *output,
                         uint32_t input_width, uint32_t output_width,
                         uint32_t output_height,
                         const uint32_t *columns, const size_t *row_offsets)
{
    if (output_width % 2 == 0 && 2ULL * input_width == 3ULL * output_width) {
        /* Exact 3:2 center sampling: keep pixels 0 and 2 in each triplet.
         * Avoid the per-pixel map loads for 1920x1080 -> 1280x720. Halfword
         * accesses also work when rows/output are not four-byte aligned. */
        for (uint32_t y = 0; y < output_height; ++y) {
            const uint16_t *row = (const uint16_t *)(input + row_offsets[y]);
            uint32_t remaining = output_width;
            while (remaining >= 4) {
                output[0] = row[0];
                output[1] = row[2];
                output[2] = row[3];
                output[3] = row[5];
                row += 6;
                output += 4;
                remaining -= 4;
            }
            if (remaining) {
                output[0] = row[0];
                output[1] = row[2];
                output += 2;
            }
        }
        return;
    }

    for (uint32_t y = 0; y < output_height; ++y) {
        const uint16_t *row = (const uint16_t *)(input + row_offsets[y]);
        for (uint32_t x = 0; x < output_width; ++x) {
            *output++ = row[columns[x]];
        }
    }
}
