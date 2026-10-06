#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Caller supplies output_width columns and output_height row offsets. */
bool rgb565_resize_prepare(uint32_t input_width, uint32_t input_height,
                           size_t input_stride, uint32_t output_width,
                           uint32_t output_height, uint32_t *columns,
                           size_t *row_offsets);

/* Use the same dimensions passed to prepare. The output is packed RGB565;
 * input rows may have padding. Input/output must not overlap. */
void rgb565_resize_frame(const uint8_t *input, uint16_t *output,
                         uint32_t input_width, uint32_t output_width,
                         uint32_t output_height,
                         const uint32_t *columns, const size_t *row_offsets);
