/* Host build:
 * gcc -std=c11 -Wall -Wextra -Werror -Iesp32-p4/main \
 *     esp32-p4/main/h264_mp4.c esp32-p4/tests/test_h264_mp4.c -o test_h264_mp4
 * No arguments: structural/bounds tests. Fixture mode:
 * test_h264_mp4 input.h264 output.mp4 [width height fps]
 * Defaults: 1920 1080 30. AUD boundaries delimit complete access units;
 * packaging begins at the first IDR containing SPS/PPS. DTS starts at zero.
 */
#include "h264_mp4.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t sps[] = {0x67, 0x42, 0xc0, 0x28, 0xda, 0x02, 0x80};
static const uint8_t pps[] = {0x68, 0xce, 0x3c, 0x80};
static const uint8_t sei[] = {0x06, 0x05, 0x00, 0x00, 0x03, 0x01, 0x80};
static const uint8_t idr[] = {0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x02, 0x80};
static const uint8_t idr2[] = {0x65, 0xab, 0x80};
static const uint8_t delta[] = {0x41, 0x9a, 0x80};
static const uint8_t aud[] = {0x09, 0xf0};

typedef struct {
    const uint8_t *data;
    size_t size;
} box_t;

static uint32_t be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

static uint16_t be16(const uint8_t *p)
{
    return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
}

static uint64_t be64(const uint8_t *p)
{
    return ((uint64_t)be32(p) << 32) | be32(p + 4);
}

static box_t find_box(const uint8_t *data, size_t len, const char type[4])
{
    box_t result = {NULL, 0};
    size_t pos = 0;
    while (pos < len) {
        assert(len - pos >= 8);
        size_t size = be32(data + pos);
        assert(size >= 8 && size <= len - pos);
        if (memcmp(data + pos + 4, type, 4) == 0) {
            assert(result.data == NULL); /* no duplicate boxes in this muxer */
            result = (box_t){data + pos, size};
        }
        pos += size;
    }
    assert(pos == len && result.data != NULL);
    return result;
}

static box_t child(box_t parent, size_t skip, const char type[4])
{
    assert(parent.size >= skip);
    return find_box(parent.data + skip, parent.size - skip, type);
}

static void append_nal(uint8_t *data, size_t capacity, size_t *len,
                       unsigned prefix, const uint8_t *nal, size_t nal_len)
{
    assert(prefix >= 3 && prefix <= 8);
    assert(*len <= capacity && prefix + nal_len <= capacity - *len);
    memset(data + *len, 0, prefix - 1);
    data[*len + prefix - 1] = 1;
    memcpy(data + *len + prefix, nal, nal_len);
    *len += prefix + nal_len;
}

static size_t mixed_frame(uint8_t *data, size_t capacity)
{
    size_t len = 0;
    append_nal(data, capacity, &len, 7, sps, sizeof(sps));
    append_nal(data, capacity, &len, 3, pps, sizeof(pps));
    append_nal(data, capacity, &len, 4, aud, sizeof(aud));
    append_nal(data, capacity, &len, 3, sei, sizeof(sei));
    append_nal(data, capacity, &len, 4, idr, sizeof(idr));
    append_nal(data, capacity, &len, 3, idr2, sizeof(idr2));
    assert(capacity - len >= 5);
    memset(data + len, 0, 5);
    return len + 5;
}

static void unchanged(const uint8_t *data, size_t len, uint8_t value)
{
    for (size_t i = 0; i < len; ++i) {
        assert(data[i] == value);
    }
}

static h264_mp4_config_t configured(void)
{
    uint8_t input[128];
    h264_mp4_config_t config;
    size_t len = mixed_frame(input, sizeof(input));
    assert(h264_mp4_configure(&config, input, len, 1280, 720));
    return config;
}

static void test_configuration(void)
{
    uint8_t input[1024];
    size_t len = mixed_frame(input, sizeof(input));
    h264_mp4_config_t config = configured();
    assert(config.width == 1280 && config.height == 720);
    assert(config.timescale == 90000);
    assert(strcmp(config.codec, "avc1.42c028") == 0);
    assert(config.sps_len == sizeof(sps) && config.pps_len == sizeof(pps));
    assert(memcmp(config.sps, sps, sizeof(sps)) == 0);
    assert(memcmp(config.pps, pps, sizeof(pps)) == 0);
    assert(h264_mp4_is_keyframe(input, len));

    /* Identical repeats are common on encoder keyframes. */
    append_nal(input, sizeof(input), &len, 4, sps, sizeof(sps));
    append_nal(input, sizeof(input), &len, 3, pps, sizeof(pps));
    assert(h264_mp4_configure(&config, input, len, 640, 480));
    assert(config.width == 640 && config.height == 480);

    uint8_t before[sizeof(config)];
    memcpy(before, &config, sizeof(config));
    const uint8_t other_sps[] = {0x67, 0x42, 0xc0, 0x29, 0x80};
    append_nal(input, sizeof(input), &len, 3, other_sps, sizeof(other_sps));
    assert(!h264_mp4_configure(&config, input, len, 640, 480));
    assert(memcmp(before, &config, sizeof(config)) == 0);

    len = 0;
    append_nal(input, sizeof(input), &len, 4, pps, sizeof(pps));
    append_nal(input, sizeof(input), &len, 3, sps, sizeof(sps));
    assert(h264_mp4_configure(&config, input, len, 1, UINT16_MAX));
    assert(!h264_mp4_configure(NULL, input, len, 1, 1));
    assert(!h264_mp4_configure(&config, NULL, len, 1, 1));
    assert(!h264_mp4_configure(&config, input, len, 0, 1));
    assert(!h264_mp4_configure(&config, input, len, 1, 0));

    len = 0;
    append_nal(input, sizeof(input), &len, 3, sps, sizeof(sps));
    assert(!h264_mp4_configure(&config, input, len, 1, 1));
    len = 0;
    append_nal(input, sizeof(input), &len, 3, pps, sizeof(pps));
    assert(!h264_mp4_configure(&config, input, len, 1, 1));

    uint8_t big_sps[257], big_pps[257];
    memset(big_sps, 0x7f, sizeof(big_sps));
    memset(big_pps, 0x7f, sizeof(big_pps));
    big_sps[0] = 0x67;
    big_sps[1] = 0x42;
    big_sps[2] = 0xc0;
    big_sps[3] = 0x28;
    big_pps[0] = 0x68;
    len = 0;
    append_nal(input, sizeof(input), &len, 4, big_sps, 256);
    append_nal(input, sizeof(input), &len, 3, big_pps, 256);
    assert(h264_mp4_configure(&config, input, len, 640, 480));
    assert(config.sps_len == 256 && config.pps_len == 256);
    uint8_t output[2048];
    assert(h264_mp4_init_segment(&config, output, sizeof(output)) != 0);
    for (unsigned which = 0; which < 2; ++which) {
        len = 0;
        append_nal(input, sizeof(input), &len, 4, big_sps, which ? 256 : 257);
        append_nal(input, sizeof(input), &len, 3, big_pps, which ? 257 : 256);
        memcpy(before, &config, sizeof(config));
        assert(!h264_mp4_configure(&config, input, len, 640, 480));
        assert(memcmp(before, &config, sizeof(config)) == 0);
    }
}

static void verify_init(const h264_mp4_config_t *c, const uint8_t *data, size_t len)
{
    box_t ftyp = find_box(data, len, "ftyp");
    box_t moov = find_box(data, len, "moov");
    assert(ftyp.data == data && ftyp.size == 32);
    assert(memcmp(ftyp.data + 8, "iso6", 4) == 0);
    assert(memcmp(ftyp.data + 16, "iso6isomavc1mp41", 16) == 0);
    assert(moov.data == data + ftyp.size && ftyp.size + moov.size == len);
    box_t mvhd = child(moov, 8, "mvhd");
    assert(mvhd.size == 108 && be32(mvhd.data + 8) == 0);
    assert(be32(mvhd.data + 20) == c->timescale);
    assert(be32(mvhd.data + 24) == 0);
    assert(be32(mvhd.data + 104) == 2);
    box_t trak = child(moov, 8, "trak");
    box_t tkhd = child(trak, 8, "tkhd");
    assert(tkhd.size == 92 && be32(tkhd.data + 8) == 7);
    assert(be32(tkhd.data + 20) == 1);
    assert(be32(tkhd.data + 84) == (uint32_t)c->width << 16);
    assert(be32(tkhd.data + 88) == (uint32_t)c->height << 16);
    box_t mdia = child(trak, 8, "mdia");
    box_t mdhd = child(mdia, 8, "mdhd");
    assert(mdhd.size == 32 && be32(mdhd.data + 20) == c->timescale);
    assert(be32(mdhd.data + 24) == 0 && be16(mdhd.data + 28) == 0x55c4);
    box_t hdlr = child(mdia, 8, "hdlr");
    assert(memcmp(hdlr.data + 16, "vide", 4) == 0);
    box_t minf = child(mdia, 8, "minf");
    box_t vmhd = child(minf, 8, "vmhd");
    assert(vmhd.size == 20 && be32(vmhd.data + 8) == 1);
    box_t dinf = child(minf, 8, "dinf");
    box_t dref = child(dinf, 8, "dref");
    assert(be32(dref.data + 12) == 1);
    box_t url = child(dref, 16, "url ");
    assert(url.size == 12 && be32(url.data + 8) == 1);
    box_t stbl = child(minf, 8, "stbl");
    box_t stsd = child(stbl, 8, "stsd");
    assert(be32(stsd.data + 8) == 0 && be32(stsd.data + 12) == 1);
    box_t avc1 = child(stsd, 16, "avc1");
    assert(be16(avc1.data + 14) == 1);
    assert(be16(avc1.data + 32) == c->width);
    assert(be16(avc1.data + 34) == c->height);
    assert(be32(avc1.data + 36) == 0x00480000);
    assert(be32(avc1.data + 40) == 0x00480000);
    assert(be16(avc1.data + 48) == 1);
    assert(be16(avc1.data + 82) == 0x18 && be16(avc1.data + 84) == 0xffff);
    box_t avcc = child(avc1, 86, "avcC");
    assert(avcc.size == 19 + c->sps_len + c->pps_len);
    assert(avcc.data[8] == 1 && memcmp(avcc.data + 9, c->sps + 1, 3) == 0);
    assert(avcc.data[12] == 0xff && avcc.data[13] == 0xe1);
    assert(be16(avcc.data + 14) == c->sps_len);
    assert(memcmp(avcc.data + 16, c->sps, c->sps_len) == 0);
    size_t pps_pos = 16 + c->sps_len;
    assert(avcc.data[pps_pos] == 1);
    assert(be16(avcc.data + pps_pos + 1) == c->pps_len);
    assert(memcmp(avcc.data + pps_pos + 3, c->pps, c->pps_len) == 0);
    static const char tables[][5] = {"stts", "stsc", "stsz", "stco"};
    for (size_t i = 0; i < 4; ++i) {
        box_t table = child(stbl, 8, tables[i]);
        assert(table.size == (i == 2 ? 20u : 16u));
        unchanged(table.data + 8, table.size - 8, 0);
    }
    box_t mvex = child(moov, 8, "mvex");
    box_t trex = child(mvex, 8, "trex");
    assert(trex.size == 32 && be32(trex.data + 8) == 0);
    assert(be32(trex.data + 12) == 1 && be32(trex.data + 16) == 1);
    assert(be32(trex.data + 20) == 0 && be32(trex.data + 24) == 0);
    assert(be32(trex.data + 28) == 0x01010000);
}

static void test_init_segment(void)
{
    h264_mp4_config_t config = configured();
    uint8_t output[2048], expected[2048];
    size_t need = h264_mp4_init_segment(&config, expected, sizeof(expected));
    assert(need > 0);
    verify_init(&config, expected, need);
    for (size_t cap = 0; cap < need; ++cap) {
        memset(output, 0xa5, sizeof(output));
        assert(h264_mp4_init_segment(&config, output + 1, cap) == 0);
        unchanged(output, sizeof(output), 0xa5);
    }
    memset(output, 0xa5, sizeof(output));
    assert(h264_mp4_init_segment(&config, output + 1, need) == need);
    assert(memcmp(output + 1, expected, need) == 0);
    assert(output[0] == 0xa5);
    unchanged(output + need + 1, sizeof(output) - need - 1, 0xa5);
    assert(h264_mp4_init_segment(NULL, output, sizeof(output)) == 0);
    assert(h264_mp4_init_segment(&config, NULL, sizeof(output)) == 0);
    assert(h264_mp4_init_segment(&config, (uint8_t *)&config, sizeof(output)) == 0);

    for (unsigned field = 0; field < 12; ++field) {
        h264_mp4_config_t bad = config;
        switch (field) {
        case 0: bad.sps_len = 3; break;
        case 1: bad.sps_len = 257; break;
        case 2: bad.sps_len = SIZE_MAX; break;
        case 3: bad.pps_len = 1; break;
        case 4: bad.pps_len = 257; break;
        case 5: bad.pps_len = SIZE_MAX; break;
        case 6: bad.sps[0] = 0xe7; break;
        case 7: bad.pps[0] = 0xe8; break;
        case 8: bad.sps[0] = 0x68; break;
        case 9: bad.width = 0; break;
        case 10: bad.height = 0; break;
        default: bad.timescale = 0; break;
        }
        memset(output, 0xa5, sizeof(output));
        assert(h264_mp4_init_segment(&bad, output, sizeof(output)) == 0);
        unchanged(output, sizeof(output), 0xa5);
    }
    config.width = UINT16_MAX;
    config.height = UINT16_MAX;
    config.timescale = UINT32_MAX;
    need = h264_mp4_init_segment(&config, output, sizeof(output));
    assert(need > 0);
    verify_init(&config, output, need);
}

static box_t verify_fragment(const uint8_t *data, size_t len, uint32_t sequence,
                             uint64_t dts, uint32_t duration, uint32_t flags)
{
    box_t moof = find_box(data, len, "moof");
    box_t mdat = find_box(data, len, "mdat");
    assert(moof.data == data && moof.size == 100);
    assert(mdat.data == data + moof.size && moof.size + mdat.size == len);
    box_t mfhd = child(moof, 8, "mfhd");
    assert(mfhd.size == 16 && be32(mfhd.data + 8) == 0);
    assert(be32(mfhd.data + 12) == sequence);
    box_t traf = child(moof, 8, "traf");
    box_t tfhd = child(traf, 8, "tfhd");
    assert(tfhd.size == 16 && be32(tfhd.data + 8) == 0x020000);
    assert(be32(tfhd.data + 12) == 1);
    box_t tfdt = child(traf, 8, "tfdt");
    assert(tfdt.size == 20 && be32(tfdt.data + 8) == 0x01000000);
    assert(be64(tfdt.data + 12) == dts);
    box_t trun = child(traf, 8, "trun");
    assert(trun.size == 32 && be32(trun.data + 8) == 0x00000701);
    assert(be32(trun.data + 12) == 1);
    assert(be32(trun.data + 16) == moof.size + 8);
    assert(data + be32(trun.data + 16) == mdat.data + 8);
    assert(be32(trun.data + 20) == duration);
    assert(be32(trun.data + 24) == mdat.size - 8);
    assert(be32(trun.data + 28) == flags);
    return mdat;
}

static void avcc_nal(box_t mdat, size_t *offset, const uint8_t *nal, size_t len)
{
    assert(*offset <= mdat.size && mdat.size - *offset >= 4 + len);
    assert(be32(mdat.data + *offset) == len);
    assert(memcmp(mdat.data + *offset + 4, nal, len) == 0);
    *offset += 4 + len;
}

static void test_fragments(void)
{
    uint8_t input[128], output[512], expected[512];
    size_t len = mixed_frame(input, sizeof(input));
    uint64_t dts = UINT64_C(0x123456789abcdef0);
    size_t need = h264_mp4_fragment(input, len, UINT32_MAX, dts, 3000,
                                   expected, sizeof(expected));
    assert(need == 108 + 12 + sizeof(sei) + sizeof(idr) + sizeof(idr2));
    box_t mdat = verify_fragment(expected, need, UINT32_MAX, dts, 3000, 0x02000000);
    size_t offset = 8;
    avcc_nal(mdat, &offset, sei, sizeof(sei));
    avcc_nal(mdat, &offset, idr, sizeof(idr));
    avcc_nal(mdat, &offset, idr2, sizeof(idr2));
    assert(offset == mdat.size);
    for (size_t cap = 0; cap < need; ++cap) {
        memset(output, 0xa5, sizeof(output));
        assert(h264_mp4_fragment(input, len, UINT32_MAX, dts, 3000,
                                 output + 1, cap) == 0);
        unchanged(output, sizeof(output), 0xa5);
    }
    memset(output, 0xa5, sizeof(output));
    assert(h264_mp4_fragment(input, len, UINT32_MAX, dts, 3000,
                             output + 1, need) == need);
    assert(memcmp(output + 1, expected, need) == 0 && output[0] == 0xa5);
    unchanged(output + need + 1, sizeof(output) - need - 1, 0xa5);
    assert(h264_mp4_fragment(input, len, 1, 0, 3000, input, sizeof(input)) == 0);
    assert(h264_mp4_fragment(input, len, 1, 0, 3000, input + 5, sizeof(input)) == 0);
    assert(h264_mp4_fragment(input, len, 1, 0, 3000, NULL, sizeof(output)) == 0);
    assert(h264_mp4_fragment(NULL, len, 1, 0, 3000, output, sizeof(output)) == 0);
    assert(h264_mp4_fragment(input, len, 1, 0, 0, output, sizeof(output)) == 0);
    assert(h264_mp4_fragment(input, len, 1, UINT64_MAX, 1,
                             output, sizeof(output)) == 0);
    need = h264_mp4_fragment(input, len, 0, UINT64_MAX - UINT32_MAX, UINT32_MAX,
                             output, sizeof(output));
    assert(need != 0);
    (void)verify_fragment(output, need, 0, UINT64_MAX - UINT32_MAX,
                          UINT32_MAX, 0x02000000);

    for (unsigned prefix = 3; prefix <= 4; ++prefix) {
        len = 0;
        append_nal(input, sizeof(input), &len, prefix, delta, sizeof(delta));
        assert(!h264_mp4_is_keyframe(input, len));
        need = h264_mp4_fragment(input, len, 7, 6000, 3000, output, sizeof(output));
        assert(need == 108 + 4 + sizeof(delta));
        mdat = verify_fragment(output, need, 7, 6000, 3000, 0x01010000);
        offset = 8;
        avcc_nal(mdat, &offset, delta, sizeof(delta));
        assert(offset == mdat.size);
    }
}

static void test_malformed(void)
{
    static const uint8_t bad[][16] = {
        {0x65, 0x80}, {0, 0, 0, 0}, {0, 0, 1}, {0, 0, 0, 1},
        {0, 0, 1, 0x65}, {0x80, 0, 0, 1, 0x65, 0x80},
        {0, 0, 1, 0, 0, 1, 0x65, 0x80},
        {0, 0, 1, 0x65, 0x80, 0, 0, 1},
        {0, 0, 1, 0xe5, 0x80}, {0, 0, 1, 0x60, 0x80},
        {0, 0, 1, 0x6d, 0x80}, {0, 0, 1, 0x67, 0x42},
        {0, 0, 1, 0x68},
        {0, 0, 1, 0x65, 0x80, 0, 0, 1, 0xe1, 0x80}
    };
    static const size_t lengths[] = {2, 4, 3, 4, 4, 6, 8, 8, 5, 5, 5, 5, 4, 10};
    h264_mp4_config_t config = configured();
    uint8_t before[sizeof(config)], output[512];
    memcpy(before, &config, sizeof(config));
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
        assert(!h264_mp4_is_keyframe(bad[i], lengths[i]));
        assert(!h264_mp4_configure(&config, bad[i], lengths[i], 1280, 720));
        assert(memcmp(before, &config, sizeof(config)) == 0);
        memset(output, 0xa5, sizeof(output));
        assert(h264_mp4_fragment(bad[i], lengths[i], 1, 0, 3000,
                                 output, sizeof(output)) == 0);
        unchanged(output, sizeof(output), 0xa5);
    }
    uint8_t input[128];
    size_t len = mixed_frame(input, sizeof(input));
    append_nal(input, sizeof(input), &len, 3, bad[8] + 3, 2);
    assert(!h264_mp4_configure(&config, input, len, 1280, 720));
    assert(memcmp(before, &config, sizeof(config)) == 0);
    assert(!h264_mp4_is_keyframe(input, len));
    len = 0;
    append_nal(input, sizeof(input), &len, 4, sps, sizeof(sps));
    append_nal(input, sizeof(input), &len, 3, pps, sizeof(pps));
    assert(h264_mp4_configure(&config, input, len, 1280, 720));
    assert(!h264_mp4_is_keyframe(input, len));
    assert(h264_mp4_fragment(input, len, 1, 0, 3000, output, sizeof(output)) == 0);
    len = 0;
    append_nal(input, sizeof(input), &len, 3, sei, sizeof(sei));
    append_nal(input, sizeof(input), &len, 4, aud, sizeof(aud));
    assert(h264_mp4_fragment(input, len, 1, 0, 3000, output, sizeof(output)) == 0);
    assert(!h264_mp4_is_keyframe(NULL, 0));
    assert(!h264_mp4_is_keyframe(input, 0));
    assert(!h264_mp4_configure(&config, input, 0, 1280, 720));
    assert(h264_mp4_fragment(input, 0, 1, 0, 3000, output, sizeof(output)) == 0);
    /* Oversized lengths must be rejected before even reading the buffer. */
    assert(!h264_mp4_is_keyframe(input, SIZE_MAX));
    assert(!h264_mp4_configure(&config, input, SIZE_MAX, 1280, 720));
    assert(h264_mp4_fragment(input, SIZE_MAX, 1, 0, 3000,
                             output, sizeof(output)) == 0);
}

static void test_limits(void)
{
    size_t input_capacity = H264_MP4_MAX_ANNEXB_SIZE;
    size_t output_capacity = input_capacity + 4 * H264_MP4_MAX_NALS + 108;
    uint8_t *input = malloc(input_capacity);
    uint8_t *output = malloc(output_capacity);
    assert(input && output);
    memset(input, 0x80, input_capacity);
    input[0] = input[1] = 0;
    input[2] = 1;
    input[3] = 0x65;
    size_t need = h264_mp4_fragment(input, input_capacity, 1, 0, 3000,
                                   output, output_capacity);
    assert(need == 108 + 4 + input_capacity - 3);
    box_t mdat = verify_fragment(output, need, 1, 0, 3000, 0x02000000);
    assert(be32(mdat.data + 8) == input_capacity - 3);
    assert(memcmp(mdat.data + 12, input + 3, input_capacity - 3) == 0);
    assert(h264_mp4_is_keyframe(input, input_capacity));
    assert(!h264_mp4_is_keyframe(input, input_capacity + 1));
    assert(h264_mp4_fragment(input, input_capacity + 1, 1, 0, 3000,
                             output, output_capacity) == 0);

    size_t len = 0;
    for (unsigned i = 0; i < H264_MP4_MAX_NALS; ++i) {
        append_nal(input, input_capacity, &len, i % 2 ? 3 : 4, delta, sizeof(delta));
    }
    need = h264_mp4_fragment(input, len, 1, 0, 3000, output, output_capacity);
    assert(need == 108 + H264_MP4_MAX_NALS * (4 + sizeof(delta)));
    (void)verify_fragment(output, need, 1, 0, 3000, 0x01010000);
    append_nal(input, input_capacity, &len, 3, idr, sizeof(idr));
    assert(!h264_mp4_is_keyframe(input, len));
    memset(output, 0xa5, output_capacity);
    assert(h264_mp4_fragment(input, len, 1, 0, 3000, output, output_capacity) == 0);
    unchanged(output, output_capacity, 0xa5);
    free(output);
    free(input);
}

static void test_random_input(void)
{
    uint32_t state = 1;
    uint8_t input[128], output[1024];
    h264_mp4_config_t config = configured();
    for (unsigned iteration = 0; iteration < 10000; ++iteration) {
        size_t len = iteration % sizeof(input);
        for (size_t i = 0; i < len; ++i) {
            state = state * 1664525u + 1013904223u;
            input[i] = (uint8_t)(state >> 24);
        }
        if (iteration % 2 == 0) {
            len = mixed_frame(input, sizeof(input));
            input[(state >> 16) % len] ^= (uint8_t)(1u << (state % 8));
        }
        uint8_t before[sizeof(config)];
        memcpy(before, &config, sizeof(config));
        if (!h264_mp4_configure(&config, input, len, 1280, 720)) {
            assert(memcmp(before, &config, sizeof(config)) == 0);
        }
        memset(output, 0xa5, sizeof(output));
        size_t size = h264_mp4_fragment(input, len, iteration, iteration * 3000u,
                                       3000, output, sizeof(output));
        if (size == 0) {
            unchanged(output, sizeof(output), 0xa5);
        } else {
            assert(size <= sizeof(output));
            uint32_t flags = h264_mp4_is_keyframe(input, len) ? 0x02000000 : 0x01010000;
            box_t mdat = verify_fragment(output, size, iteration, iteration * 3000u,
                                         3000, flags);
            size_t pos = 8;
            while (pos < mdat.size) {
                assert(mdat.size - pos >= 4);
                size_t nal_len = be32(mdat.data + pos);
                pos += 4;
                assert(nal_len >= 2 && nal_len <= mdat.size - pos);
                unsigned type = mdat.data[pos] & 31u;
                assert(type != 7 && type != 8 && type != 9);
                pos += nal_len;
            }
            assert(pos == mdat.size);
            unchanged(output + size, sizeof(output) - size, 0xa5);
        }
    }
}

/* These helpers are host-only; the embedded helper has no file I/O or heap. */
static bool parse_number(const char *text, uint32_t max, uint32_t *value)
{
    uint32_t next = 0;
    if (!text || !*text) {
        return false;
    }
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        uint32_t digit = (uint32_t)(*p - '0');
        if (digit > max || next > (max - digit) / 10) {
            return false;
        }
        next = next * 10 + digit;
    }
    if (!next) {
        return false;
    }
    *value = next;
    return true;
}

static bool fixture_start(const uint8_t *data, size_t len, size_t from,
                          size_t *prefix, size_t *payload)
{
    size_t zeros = 0;
    for (size_t i = from; i < len; ++i) {
        if (data[i] == 0) {
            ++zeros;
        } else {
            if (data[i] == 1 && zeros >= 2) {
                *prefix = i - zeros;
                *payload = i + 1;
                return true;
            }
            zeros = 0;
        }
    }
    return false;
}

static bool write_access_unit(FILE *file, const uint8_t *data, size_t len,
                              uint8_t *output, size_t capacity,
                              h264_mp4_config_t *config, uint32_t *frames,
                              uint16_t width, uint16_t height, uint32_t fps)
{
    uint64_t dts = (uint64_t)*frames * H264_MP4_TIMESCALE / fps;
    uint64_t next_dts = ((uint64_t)*frames + 1) * H264_MP4_TIMESCALE / fps;
    uint32_t duration = (uint32_t)(next_dts - dts);
    bool keyframe = h264_mp4_is_keyframe(data, len);
    if (*frames == 0) {
        if (!keyframe) {
            /* Validate skipped leading P access units before waiting for IDR. */
            return h264_mp4_fragment(data, len, 1, 0, duration, output, capacity) != 0;
        }
        if (!h264_mp4_configure(config, data, len, width, height)) {
            fprintf(stderr, "First IDR access unit must contain SPS/PPS.\n");
            return false;
        }
        size_t init_len = h264_mp4_init_segment(config, output, capacity);
        if (!init_len || fwrite(output, 1, init_len, file) != init_len) {
            return false;
        }
    }
    size_t size = h264_mp4_fragment(data, len, *frames + 1, dts, duration,
                                   output, capacity);
    if (!size || fwrite(output, 1, size, file) != size) {
        fprintf(stderr, "Cannot package access unit %u.\n", (unsigned)*frames);
        return false;
    }
    ++*frames;
    return true;
}

static void test_fixture_helpers(void)
{
    uint32_t number = 123;
    assert(parse_number("65535", UINT16_MAX, &number) && number == 65535);
    assert(parse_number("90000", H264_MP4_TIMESCALE, &number) && number == 90000);
    static const char *bad[] = {"", "0", "-1", "+1", " 30", "30.0", "65536",
                               "999999999999999999999999999999"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        number = 123;
        assert(!parse_number(bad[i], UINT16_MAX, &number));
        assert(number == 123);
    }
    uint8_t input[256];
    size_t len = mixed_frame(input, sizeof(input));
    size_t boundary = len - 5; /* trailing zeros belong to the next prefix */
    append_nal(input, sizeof(input), &len, 3, aud, sizeof(aud));
    append_nal(input, sizeof(input), &len, 3, delta, sizeof(delta));
    size_t cursor = 0, prefix, payload;
    unsigned nals = 0, auds = 0;
    while (fixture_start(input, len, cursor, &prefix, &payload)) {
        assert(prefix >= cursor && payload < len);
        if ((input[payload] & 31u) == 9) {
            if (++auds == 2) {
                assert(prefix == boundary);
            }
        }
        ++nals;
        cursor = payload;
    }
    assert(nals == 8 && auds == 2);
}

static int convert_fixture(const char *input_path, const char *output_path,
                           uint16_t width, uint16_t height, uint32_t fps)
{
    const long max_fixture = 64L * 1024L * 1024L;
    size_t capacity = H264_MP4_MAX_ANNEXB_SIZE + 4 * H264_MP4_MAX_NALS + 108;
    if (strcmp(input_path, output_path) == 0) {
        fprintf(stderr, "Input and output paths must differ.\n");
        return 1;
    }
    FILE *input_file = fopen(input_path, "rb");
    if (!input_file) {
        perror(input_path);
        return 1;
    }
    if (fseek(input_file, 0, SEEK_END) != 0) {
        fclose(input_file);
        return 1;
    }
    long length = ftell(input_file);
    if (length <= 0 || length > max_fixture || fseek(input_file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Fixture must be 1..64 MiB.\n");
        fclose(input_file);
        return 1;
    }
    size_t len = (size_t)length;
    uint8_t *data = malloc(len), *output = malloc(capacity);
    if (!data || !output || fread(data, 1, len, input_file) != len) {
        fclose(input_file);
        free(data);
        free(output);
        return 1;
    }
    fclose(input_file);
    FILE *file = fopen(output_path, "wb");
    if (!file) {
        perror(output_path);
        free(data);
        free(output);
        return 1;
    }
    size_t cursor = 0, unit_start = 0, prefix, payload;
    uint32_t frames = 0;
    bool seen_aud = false, ok = true;
    h264_mp4_config_t config = {0};
    while (fixture_start(data, len, cursor, &prefix, &payload)) {
        if (payload >= len) {
            ok = false;
            break;
        }
        if ((data[payload] & 31u) == 9) {
            if (seen_aud) {
                if (!write_access_unit(file, data + unit_start, prefix - unit_start,
                                       output, capacity, &config, &frames,
                                       width, height, fps)) {
                    ok = false;
                    break;
                }
                unit_start = prefix;
            }
            seen_aud = true;
        }
        cursor = payload;
    }
    if (ok && seen_aud) {
        ok = write_access_unit(file, data + unit_start, len - unit_start,
                               output, capacity, &config, &frames, width, height, fps);
    } else if (!seen_aud) {
        fprintf(stderr, "Fixture requires AUD NAL boundaries.\n");
        ok = false;
    }
    if (ok && !frames) {
        fprintf(stderr, "Fixture requires an IDR access unit with SPS/PPS.\n");
        ok = false;
    }
    if (fclose(file) != 0) {
        ok = false;
    }
    if (ok) {
        printf("Packaged %u frames, %ux%u, %s, %u fps.\n", (unsigned)frames,
               (unsigned)config.width, (unsigned)config.height, config.codec,
               (unsigned)fps);
    }
    free(output);
    free(data);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 3 || argc == 6) {
        uint32_t width = 1920, height = 1080, fps = 30;
        if (argc == 6 && (!parse_number(argv[3], UINT16_MAX, &width) ||
                          !parse_number(argv[4], UINT16_MAX, &height) ||
                          !parse_number(argv[5], H264_MP4_TIMESCALE, &fps))) {
            fprintf(stderr, "Require width/height 1..65535, integer fps 1..90000.\n");
            return 1;
        }
        return convert_fixture(argv[1], argv[2], (uint16_t)width, (uint16_t)height, fps);
    }
    if (argc != 1) {
        fprintf(stderr, "Usage: %s [input.h264 output.mp4 [width height fps]]\n", argv[0]);
        return 1;
    }
    test_configuration();
    test_init_segment();
    test_fragments();
    test_malformed();
    test_limits();
    test_random_input();
    test_fixture_helpers();
    puts("H.264 fragmented MP4 checks passed (boxes, AVCC, bounds, 10000 mutations).");
    return 0;
}
