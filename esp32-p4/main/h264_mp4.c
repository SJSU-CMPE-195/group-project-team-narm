#include "h264_mp4.h"

#include <string.h>

typedef struct {
  const uint8_t *data;
  size_t len;
  size_t cursor;
  size_t count;
  bool finished;
} annexb_reader_t;

typedef struct {
  const uint8_t *data;
  size_t len;
  unsigned type;
} nal_t;

typedef struct {
  uint8_t *data;
  size_t capacity;
  size_t pos;
  bool ok;
} writer_t;

/* Consume the entire zero run before 01: any extra zeros belong to Annex-B
 * framing, never to the preceding EBSP. 00 00 03 xx remains untouched. */
static bool start_code(const uint8_t *data, size_t len, size_t from,
                       size_t *prefix, size_t *payload) {
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

static bool annexb_open(annexb_reader_t *r, const uint8_t *data, size_t len) {
  size_t prefix, payload;
  if (!data || len == 0 || len > H264_MP4_MAX_ANNEXB_SIZE ||
      !start_code(data, len, 0, &prefix, &payload)) {
    return false;
  }
  for (size_t i = 0; i < prefix; ++i) {
    if (data[i] != 0) {
      return false;
    }
  }
  *r = (annexb_reader_t){data, len, payload, 0, false};
  return true;
}

/* 1 = NAL, 0 = clean end, -1 = malformed/unsupported input. Baseline AVC
 * NAL types 1..12 are supported; extension/reserved types are rejected. */
static int annexb_next(annexb_reader_t *r, nal_t *nal) {
  size_t prefix, payload, end;
  if (r->finished) {
    return 0;
  }
  if (++r->count > H264_MP4_MAX_NALS) {
    return -1;
  }
  if (start_code(r->data, r->len, r->cursor, &prefix, &payload)) {
    end = prefix;
  } else {
    end = r->len;
    payload = r->len;
    r->finished = true;
  }
  while (end > r->cursor && r->data[end - 1] == 0) {
    --end;
  }
  if (end <= r->cursor) {
    return -1;
  }
  nal->data = r->data + r->cursor;
  nal->len = end - r->cursor;
  nal->type = nal->data[0] & 31u;
  r->cursor = payload;
  if ((nal->data[0] & 0x80u) || nal->type == 0 || nal->type > 12 ||
      nal->len < 2 ||
      (nal->type == 7 &&
       (nal->len < 4 || nal->len > H264_MP4_MAX_PARAMETER_SET_SIZE)) ||
      (nal->type == 8 && nal->len > H264_MP4_MAX_PARAMETER_SET_SIZE)) {
    return -1;
  }
  return 1;
}

static bool overlaps(const void *a, size_t a_len, const void *b, size_t b_len) {
  uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
  return aa <= bb ? bb - aa < a_len : aa - bb < b_len;
}

bool h264_mp4_configure(h264_mp4_config_t *config, const uint8_t *annexb,
                        size_t len, uint16_t width, uint16_t height) {
  annexb_reader_t reader;
  nal_t nal;
  int status;
  h264_mp4_config_t next = {0};
  static const char hex[] = "0123456789abcdef";
  if (!config || !width || !height || !annexb_open(&reader, annexb, len)) {
    return false;
  }
  while ((status = annexb_next(&reader, &nal)) > 0) {
    if (nal.type == 7 || nal.type == 8) {
      uint8_t *dest = nal.type == 7 ? next.sps : next.pps;
      size_t *dest_len = nal.type == 7 ? &next.sps_len : &next.pps_len;
      if (*dest_len &&
          (*dest_len != nal.len || memcmp(dest, nal.data, nal.len) != 0)) {
        return false;
      }
      memcpy(dest, nal.data, nal.len);
      *dest_len = nal.len;
    }
  }
  if (status < 0 || !next.sps_len || !next.pps_len) {
    return false;
  }
  next.width = width;
  next.height = height;
  next.timescale = H264_MP4_TIMESCALE;
  memcpy(next.codec, "avc1.", 5);
  for (size_t i = 0; i < 3; ++i) {
    next.codec[5 + i * 2] = hex[next.sps[i + 1] >> 4];
    next.codec[6 + i * 2] = hex[next.sps[i + 1] & 15u];
  }
  *config = next;
  return true;
}

bool h264_mp4_is_keyframe(const uint8_t *data, size_t len) {
  annexb_reader_t reader;
  nal_t nal;
  int status;
  bool idr = false;
  if (!annexb_open(&reader, data, len)) {
    return false;
  }
  while ((status = annexb_next(&reader, &nal)) > 0) {
    idr = idr || nal.type == 5;
  }
  return status == 0 && idr;
}

static void bytes(writer_t *w, const void *data, size_t len) {
  if (!w->ok || len > w->capacity - w->pos) {
    w->ok = false;
    return;
  }
  if (w->data) {
    if (data) {
      memcpy(w->data + w->pos, data, len);
    } else {
      memset(w->data + w->pos, 0, len);
    }
  }
  w->pos += len;
}

static void u8(writer_t *w, uint8_t value) { bytes(w, &value, 1); }

static void u16(writer_t *w, uint16_t value) {
  uint8_t data[] = {(uint8_t)(value >> 8), (uint8_t)value};
  bytes(w, data, sizeof(data));
}

static void u32(writer_t *w, uint32_t value) {
  uint8_t data[] = {(uint8_t)(value >> 24), (uint8_t)(value >> 16),
                    (uint8_t)(value >> 8), (uint8_t)value};
  bytes(w, data, sizeof(data));
}

static void u64(writer_t *w, uint64_t value) {
  u32(w, (uint32_t)(value >> 32));
  u32(w, (uint32_t)value);
}

static void patch32(writer_t *w, size_t offset, uint32_t value) {
  if (!w->ok || offset > w->pos || w->pos - offset < 4) {
    w->ok = false;
    return;
  }
  if (w->data) {
    writer_t patch = {w->data + offset, 4, 0, true};
    u32(&patch, value);
  }
}

static size_t box(writer_t *w, const char type[4]) {
  size_t pos = w->pos;
  u32(w, 0);
  bytes(w, type, 4);
  return pos;
}

static void end_box(writer_t *w, size_t pos) {
  if (pos > w->pos || w->pos - pos > UINT32_MAX) {
    w->ok = false;
  } else {
    patch32(w, pos, (uint32_t)(w->pos - pos));
  }
}

static void matrix(writer_t *w) {
  static const uint32_t values[] = {0x00010000, 0, 0, 0,         0x00010000,
                                    0,          0, 0, 0x40000000};
  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
    u32(w, values[i]);
  }
}

static void empty_table(writer_t *w, const char type[4], bool sample_size) {
  size_t pos = box(w, type);
  u32(w, 0); /* version/flags */
  if (sample_size) {
    u32(w, 0);
  }
  u32(w, 0); /* entry/sample count */
  end_box(w, pos);
}

static void init_boxes(writer_t *w, const h264_mp4_config_t *c) {
  size_t ftyp = box(w, "ftyp");
  bytes(w, "iso6", 4);
  u32(w, 1);
  bytes(w, "iso6isomavc1mp41", 16);
  end_box(w, ftyp);

  size_t moov = box(w, "moov");
  size_t mvhd = box(w, "mvhd");
  u32(w, 0);
  bytes(w, NULL, 8); /* creation/modification */
  u32(w, c->timescale);
  u32(w, 0); /* unknown duration for a live stream */
  u32(w, 0x00010000);
  u16(w, 0x0100);
  bytes(w, NULL, 10);
  matrix(w);
  bytes(w, NULL, 24);
  u32(w, 2); /* next track ID */
  end_box(w, mvhd);

  size_t trak = box(w, "trak");
  size_t tkhd = box(w, "tkhd");
  u32(w, 7); /* enabled, in movie, in preview */
  bytes(w, NULL, 8);
  u32(w, 1);
  bytes(w, NULL, 16); /* reserved, duration, reserved[2] */
  bytes(w, NULL, 8);  /* layer, alternate group, volume, reserved */
  matrix(w);
  u32(w, (uint32_t)c->width << 16);
  u32(w, (uint32_t)c->height << 16);
  end_box(w, tkhd);

  size_t mdia = box(w, "mdia");
  size_t mdhd = box(w, "mdhd");
  u32(w, 0);
  bytes(w, NULL, 8);
  u32(w, c->timescale);
  u32(w, 0);
  u16(w, 0x55c4); /* und */
  u16(w, 0);
  end_box(w, mdhd);

  size_t hdlr = box(w, "hdlr");
  bytes(w, NULL, 8);
  bytes(w, "vide", 4);
  bytes(w, NULL, 12);
  bytes(w, "VideoHandler", sizeof("VideoHandler"));
  end_box(w, hdlr);

  size_t minf = box(w, "minf");
  size_t vmhd = box(w, "vmhd");
  u32(w, 1);
  bytes(w, NULL, 8);
  end_box(w, vmhd);

  size_t dinf = box(w, "dinf");
  size_t dref = box(w, "dref");
  u32(w, 0);
  u32(w, 1);
  size_t url = box(w, "url ");
  u32(w, 1); /* self-contained */
  end_box(w, url);
  end_box(w, dref);
  end_box(w, dinf);

  size_t stbl = box(w, "stbl");
  size_t stsd = box(w, "stsd");
  u32(w, 0);
  u32(w, 1);
  size_t avc1 = box(w, "avc1");
  bytes(w, NULL, 6);
  u16(w, 1); /* data reference index */
  bytes(w, NULL, 16);
  u16(w, c->width);
  u16(w, c->height);
  u32(w, 0x00480000);
  u32(w, 0x00480000);
  u32(w, 0);
  u16(w, 1);          /* frame count */
  bytes(w, NULL, 32); /* empty compressor name */
  u16(w, 0x0018);
  u16(w, 0xffff);

  size_t avcc = box(w, "avcC");
  u8(w, 1);
  bytes(w, c->sps + 1, 3);
  u8(w, 0xff); /* reserved bits + four-byte NAL lengths */
  u8(w, 0xe1); /* reserved bits + one SPS */
  u16(w, (uint16_t)c->sps_len);
  bytes(w, c->sps, c->sps_len);
  u8(w, 1);
  u16(w, (uint16_t)c->pps_len);
  bytes(w, c->pps, c->pps_len);
  end_box(w, avcc);
  end_box(w, avc1);
  end_box(w, stsd);
  empty_table(w, "stts", false);
  empty_table(w, "stsc", false);
  empty_table(w, "stsz", true);
  empty_table(w, "stco", false);
  end_box(w, stbl);
  end_box(w, minf);
  end_box(w, mdia);
  end_box(w, trak);

  size_t mvex = box(w, "mvex");
  size_t trex = box(w, "trex");
  u32(w, 0);
  u32(w, 1); /* track */
  u32(w, 1); /* sample description index */
  u32(w, 0); /* default duration supplied by trun */
  u32(w, 0); /* default size supplied by trun */
  u32(w, 0x01010000);
  end_box(w, trex);
  end_box(w, mvex);
  end_box(w, moov);
}

size_t h264_mp4_init_segment(const h264_mp4_config_t *config, uint8_t *out,
                             size_t capacity) {
  writer_t count = {NULL, SIZE_MAX, 0, true};
  if (!config || !out || config->sps_len < 4 ||
      config->sps_len > H264_MP4_MAX_PARAMETER_SET_SIZE ||
      config->pps_len < 2 ||
      config->pps_len > H264_MP4_MAX_PARAMETER_SET_SIZE ||
      (config->sps[0] & 0x9fu) != 7 || (config->pps[0] & 0x9fu) != 8 ||
      !config->width || !config->height || !config->timescale) {
    return 0;
  }
  init_boxes(&count, config);
  if (!count.ok || count.pos > capacity ||
      overlaps(config, sizeof(*config), out, count.pos)) {
    return 0;
  }
  writer_t w = {out, capacity, 0, true};
  init_boxes(&w, config);
  return w.ok ? w.pos : 0;
}

static bool media_nal(unsigned type) {
  return type != 7 && type != 8 && type != 9;
}

size_t h264_mp4_fragment(const uint8_t *annexb, size_t len, uint32_t sequence,
                         uint64_t dts, uint32_t duration, uint8_t *out,
                         size_t capacity) {
  annexb_reader_t reader;
  nal_t nal;
  int status;
  size_t sample_size = 0;
  bool idr = false, vcl = false;
  if (!out || !duration || dts > UINT64_MAX - duration ||
      !annexb_open(&reader, annexb, len)) {
    return 0;
  }
  while ((status = annexb_next(&reader, &nal)) > 0) {
    idr = idr || nal.type == 5;
    vcl = vcl || nal.type <= 5;
    if (media_nal(nal.type)) {
      if (nal.len > UINT32_MAX - 4u ||
          sample_size > UINT32_MAX - 4u - nal.len) {
        return 0;
      }
      sample_size += 4 + nal.len;
    }
  }
  if (status < 0 || !vcl ||
      sample_size > SIZE_MAX - H264_MP4_FRAGMENT_OVERHEAD ||
      capacity < H264_MP4_FRAGMENT_OVERHEAD + sample_size ||
      overlaps(annexb, len, out, H264_MP4_FRAGMENT_OVERHEAD + sample_size)) {
    return 0;
  }

  writer_t w = {out, capacity, 0, true};
  size_t moof = box(&w, "moof");
  size_t mfhd = box(&w, "mfhd");
  u32(&w, 0);
  u32(&w, sequence);
  end_box(&w, mfhd);
  size_t traf = box(&w, "traf");
  size_t tfhd = box(&w, "tfhd");
  u32(&w, 0x00020000); /* default-base-is-moof */
  u32(&w, 1);
  end_box(&w, tfhd);
  size_t tfdt = box(&w, "tfdt");
  u32(&w, 0x01000000); /* version 1, 64-bit decode time */
  u64(&w, dts);
  end_box(&w, tfdt);
  size_t trun = box(&w, "trun");
  u32(&w, 0x00000701); /* data offset, duration, size, flags */
  u32(&w, 1);
  size_t data_offset = w.pos;
  u32(&w, 0);
  u32(&w, duration);
  u32(&w, (uint32_t)sample_size);
  u32(&w, idr ? 0x02000000 : 0x01010000);
  end_box(&w, trun);
  end_box(&w, traf);
  end_box(&w, moof);
  patch32(&w, data_offset, (uint32_t)(w.pos - moof + 8));

  size_t mdat = box(&w, "mdat");
  (void)annexb_open(&reader, annexb, len); /* validated in the first pass */
  while (annexb_next(&reader, &nal) > 0) {
    if (media_nal(nal.type)) {
      u32(&w, (uint32_t)nal.len);
      bytes(&w, nal.data, nal.len);
    }
  }
  end_box(&w, mdat);
  return w.ok ? w.pos : 0;
}
