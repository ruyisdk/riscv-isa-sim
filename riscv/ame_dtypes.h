// See LICENSE for license details.
// Ztt / AME datatype helpers (provisional v0.6).

#ifndef _RISCV_AME_DTYPES_H
#define _RISCV_AME_DTYPES_H

#include <cstdint>
#include <algorithm>

struct ame_dtype_t {
  uint32_t raw = 0;
  uint32_t size_bits = 0;
  bool is_fp = false;
  bool is_custom = false;
  bool is_signed = false;
  bool saturate = false;
  bool has_inf = true;   // FP only: bit 21, 0 = saturate to min/max
  bool denorm = true;    // FP only: bit 20, 0 = flush-to-zero
  uint8_t int_rm = 0;    // bits 28:27
  uint8_t fp_rm = 0;     // bits 25:22
  uint8_t exp_width = 0;
};

// Spec 4.4: widest integer used to interpret mldexp*.ew.x scalar exponents.
static const uint32_t AME_MAX_INT_DTYPE = 0x40000080u; // Int128

inline ame_dtype_t ame_parse_dtype(uint32_t enc) {
  ame_dtype_t d;
  d.raw = enc;
  d.size_bits = enc & 0xff;
  d.is_custom = (enc >> 31) & 1;
  d.is_fp = (enc >> 8) & 1;
  if (!d.is_fp) {
    d.is_signed = (enc >> 30) & 1;
    d.saturate = (enc >> 29) & 1;
    d.int_rm = (enc >> 27) & 0x3;
  } else {
    d.exp_width = (enc >> 26) & 0x1f;
    d.fp_rm = (enc >> 22) & 0xf;
    d.has_inf = (enc >> 21) & 1;
    d.denorm = (enc >> 20) & 1;
    d.is_signed = true;
  }
  return d;
}

// Canonical key: ignore RM, SAT, and unused middle bits so Sail encodings
// with don't-care fields still match the standard families.
inline uint32_t ame_dtype_family_key(uint32_t enc) {
  if (enc == 0)
    return 0;
  const ame_dtype_t d = ame_parse_dtype(enc);
  if (d.is_custom)
    return enc;
  if (!d.is_fp)
    return (d.is_signed ? 0x40000000u : 0) | d.size_bits;
  return (1u << 8) | d.size_bits | (uint32_t(d.exp_width) << 26);
}

// Identical-descriptor compare for ops that require Md[md]==Md[ms*]: ignore
// integer RM don't-care (Table 3 "xx") and reserved bits, but keep SAT.
inline uint32_t ame_dtype_ident_key(uint32_t enc) {
  if (enc == 0)
    return 0;
  const ame_dtype_t d = ame_parse_dtype(enc);
  if (d.is_custom)
    return enc;
  if (!d.is_fp)
    return (d.is_signed ? 0x40000000u : 0) | (d.saturate ? 0x20000000u : 0) | d.size_bits;
  return ame_dtype_family_key(enc);
}

inline bool ame_is_int_family(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  if (d.is_fp || d.is_custom)
    return false;
  return d.size_bits == 4 || d.size_bits == 8 || d.size_bits == 16 ||
         d.size_bits == 32 || d.size_bits == 64 || d.size_bits == 128;
}

inline bool ame_is_fp16(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  return d.is_fp && !d.is_custom && d.size_bits == 16 && d.exp_width == 5;
}
inline bool ame_is_bf16(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  return d.is_fp && !d.is_custom && d.size_bits == 16 && d.exp_width == 8;
}
inline bool ame_is_fp32(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  return d.is_fp && !d.is_custom && d.size_bits == 32 && d.exp_width == 8;
}
inline bool ame_is_fp64(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  return d.is_fp && !d.is_custom && d.size_bits == 64 && d.exp_width == 11;
}
inline bool ame_is_e4m3(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  return d.is_fp && !d.is_custom && d.size_bits == 8 && d.exp_width == 4;
}
inline bool ame_is_e5m2(uint32_t enc) {
  const ame_dtype_t d = ame_parse_dtype(enc);
  return d.is_fp && !d.is_custom && d.size_bits == 8 && d.exp_width == 5;
}
inline bool ame_is_fp_family(uint32_t enc) {
  return ame_is_fp16(enc) || ame_is_bf16(enc) || ame_is_fp32(enc) ||
         ame_is_fp64(enc) || ame_is_e4m3(enc) || ame_is_e5m2(enc);
}

inline bool ame_dtype_storage_ratio_ok(uint32_t size_bits, uint32_t udsz) {
  if (size_bits == 0)
    return true; // uninitialized
  if (size_bits < udsz)
    return (udsz % size_bits) == 0;
  return (size_bits % udsz) == 0;
}

inline uint32_t ame_pack_factor(uint32_t size_bits, uint32_t udsz) {
  if (size_bits == 0 || size_bits >= udsz)
    return 1;
  return udsz / size_bits;
}

// Packed element storage: E-bit fields back-to-back, first field in the low bits.
inline uint32_t ame_packed_bytes(uint32_t n_elems, uint32_t size_bits) {
  return uint32_t((uint64_t(n_elems) * size_bits + 7) / 8);
}

// Spec 2.6: base alignment is ceil(E/8).
inline uint32_t ame_elem_align_bytes(uint32_t size_bits) {
  if (size_bits == 0)
    return 1;
  return (size_bits + 7) / 8;
}

inline uint32_t ame_group_size(uint32_t size_bits, uint32_t udsz) {
  if (size_bits == 0)
    return 1;
  return std::max(uint32_t(1), (size_bits + udsz - 1) / udsz);
}

// Phase-1 supported standard datatypes (rnd don't-care).
inline bool ame_dtype_supported_m(uint32_t enc, uint32_t udsz) {
  if (enc == 0)
    return true;
  if ((enc >> 31) & 1)
    return false;
  const ame_dtype_t d = ame_parse_dtype(enc);
  if (!ame_dtype_storage_ratio_ok(d.size_bits, udsz))
    return false;

  return ame_is_int_family(enc) || ame_is_fp_family(enc);
}

inline bool ame_dtype_supported_acc(uint32_t enc, uint32_t udsz) {
  if (enc == 0)
    return true;
  if ((enc >> 31) & 1)
    return false;
  const ame_dtype_t d = ame_parse_dtype(enc);
  // Acc holds one square; storage ratio still applies for packability rules.
  if (d.size_bits != 0 && d.size_bits < udsz && (udsz % d.size_bits) != 0)
    return false;
  if (d.size_bits > udsz && (d.size_bits % udsz) != 0)
    return false;

  return ame_is_int_family(enc) || ame_is_fp_family(enc);
}

#endif
