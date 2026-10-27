// See LICENSE for license details.

#include "ame_ops.h"
#include "softfloat.h"
#include "specialize.h"
#include "internals.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace {

// Defined with the FP helpers further below; used by matmul/ew/ldexp before
// that point.
uint8_t ame_fp_rm_to_softfloat(uint8_t fp_rm);
bool ame_rm_is_rno(uint8_t fp_rm);
float32_t f32_host_overflow_fixup(float32_t v, bool input_finite);
float32_t f32_neg(float32_t a);
float64_t f64_neg(float64_t a);

unsigned __int128 load_u128_bits(const uint8_t* p, uint32_t bit_off, uint32_t size_bits) {
  unsigned __int128 u = 0;
  if (size_bits == 0)
    return 0;
  if ((bit_off % 8) == 0 && (size_bits % 8) == 0) {
    const uint32_t nbytes = size_bits / 8;
    const uint8_t* q = p + bit_off / 8;
    for (uint32_t i = 0; i < nbytes; i++)
      u |= (unsigned __int128)(q[i]) << (8 * i);
  } else {
    for (uint32_t b = 0; b < size_bits; b++) {
      const uint32_t pos = bit_off + b;
      if (p[pos / 8] & uint8_t(1u << (pos % 8)))
        u |= (unsigned __int128)(1) << b;
    }
  }
  if (size_bits < 128)
    u &= ((unsigned __int128)(1) << size_bits) - 1;
  return u;
}

void store_u128_bits(uint8_t* p, uint32_t bit_off, uint32_t size_bits, unsigned __int128 u) {
  if (size_bits == 0)
    return;
  if (size_bits < 128)
    u &= ((unsigned __int128)(1) << size_bits) - 1;
  if ((bit_off % 8) == 0 && (size_bits % 8) == 0) {
    const uint32_t nbytes = size_bits / 8;
    uint8_t* q = p + bit_off / 8;
    for (uint32_t i = 0; i < nbytes; i++)
      q[i] = uint8_t(u >> (8 * i));
    return;
  }
  for (uint32_t b = 0; b < size_bits; b++) {
    const uint32_t pos = bit_off + b;
    const uint8_t bit = uint8_t(1u << (pos % 8));
    if ((u >> b) & 1)
      p[pos / 8] |= bit;
    else
      p[pos / 8] &= uint8_t(~bit);
  }
}

int64_t load_int_bits(const uint8_t* p, uint32_t bit_off, uint32_t size_bits,
                      bool is_signed) {
  const unsigned __int128 u = load_u128_bits(p, bit_off, size_bits <= 64 ? size_bits : 64);
  if (!is_signed || size_bits >= 64)
    return int64_t(uint64_t(u));
  const uint32_t shift = 64 - size_bits;
  return int64_t(uint64_t(u) << shift) >> shift;
}

void store_int_bits(uint8_t* p, uint32_t bit_off, uint32_t size_bits, int64_t v) {
  store_u128_bits(p, bit_off, size_bits <= 64 ? size_bits : 64, (unsigned __int128)(uint64_t(v)));
}

unsigned __int128 load_uwide_at(const uint8_t* base, uint32_t index, uint32_t size_bits) {
  return load_u128_bits(base, index * size_bits, size_bits);
}

void store_uwide_at(uint8_t* base, uint32_t index, uint32_t size_bits, unsigned __int128 v) {
  store_u128_bits(base, index * size_bits, size_bits, v);
}

__int128 load_swide_at(const uint8_t* base, uint32_t index, uint32_t size_bits) {
  const unsigned __int128 u = load_uwide_at(base, index, size_bits);
  if (size_bits >= 128)
    return __int128(u);
  const unsigned __int128 sign = (unsigned __int128)(1) << (size_bits - 1);
  if (u & sign)
    return __int128(u | ~(((unsigned __int128)(1) << size_bits) - 1));
  return __int128(u);
}

__int128 clamp_wide128(__int128 v, uint32_t size_bits, bool is_signed, bool saturate,
                       bool* sat);

int64_t load_int_elem(const uint8_t* p, uint32_t size_bits, bool is_signed) {
  return load_int_bits(p, 0, size_bits, is_signed);
}

void store_int_elem(uint8_t* p, uint32_t size_bits, int64_t v) {
  store_int_bits(p, 0, size_bits, v);
}

int64_t load_int_at(const uint8_t* base, uint32_t index, uint32_t size_bits,
                    bool is_signed) {
  return load_int_bits(base, index * size_bits, size_bits, is_signed);
}

void store_int_at(uint8_t* base, uint32_t index, uint32_t size_bits, int64_t v) {
  store_int_bits(base, index * size_bits, size_bits, v);
}

uint32_t square_index(uint32_t n, uint32_t i, uint32_t j) {
  return i * n + j;
}

float32_t load_f32(const uint8_t* p) {
  float32_t f;
  std::memcpy(&f.v, p, 4);
  return f;
}

void store_f32(uint8_t* p, float32_t f) {
  std::memcpy(p, &f.v, 4);
}

float64_t load_f64(const uint8_t* p) {
  float64_t f;
  std::memcpy(&f.v, p, 8);
  return f;
}

void store_f64(uint8_t* p, float64_t f) {
  std::memcpy(p, &f.v, 8);
}

bool is_int_family(uint32_t key) {
  return ame_is_int_family(key);
}

bool is_fp32(uint32_t key) {
  return ame_is_fp32(key);
}

bool is_fp_family(uint32_t key) {
  return ame_is_fp_family(key);
}

float32_t raw_to_f32(uint64_t raw, uint32_t key) {
  if (ame_is_fp32(key)) {
    float32_t f;
    f.v = uint32_t(raw);
    return f;
  }
  if (ame_is_fp16(key)) {
    float16_t h;
    h.v = uint16_t(raw);
    return f16_to_f32(h);
  }
  if (ame_is_bf16(key)) {
    bfloat16_t h;
    h.v = uint16_t(raw);
    return bf16_to_f32(h);
  }
  if (ame_is_fp64(key)) {
    float64_t f;
    f.v = raw;
    return f64_to_f32(f);
  }
  if (ame_is_e4m3(key)) {
    e4m3_t e;
    e.v = uint8_t(raw);
    return f16_to_f32(e4m3_to_f16(e));
  }
  if (ame_is_e5m2(key)) {
    e5m2_t e;
    e.v = uint8_t(raw);
    return f16_to_f32(e5m2_to_f16(e));
  }
  return i32_to_f32(0);
}

// dtype_enc is the raw descriptor: the E5M2 saturation mode follows the
// Inf bit (0 = saturate, 1 = deliver Inf per the rounding mode); E4M3 has
// no Inf encoding and always saturates.
uint64_t f32_to_raw(float32_t f, uint32_t dtype_enc) {
  if (ame_is_fp32(dtype_enc))
    return f.v;
  if (ame_is_fp16(dtype_enc))
    return f32_to_f16(f).v;
  if (ame_is_bf16(dtype_enc))
    return f32_to_bf16(f).v;
  if (ame_is_fp64(dtype_enc))
    return f32_to_f64(f).v;
  if (ame_is_e4m3(dtype_enc))
    return f32_to_e4m3(f, true).v;
  if (ame_is_e5m2(dtype_enc))
    return f32_to_e5m2(f, !ame_parse_dtype(dtype_enc).has_inf).v;
  return 0;
}

// Spec 6.2.3: a destination without an Inf encoding (datatype Inf bit 0)
// saturates a required infinity to the same-sign max finite. Covers FP32,
// FP16, BF16 and E5M2 raws; E4M3 caps inside f32_to_e4m3 already.
uint64_t saturate_inf_raw(uint64_t raw, uint32_t key, bool has_inf) {
  if (has_inf)
    return raw;
  uint8_t sat_ff = 0;
  if (ame_is_fp32(key)) {
    if ((raw & 0x7fffffffull) == 0x7f800000ull) {
      raw = (raw & 0x80000000ull) | 0x7f7fffffull;
      sat_ff = softfloat_flag_overflow | softfloat_flag_inexact;
    }
  } else if (ame_is_fp16(key)) {
    if ((raw & 0x7fffull) == 0x7c00ull) {
      raw = (raw & 0x8000ull) | 0x7bffull;
      sat_ff = softfloat_flag_overflow | softfloat_flag_inexact;
    }
  } else if (ame_is_bf16(key)) {
    if ((raw & 0x7fffull) == 0x7f80ull) {
      raw = (raw & 0x8000ull) | 0x7f7full;
      sat_ff = softfloat_flag_overflow | softfloat_flag_inexact;
    }
  } else if (ame_is_e5m2(key)) {
    if ((raw & 0x7full) == 0x7cull) {
      raw = (raw & 0x80ull) | 0x7bull;
      sat_ff = softfloat_flag_overflow | softfloat_flag_inexact;
    }
  }
  // Saturating an infinity to the max finite value is an inexact overflow;
  // the caller must accrue softfloat_exceptionFlags after this call.
  softfloat_exceptionFlags |= sat_ff;
  return raw;
}

// f32 -> narrow store-back with the destination rounding mode, Inf-bit
// saturation and RNO jam. Returns the accrued conversion flags.
uint8_t narrow_store_back(uint32_t key, const ame_dtype_t& dd,
                          uint32_t nelem, const uint8_t* fd, uint8_t* d) {
  softfloat_roundingMode = ame_fp_rm_to_softfloat(dd.fp_rm);
  softfloat_exceptionFlags = 0;
  const bool jam = ame_rm_is_rno(dd.fp_rm);
  uint8_t accrued = 0;
  for (uint32_t i = 0; i < nelem; i++) {
    // Per-element reset: the jam must see THIS conversion's flags, not a
    // delta against flags already accrued by earlier elements.
    softfloat_exceptionFlags = 0;
    uint64_t raw = f32_to_raw(load_f32(fd + size_t(i) * 4), dd.raw);
    raw = saturate_inf_raw(raw, key, dd.has_inf);
    const uint8_t elem_ff = softfloat_exceptionFlags;
    // Jam on INEXACT only: an OF result is already the saturating max and
    // forcing its LSB would turn E4M3's 0x7E into the NaN encoding 0x7F.
    if (jam && (elem_ff & softfloat_flag_inexact) &&
        !(elem_ff & softfloat_flag_overflow))
      raw |= 1;
    accrued |= elem_ff;
    store_int_at(d, i, dd.size_bits, int64_t(raw));
  }
  softfloat_exceptionFlags = accrued;
  return accrued;
}

using u128 = unsigned __int128;

__int128 round_div2(__int128 v, uint8_t rm);

struct u256w {
  u128 hi, lo;
};

u256w mul_wide_u128(u128 x, u128 y) {
  const u128 x0 = (uint64_t)x, x1 = x >> 64;
  const u128 y0 = (uint64_t)y, y1 = y >> 64;
  const u128 p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
  const u128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
  u256w r;
  r.lo = (mid << 64) | (uint64_t)p00;
  r.hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
  return r;
}

bool add256(u256w& x, const u256w& y) {
  const u128 lo = x.lo + y.lo;
  const u128 carry = lo < x.lo ? 1 : 0;
  const u128 hi = x.hi + y.hi + carry;
  const bool cout = hi < x.hi;
  x.lo = lo;
  x.hi = hi;
  return cout;
}

// x -= y, requires x >= y
void sub256(u256w& x, const u256w& y) {
  const u128 lo = x.lo - y.lo;
  const u128 borrow = x.lo < y.lo ? 1 : 0;
  x.hi = x.hi - y.hi - borrow;
  x.lo = lo;
}

int cmp256(const u256w& x, const u256w& y) {
  if (x.hi != y.hi)
    return x.hi < y.hi ? -1 : 1;
  return x.lo == y.lo ? 0 : (x.lo < y.lo ? -1 : 1);
}

// Signed-magnitude sum of two magnitudes given their signs.
void add_mag256(bool sneg, const u256w& ma, bool tneg, const u256w& mb,
                bool& rneg, u256w& rmag) {
  if (sneg == tneg) {
    rneg = sneg;
    rmag = ma;
    add256(rmag, mb);
  } else {
    const int c = cmp256(ma, mb);
    if (c == 0) {
      rneg = false;
      rmag = u256w{0, 0};
    } else if (c > 0) {
      rneg = sneg;
      rmag = ma;
      sub256(rmag, mb);
    } else {
      rneg = tneg;
      rmag = mb;
      sub256(rmag, ma);
    }
  }
}

// True result magnitude -> stored 128-bit pattern with SAT handling.
// Signed hi = 2^127-1; a magnitude above it saturates to the sign's extreme.
// `saturate` comes from the destination datatype's SAT bit: a saturating
// destination clamps to the sign extreme and raises amexsat; a
// non-saturating destination keeps the low 128 bits of the exact result.
u128 fit_int128(bool neg, const u256w& m, bool saturate, bool& sat) {
  const u128 hi = ((u128)1 << 127) - 1;
  if (m.hi != 0 || m.lo > hi) {
    if (!saturate)
      return neg ? (u128)0 - m.lo : m.lo;
    sat = true;
    return neg ? ((u128)1 << 127) : hi;
  }
  return neg ? (u128)0 - m.lo : m.lo;
}

// Unsigned: a magnitude above 2^128-1 saturates to all-ones when the
// destination saturates; a non-saturating destination keeps the low bits.
u128 fit_uint128_capped(const u256w& m, bool saturate, bool& sat) {
  if (m.hi != 0) {
    if (!saturate)
      return m.lo;
    sat = true;
    return ~(u128)0;
  }
  return m.lo;
}

// ---- Saturation-only bucket accumulator (matmul) ----------------------------
// Saturating integer matmul only needs to know WHICH side of the range the
// true sum landed on, never the exact value. Split each mod-2^128 term into
// a sign and magnitude and accumulate magnitudes into per-sign 320-bit
// buckets; at the end one compare per range bound decides clamp vs exact.
// Bucket capacity: n_sq*n terms of 256-bit magnitudes; n_sq <= 8 (M-reg
// geometry) and n <= 32, so 256 + 8 bits is ample — 320 is generous.
struct bucket320 {
  u128 w[3] = {0, 0, 0}; // little-endian words (value-initialized)
  void add(const u256w& m) {
    const u128 c0 = w[0] + m.lo;
    u128 c = c0 < w[0] ? 1 : 0;
    w[0] = c0;
    const u128 s1 = w[1] + m.hi;
    const u128 c1 = s1 < w[1] ? 1 : 0;
    w[1] = s1 + c;
    c = (w[1] < s1) ? 1 : c1;
    if (c) w[2] += 1; // cannot overflow at these sizes
  }
  bool ge(u128 bound) const { // >= a 128-bit bound
    return w[2] != 0 || w[1] != 0 || w[0] >= bound;
  }
  bool is_zero() const { return w[0] == 0 && w[1] == 0 && w[2] == 0; }
};

// Signed magnitude of a mod-2^128 product term.
void bucket_term(__int128 v, bucket320& pos, bucket320& neg) {
  if (v < 0)
    neg.add(u256w{0, (u128)0 - (u128)v});
  else
    pos.add(u256w{0, (u128)v});
}

// Return 2^bits for the supported integer widths.  2^128 is represented by
// the caller's bucket comparison because it cannot be represented in u128.
u128 int_width_limit(uint32_t bits) {
  return bits == 128 ? 0 : (u128)1 << bits;
}

bool bucket_ge_power_of_two(const bucket320& value, uint32_t bits) {
  if (bits == 128)
    return value.w[2] != 0 || value.w[1] != 0;
  return value.w[2] != 0 || value.w[1] != 0 ||
         value.w[0] >= int_width_limit(bits);
}

bool raw_is_negative(u128 raw, uint32_t bits) {
  return (raw & ((u128)1 << (bits - 1))) != 0;
}

u128 raw_magnitude(u128 raw, uint32_t bits) {
  if (!raw_is_negative(raw, bits))
    return raw;
  if (bits == 128)
    return (u128)0 - raw;
  const u128 mask = ((u128)1 << bits) - 1;
  return ((u128)0 - raw) & mask;
}

// Signed saturating resolve: pos-neg vs [-2^(bits-1), 2^(bits-1)-1].
u128 bucket_resolve_signed(const bucket320& pos, const bucket320& neg,
                           uint32_t bits, bool& sat) {
  const u128 min_mag = (u128)1 << (bits - 1);
  const u128 max_pos = min_mag - 1;
  // Exact decision by magnitude compare: pos - neg > hi <=> pos > neg + hi,
  // and neg - pos >= min_mag <=> neg >= pos + min_mag.
  auto add_bound = [](const bucket320& b, u128 bound) {
    bucket320 r = b;
    const u128 c0 = r.w[0] + bound;
    u128 c = c0 < r.w[0] ? 1 : 0;
    r.w[0] = c0;
    const u128 s1 = r.w[1] + c;
    c = s1 < r.w[1] ? 1 : 0;
    r.w[1] = s1;
    if (c) r.w[2] += 1;
    return r;
  };
  auto cmp = [](const bucket320& x, const bucket320& y) {
    for (int i = 2; i >= 0; i--) {
      if (x.w[i] != y.w[i])
        return x.w[i] < y.w[i] ? -1 : 1;
    }
    return 0;
  };
  if (cmp(pos, add_bound(neg, max_pos)) > 0) {
    sat = true;
    return max_pos;
  }
  if (cmp(neg, add_bound(pos, min_mag)) >= 0) {
    sat = true;
    return min_mag;
  }
  // In range: the true value fits; recover it exactly (no cancellation
  // overflow at these widths since both buckets are <= the totals).
  bucket320 big = pos, small = neg;
  bool negres = false;
  if (cmp(pos, neg) < 0) {
    big = neg;
    small = pos;
    negres = true;
  }
  // big - small (both <= max term sum; difference fits 256 bits)
  u256w d{big.w[1], big.w[0]};
  const u256w sm{small.w[1], small.w[0]};
  sub256(d, sm);
  const u128 lo = d.lo;
  return negres ? (u128)0 - lo : lo;
}

// Unsigned saturating resolve: pos-neg vs [0, 2^bits-1].
u128 bucket_resolve_unsigned(const bucket320& pos, const bucket320& neg,
                             uint32_t bits, bool& sat) {
  if (!neg.is_zero()) {
    sat = true;
    return 0;
  }
  if (bucket_ge_power_of_two(pos, bits)) {
    sat = true;
    return bits == 128 ? ~(u128)0 : int_width_limit(bits) - 1;
  }
  return pos.w[0];
}

// Halve a magnitude per the integer RM (d = 1 discarded bit).
//
// `neg` selects the sign of the true quotient.  RNU and RDN are asymmetric
// around zero (ties toward +/-inf), so for a negative quotient the roles
// swap relative to the magnitude: RNU truncates the magnitude and RDN
// rounds away from zero.  RNE and ROD are sign-symmetric.  The <=64-bit
// paths use round_div2 on the signed value, which already has this
// behavior; the 128-bit paths must round through the magnitude and
// therefore need the correction here.
u256w halve_mag_rm(const u256w& m, bool neg, uint8_t int_rm) {
  const bool odd = m.lo & 1;
  u256w q{m.hi >> 1, (m.lo >> 1) | ((m.hi & 1) << 127)};
  switch (int_rm & 3) {
    case 0: // RNU: ties toward +inf
      if (!neg && odd) {
        const u128 lo = q.lo + 1;
        if (lo == 0)
          q.hi += 1;
        q.lo = lo;
      }
      break;
    case 1: // RNE: tie -> even
      if (odd && (q.lo & 1)) {
        const u128 lo = q.lo + 1;
        if (lo == 0)
          q.hi += 1;
        q.lo = lo;
      }
      break;
    case 3: // ROD: jam
      if (odd)
        q.lo |= 1;
      break;
    default: // RDN: ties toward -inf
      if (neg && odd) {
        const u128 lo = q.lo + 1;
        if (lo == 0)
          q.hi += 1;
        q.lo = lo;
      }
      break;
  }
  return q;
}

} // namespace

namespace {

// Defined with the FP helpers below; used by matmul/ew/ldexp before that point.
uint8_t ame_fp_rm_to_softfloat(uint8_t fp_rm);
bool ame_rm_is_rno(uint8_t fp_rm);
float32_t f32_host_overflow_fixup(float32_t v, bool input_finite);
float32_t f32_neg(float32_t a);
float64_t f64_neg(float64_t a);

} // namespace

bool ame_matmul_supported(uint32_t a_dtype, uint32_t b_dtype, uint32_t c_dtype,
                          uint32_t udsz) {
  if (!ame_dtype_supported_m(a_dtype, udsz) || a_dtype == 0)
    return false;
  if (!ame_dtype_supported_m(b_dtype, udsz) || b_dtype == 0)
    return false;
  if (!ame_dtype_supported_acc(c_dtype, udsz) || c_dtype == 0)
    return false;

  const uint32_t ak = ame_dtype_family_key(a_dtype);
  const uint32_t bk = ame_dtype_family_key(b_dtype);
  const uint32_t ck = ame_dtype_family_key(c_dtype);

  // Integer sources may have independent signedness.  Their element widths
  // still have to match; the destination width is allowed to widen or
  // narrow according to the destination datatype rules below.
  if (is_int_family(ak) && is_int_family(bk)) {
    if (ame_parse_dtype(a_dtype).size_bits != ame_parse_dtype(b_dtype).size_bits)
      return false;
    return is_int_family(ck);
  }

  // Floating-point source formats remain an exact tuple match.
  if (ak != bk)
    return false;

  return is_fp_family(ak) && is_fp_family(ck);
}

bool ame_host_matmul(uint32_t n, uint32_t n_sq,
                     uint32_t a_dtype, const uint8_t* a,
                     uint32_t b_dtype, const uint8_t* b,
                     uint32_t c_dtype, uint8_t* c,
                     bool accumulate, bool negate,
                     bool transpose_a, bool transpose_b,
                     uint8_t* fflags, uint8_t* xsat) {
  const ame_dtype_t da = ame_parse_dtype(a_dtype);
  const ame_dtype_t db = ame_parse_dtype(b_dtype);
  const ame_dtype_t dc = ame_parse_dtype(c_dtype);
  const uint32_t ak = ame_dtype_family_key(a_dtype);
  const uint32_t ck = ame_dtype_family_key(c_dtype);
  const uint32_t a_bits = da.size_bits;
  const uint32_t b_bits = db.size_bits;
  const uint32_t c_bits = dc.size_bits;

  auto a_at = [&](uint32_t s, uint32_t i, uint32_t q) -> uint32_t {
    const uint32_t r = transpose_a ? q : i;
    const uint32_t c = transpose_a ? i : q;
    return s * n * n + r * n + c;
  };
  auto b_at = [&](uint32_t s, uint32_t q, uint32_t j) -> uint32_t {
    const uint32_t r = transpose_b ? j : q;
    const uint32_t c = transpose_b ? q : j;
    return s * n * n + r * n + c;
  };

  auto bits_to_f32 = [](uint64_t raw, uint32_t key) -> float32_t {
    return raw_to_f32(raw, key);
  };
  auto f32_to_bits = [](float32_t f, uint32_t enc) -> uint64_t {
    return f32_to_raw(f, enc);
  };

  const bool afp = is_fp_family(ak);
  const bool cfp = is_fp_family(ck);
  if (ame_is_fp64(ak) && ame_is_fp64(ck)) {
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dc.fp_rm);
    softfloat_exceptionFlags = 0;
    const bool jam = ame_rm_is_rno(dc.fp_rm);
    uint8_t kern_ff = 0;
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        float64_t acc;
        if (accumulate) {
          acc.v = uint64_t(load_int_at(c, i * n + j, c_bits, false));
        } else {
          acc.v = 0;
        }
        for (uint32_t s = 0; s < n_sq; s++) {
          for (uint32_t q = 0; q < n; q++) {
            float64_t av, bv;
            av.v = uint64_t(load_uwide_at(a, a_at(s, i, q), a_bits));
            bv.v = uint64_t(load_uwide_at(b, b_at(s, q, j), b_bits));
            // Sail per-step destination FMA; negate an input (exact) so
            // directed modes see the sign of the true term.
            softfloat_exceptionFlags = 0;
            acc = negate ? f64_mulAdd(f64_neg(av), bv, acc)
                         : f64_mulAdd(av, bv, acc);
            if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
                !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
                !isNaNF64UI(acc.v))
              acc.v |= 1;
            kern_ff |= softfloat_exceptionFlags;
          }
        }
        if (!dc.has_inf &&
            (acc.v & 0x7fffffffffffffffull) == 0x7ff0000000000000ull)
          acc.v = (acc.v & 0x8000000000000000ull) | 0x7fefffffffffffffull;
        store_int_at(c, i * n + j, c_bits, int64_t(acc.v));
      }
    }
    softfloat_exceptionFlags = kern_ff;
    if (fflags)
      *fflags |= uint8_t(softfloat_exceptionFlags & 0x1f);
    return true;
  }
  if (afp && cfp) {
    // Sail conformance: the accumulation is a destination-dtype FMA per
    // term — exact product plus acc, rounded once to the destination each
    // step. Wide accumulation with an end-of-cell narrowing diverges as
    // soon as per-step rounding matters (E5M2/BF16 directed cases).
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dc.fp_rm);
    softfloat_exceptionFlags = 0;
    const bool jam = ame_rm_is_rno(dc.fp_rm);
    uint8_t kern_ff = 0;
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        if (ame_is_fp32(ck)) {
          float32_t acc = accumulate
              ? bits_to_f32(uint64_t(load_int_at(c, i * n + j, c_bits, false)), ck)
              : i32_to_f32(0);
          for (uint32_t s = 0; s < n_sq; s++) {
            for (uint32_t q = 0; q < n; q++) {
              const float32_t av =
                  bits_to_f32(uint64_t(load_uwide_at(a, a_at(s, i, q), a_bits)), ak);
              const float32_t bv =
                  bits_to_f32(uint64_t(load_uwide_at(b, b_at(s, q, j), b_bits)), ak);
              softfloat_exceptionFlags = 0;
              acc = negate ? f32_mulAdd(f32_neg(av), bv, acc)
                           : f32_mulAdd(av, bv, acc);
              if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
                  !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
                  !isNaNF32UI(acc.v))
                acc.v |= 1;
              kern_ff |= softfloat_exceptionFlags;
            }
          }
          uint64_t raw = acc.v;
          raw = saturate_inf_raw(raw, c_dtype, dc.has_inf);
          kern_ff |= softfloat_exceptionFlags;
          store_int_at(c, i * n + j, c_bits, int64_t(raw));
        } else if (ame_is_fp16(ck) || ame_is_bf16(ck)) {
          // Native fused ops: exact product + acc, one rounding to dest.
          const bool isf16 = ame_is_fp16(ck);
          uint64_t acc = accumulate
              ? load_uwide_at(c, i * n + j, c_bits)
              : 0;
          for (uint32_t s = 0; s < n_sq; s++) {
            for (uint32_t q = 0; q < n; q++) {
              const uint64_t avr = load_uwide_at(a, a_at(s, i, q), a_bits);
              const uint64_t bvr = load_uwide_at(b, b_at(s, q, j), b_bits);
              softfloat_exceptionFlags = 0;
              if (isf16) {
                const float16_t av16{uint16_t(avr)}, bv16{uint16_t(bvr)};
                float16_t ac16{uint16_t(acc)};
                acc = (negate ? f16_mulAdd(float16_t{uint16_t(av16.v ^ 0x8000)}, bv16, ac16)
                              : f16_mulAdd(av16, bv16, ac16)).v;
                if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
                    !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
                    ((acc & 0x7C00) != 0x7C00 || (acc & 0x03FF) == 0))
                  acc |= 1;
              } else {
                const bfloat16_t avb{uint16_t(avr)}, bvb{uint16_t(bvr)};
                bfloat16_t acb{uint16_t(acc)};
                acc = (negate ? bf16_mulAdd(bfloat16_t{uint16_t(avb.v ^ 0x8000)}, bvb, acb)
                              : bf16_mulAdd(avb, bvb, acb)).v;
                if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
                    !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
                    ((acc & 0x7F80) != 0x7F80 || (acc & 0x007F) == 0))
                  acc |= 1;
              }
              kern_ff |= softfloat_exceptionFlags;
            }
          }
          uint64_t raw = saturate_inf_raw(acc, c_dtype, dc.has_inf);
          store_int_at(c, i * n + j, c_bits, int64_t(raw));
        } else {
          // E4M3/E5M2 have no fused primitive. Emulate dest-step FMA:
          // the product and the sum are exact in f64 for these ranges, so
          // one directed rounding per step to the destination (via an
          // f32 hop with the same rounding mode) matches Sail.
          uint64_t acc = accumulate
              ? load_uwide_at(c, i * n + j, c_bits)
              : 0;
          for (uint32_t s = 0; s < n_sq; s++) {
            for (uint32_t q = 0; q < n; q++) {
              const float32_t av =
                  bits_to_f32(uint64_t(load_uwide_at(a, a_at(s, i, q), a_bits)), ak);
              const float32_t bv =
                  bits_to_f32(uint64_t(load_uwide_at(b, b_at(s, q, j), b_bits)), ak);
              float64_t av64 = f32_to_f64(av);
              if (negate)
                av64 = f64_neg(av64);
              const float64_t acc64 =
                  f32_to_f64(bits_to_f32(uint64_t(acc), ck));
              softfloat_exceptionFlags = 0;
              const float64_t t = f64_add(acc64, f64_mul(av64, f32_to_f64(bv)));
              kern_ff |= softfloat_exceptionFlags;
              softfloat_exceptionFlags = 0;
              const float32_t tf = f64_to_f32(t);
              kern_ff |= softfloat_exceptionFlags;
              softfloat_exceptionFlags = 0;
              uint64_t raw = f32_to_raw(tf, c_dtype);
              raw = saturate_inf_raw(raw, c_dtype, dc.has_inf);
              if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
                  !(softfloat_exceptionFlags & softfloat_flag_overflow))
                raw |= 1;
              kern_ff |= softfloat_exceptionFlags;
              acc = raw;
            }
          }
          store_int_at(c, i * n + j, c_bits, int64_t(acc));
        }
      }
    }
    softfloat_exceptionFlags = kern_ff;
    if (fflags)
      *fflags |= uint8_t(softfloat_exceptionFlags & 0x1f);
    return true;
  }

  if (!is_int_family(ak) || !is_int_family(ck))
    return false;

  if (dc.saturate) {
    // Saturating lane: mod-2^128 accumulation cannot decide saturation once
    // the true sum crosses a power-of-two boundary (n_sq*n u64 products can
    // exceed 2^128), so accumulate magnitudes per sign and resolve once.
    bool sat_any = false;
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        bucket320 pos, neg;
        if (accumulate) {
          // The initial Acc value fits its dtype; bucket its true magnitude.
        const u128 d0 = load_uwide_at(c, i * n + j, c_bits);
        if (dc.is_signed) {
          const bool dneg = raw_is_negative(d0, c_bits);
          if (dneg)
            neg.add(u256w{0, raw_magnitude(d0, c_bits)});
          else
            pos.add(u256w{0, d0});
        } else {
          pos.add(u256w{0, d0});
          }
        }
        for (uint32_t s = 0; s < n_sq; s++) {
          for (uint32_t q = 0; q < n; q++) {
            // Exact 256-bit product bucketed by true sign: a mod-2^128
            // product reinterpreted as signed would poison the buckets
            // once the true sum crosses 2^127/2^128 boundaries.
            const u128 araw = load_uwide_at(a, a_at(s, i, q), a_bits);
            const u128 braw = load_uwide_at(b, b_at(s, q, j), b_bits);
            bool term_neg;
            u256w pm;
            if (da.is_signed || db.is_signed) {
              const bool sa = da.is_signed && raw_is_negative(araw, a_bits);
              const bool sb = db.is_signed && raw_is_negative(braw, b_bits);
              const u128 ma = da.is_signed ? raw_magnitude(araw, a_bits) : araw;
              const u128 mb = db.is_signed ? raw_magnitude(braw, b_bits) : braw;
              pm = mul_wide_u128(ma, mb);
              term_neg = (sa != sb);
            } else {
              pm = mul_wide_u128(araw, braw);
              term_neg = false;
            }
            if (negate)
              term_neg = !term_neg;
            if (term_neg)
              neg.add(pm);
            else
              pos.add(pm);
          }
        }
        bool sat = false;
        const u128 stored = dc.is_signed
                                ? bucket_resolve_signed(pos, neg, c_bits, sat)
                                : bucket_resolve_unsigned(pos, neg, c_bits, sat);
        store_uwide_at(c, i * n + j, c_bits, stored);
        sat_any |= sat;
      }
    }
    if (xsat && sat_any)
      *xsat |= 1;
    return true;
  }

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      __int128 acc = 0;
      if (accumulate) {
        acc = dc.is_signed ? load_swide_at(c, i * n + j, c_bits)
                           : __int128(load_uwide_at(c, i * n + j, c_bits));
      }
      for (uint32_t s = 0; s < n_sq; s++) {
        for (uint32_t q = 0; q < n; q++) {
          const __int128 av = da.is_signed
              ? load_swide_at(a, a_at(s, i, q), a_bits)
              : __int128(load_uwide_at(a, a_at(s, i, q), a_bits));
          const __int128 bv = db.is_signed
              ? load_swide_at(b, b_at(s, q, j), b_bits)
              : __int128(load_uwide_at(b, b_at(s, q, j), b_bits));
          const __int128 prod = av * bv;
          acc = negate ? acc - prod : acc + prod;
        }
      }
      store_uwide_at(c, i * n + j, c_bits, (unsigned __int128)(acc));
    }
  }
  return true;
}

void ame_copy_elem(uint8_t* dst, uint32_t dst_index,
                   const uint8_t* src, uint32_t src_index, uint32_t size_bits) {
  store_uwide_at(dst, dst_index, size_bits, load_uwide_at(src, src_index, size_bits));
}

void ame_transpose_squares(uint32_t n, uint32_t p, uint32_t e_bits,
                           const uint8_t* src, uint8_t* dst) {
  for (uint32_t sq = 0; sq < p; sq++) {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        ame_copy_elem(dst, sq * n * n + j * n + i,
                      src, sq * n * n + i * n + j, e_bits);
      }
    }
  }
}

void ame_reg_to_st(uint32_t n, uint32_t p, uint32_t e_bits,
                   const uint8_t* src, uint8_t* dst) {
  // Spec 2.6.2: N segments of P*N*E bits. Element (sq,i,j) is at bit
  // offset (sq*N+j)*E within segment i (sub-squares side by side).
  // Indexing is per-segment so ceil(P*N*E/8) padding stays at each
  // segment tail and is not consumed as elements of the next row.
  const uint32_t seg_bytes = ame_packed_bytes(p * n, e_bits);
  for (uint32_t sq = 0; sq < p; sq++) {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        ame_copy_elem(dst + i * seg_bytes, sq * n + j,
                      src, sq * n * n + i * n + j, e_bits);
      }
    }
  }
}

void ame_st_to_reg(uint32_t n, uint32_t p, uint32_t e_bits,
                   const uint8_t* src, uint8_t* dst) {
  const uint32_t seg_bytes = ame_packed_bytes(p * n, e_bits);
  for (uint32_t sq = 0; sq < p; sq++) {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        ame_copy_elem(dst, sq * n * n + i * n + j,
                      src + i * seg_bytes, sq * n + j, e_bits);
      }
    }
  }
}

void ame_reg_to_tst(uint32_t n, uint32_t p, uint32_t e_bits,
                    const uint8_t* src, uint8_t* dst) {
  // Spec 2.6.2: P*N segments of N*E bits. Segment sq*N+j is column j
  // of square sq; element (sq,i,j) starts at bit offset i*E.
  const uint32_t seg_bytes = ame_packed_bytes(n, e_bits);
  for (uint32_t sq = 0; sq < p; sq++) {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        ame_copy_elem(dst + (sq * n + j) * seg_bytes, i,
                      src, sq * n * n + i * n + j, e_bits);
      }
    }
  }
}

void ame_tst_to_reg(uint32_t n, uint32_t p, uint32_t e_bits,
                    const uint8_t* src, uint8_t* dst) {
  const uint32_t seg_bytes = ame_packed_bytes(n, e_bits);
  for (uint32_t sq = 0; sq < p; sq++) {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        ame_copy_elem(dst, sq * n * n + i * n + j,
                      src + (sq * n + j) * seg_bytes, i, e_bits);
      }
    }
  }
}

namespace {

__int128 clamp_wide128(__int128 v, uint32_t size_bits, bool is_signed, bool saturate,
                       bool* sat) {
  if (size_bits == 0 || size_bits > 128)
    return v;
  if (!saturate) {
    if (size_bits >= 128)
      return v;
    const unsigned __int128 mask = ((unsigned __int128)(1) << size_bits) - 1;
    return __int128(static_cast<unsigned __int128>(v) & mask);
  }
  if (is_signed) {
    // size_bits == 128: sign_bit reinterpreted is INT128_MIN and sign_bit-1
    // is INT128_MAX, so the same bounds cover the full 128-bit range.
    const unsigned __int128 sign_bit = (unsigned __int128)(1) << (size_bits - 1);
    const __int128 lo = size_bits >= 128 ? __int128(sign_bit) : -__int128(sign_bit);
    const __int128 hi = __int128(sign_bit - 1);
    if (v < lo) { *sat = true; return lo; }
    if (v > hi) { *sat = true; return hi; }
    return v;
  }
  // Unsigned saturation. For <128-bit results computed in __int128, a negative
  // value is genuine underflow (inputs are non-negative) and clamps to 0; at
  // 128 bits mod-2^128 arithmetic cannot separate underflow from a legal value
  // >= 2^127, so the full range is kept (wrap, no amexsat).
  if (size_bits < 128 && v < 0) { *sat = true; return 0; }
  const unsigned __int128 uv = static_cast<unsigned __int128>(v);
  const unsigned __int128 uhi = size_bits >= 128
      ? ~(unsigned __int128)(0)
      : (((unsigned __int128)(1) << size_bits) - 1);
  if (uv > uhi) { *sat = true; return __int128(uhi); }
  return v;
}

// Convert a mathematical sign/magnitude without losing the unsigned 128th bit.
// Fold addition needs one extra carry bit; integer conversion uses hi == 0.
u128 convert_int_magnitude(bool neg, const u256w& mag,
                           const ame_dtype_t& dst, bool& sat) {
  const u128 mask = ~u128(0) >> (128 - dst.size_bits);
  if (dst.saturate) {
    if (!dst.is_signed && neg && (mag.hi || mag.lo)) {
      sat = true;
      return 0;
    }
    const u128 sign = u128(1) << (dst.size_bits - 1);
    const u128 limit = dst.is_signed ? (neg ? sign : sign - 1) : mask;
    if (mag.hi || mag.lo > limit) {
      sat = true;
      return (neg ? u128(0) - limit : limit) & mask;
    }
  }
  return (neg ? u128(0) - mag.lo : mag.lo) & mask;
}

int64_t clamp_wide(__int128 v, uint32_t size_bits, bool is_signed, bool saturate,
                   bool* sat) {
  return int64_t(uint64_t(clamp_wide128(v, size_bits, is_signed, saturate, sat)));
}

// ---- Exact 128-bit integer lane -------------------------------------------
// __int128 arithmetic wraps, so 128-bit results that can leave the range are
// computed on magnitudes (spec 6.2.2: evaluate exactly, convert once). The
// product of two 128-bit values needs 256 bits; mag42 helpers below.
// One 128-bit signed element: exact evaluation then single conversion.
u128 ew_int128_signed(ame_ew_op_t op, const ame_dtype_t& dt, u128 ua, u128 ub,
                      u128 ud, bool& sat, bool& have) {
  const bool sga = (__int128)ua < 0, sgb = (__int128)ub < 0,
             sgd = (__int128)ud < 0;
  const u128 ma = sga ? (u128)0 - ua : ua;
  const u128 mb = sgb ? (u128)0 - ub : ub;
  const u128 md = sgd ? (u128)0 - ud : ud;
  have = true;
  switch (op) {
    case ame_ew_op_t::MIN:
      return (__int128)ua < (__int128)ub ? ua : ub;
    case ame_ew_op_t::MAX:
      return (__int128)ua > (__int128)ub ? ua : ub;
    case ame_ew_op_t::ABS: {
      const u256w m{0, ma};
      return fit_int128(false, m, dt.saturate, sat); // |INT128_MIN| = 2^127 saturates
    }
    case ame_ew_op_t::ABSDIFF: {
      u256w m;
      bool neg;
      add_mag256(sga, u256w{0, ma}, !sgb, u256w{0, mb}, neg, m);
      return fit_int128(false, m, dt.saturate, sat); // |a-b| >= 0
    }
    case ame_ew_op_t::ADD:
    case ame_ew_op_t::SUB: {
      // ADD: a+b; SUB: b-a == b + (-a)
      u256w m;
      bool neg;
      if (op == ame_ew_op_t::ADD)
        add_mag256(sga, u256w{0, ma}, sgb, u256w{0, mb}, neg, m);
      else
        add_mag256(sgb, u256w{0, mb}, !sga, u256w{0, ma}, neg, m);
      return fit_int128(neg, m, dt.saturate, sat);
    }
    case ame_ew_op_t::HDIFF:
    case ame_ew_op_t::MEAN: {
      // (b-a)/2 or (a+b)/2, rounded per the datatype RM before conversion.
      u256w m;
      bool neg;
      if (op == ame_ew_op_t::MEAN)
        add_mag256(sga, u256w{0, ma}, sgb, u256w{0, mb}, neg, m);
      else
        add_mag256(sgb, u256w{0, mb}, !sga, u256w{0, ma}, neg, m);
      return fit_int128(neg, halve_mag_rm(m, neg, dt.int_rm), dt.saturate, sat);
    }
    default: {
      // product family: p = x*y with signs per op
      bool pneg = sga != sgb;
      u256w pm = mul_wide_u128(ma, mb);
      switch (op) {
        case ame_ew_op_t::MUL:
          return fit_int128(pneg, pm, dt.saturate, sat);
        case ame_ew_op_t::MULNEG:
          return fit_int128(!pneg, pm, dt.saturate, sat);
        case ame_ew_op_t::MULACC: // dest + a*b
        case ame_ew_op_t::MULACCNEG: { // dest - a*b
          u256w m;
          bool neg;
          add_mag256(sgd, u256w{0, md}, op == ame_ew_op_t::MULACC ? pneg : !pneg,
                     pm, neg, m);
          return fit_int128(neg, m, dt.saturate, sat);
        }
        case ame_ew_op_t::MULADD: { // a + b*dest
          u256w pm2 = mul_wide_u128(mb, md);
          const bool p2neg = sgb != sgd;
          u256w m;
          bool neg;
          add_mag256(sga, u256w{0, ma}, p2neg, pm2, neg, m);
          return fit_int128(neg, m, dt.saturate, sat);
        }
        case ame_ew_op_t::MULSUB: { // a - b*dest
          u256w pm2 = mul_wide_u128(mb, md);
          const bool p2neg = sgb != sgd;
          u256w m;
          bool neg;
          add_mag256(sga, u256w{0, ma}, !p2neg, pm2, neg, m);
          return fit_int128(neg, m, dt.saturate, sat);
        }
        default:
          have = false;
          return 0;
      }
    }
  }
}

// One 128-bit unsigned element.
u128 ew_int128_unsigned(ame_ew_op_t op, const ame_dtype_t& dt, u128 ua, u128 ub,
                        u128 ud, bool& sat, bool& have) {
  have = true;
  switch (op) {
    case ame_ew_op_t::MIN:
      return ua < ub ? ua : ub;
    case ame_ew_op_t::MAX:
      return ua > ub ? ua : ub;
    case ame_ew_op_t::ABS:
      return ua;
    case ame_ew_op_t::ABSDIFF:
      return ua > ub ? ua - ub : ub - ua;
    case ame_ew_op_t::ADD: {
      const u128 r = ua + ub;
      if (r < ua && dt.saturate) {
        sat = true;
        return ~(u128)0;
      }
      return r; // non-saturating wrap keeps the low bits
    }
    case ame_ew_op_t::SUB:
      if (ub < ua && dt.saturate) {
        sat = true;
        return 0;
      }
      return ub - ua;
    case ame_ew_op_t::MEAN: {
      // Spec 6.2.2: evaluate exactly (no intermediate wrap), round once per
      // the datatype RM, then convert.  The wrapped-sum shortcut below the
      // 128-bit lane is exact only because those sums cannot overflow the
      // work range; at 128 bits the sum may need 129 bits.
      u256w m{0, 0};
      bool neg = false;
      add_mag256(false, u256w{0, ua}, false, u256w{0, ub}, neg, m);
      const u256w h = halve_mag_rm(m, neg, dt.int_rm);
      if (h.hi != 0) {
        if (dt.saturate) { sat = true; return ~(u128)0; }
        return h.lo; // wrap keeps the low 128 bits
      }
      sat = false;
      return h.lo;
    }
    case ame_ew_op_t::HDIFF: {
      // (b-a)/2: the exact difference is signed (b may be the smaller
      // operand), so compute the magnitude/sign pair, round per RM with the
      // signed asymmetry (RNU/RDN), then convert.
      u256w m;
      bool neg;
      add_mag256(false, u256w{0, ub}, true, u256w{0, ua}, neg, m);
      const u256w h = halve_mag_rm(m, neg, dt.int_rm);
      if (neg) {
        if (dt.saturate) { sat = true; return 0; }
        return (u128)0 - h.lo; // low 128 bits of the negative exact value
      }
      if (h.hi != 0) {
        if (dt.saturate) { sat = true; return ~(u128)0; }
        return h.lo;
      }
      sat = false;
      return h.lo;
    }
    case ame_ew_op_t::MUL: {
      const u256w p = mul_wide_u128(ua, ub);
      return fit_uint128_capped(p, dt.saturate, sat);
    }
    case ame_ew_op_t::MULNEG: { // -(a*b) <= 0: SAT clamps to 0, wrap keeps low bits
      const u256w p = mul_wide_u128(ua, ub);
      if ((p.hi || p.lo) && dt.saturate) {
        sat = true;
        return 0;
      }
      return (u128)0 - p.lo;
    }
    case ame_ew_op_t::MULACC: { // dest + a*b
      u256w p = mul_wide_u128(ua, ub);
      const u256w d{0, ud};
      // The 256-bit sum of product and dest always fits 256 bits, so the
      // u128-overflow signal is the post-add high word, not add256's carry.
      add256(p, d);
      if (p.hi != 0 && dt.saturate) {
        sat = true;
        return ~(u128)0;
      }
      return p.lo;
    }
    case ame_ew_op_t::MULADD: { // a + b*dest
      u256w p = mul_wide_u128(ub, ud);
      const u256w x{0, ua};
      add256(p, x);
      if (p.hi != 0 && dt.saturate) {
        sat = true;
        return ~(u128)0;
      }
      return p.lo;
    }
    case ame_ew_op_t::MULSUB: { // a - b*dest
      const u256w p = mul_wide_u128(ub, ud);
      if ((p.hi || p.lo > ua) && dt.saturate) {
        sat = true;
        return 0;
      }
      return ua - p.lo;
    }
    case ame_ew_op_t::MULACCNEG: { // dest - a*b
      const u256w p = mul_wide_u128(ua, ub);
      if ((p.hi || p.lo > ud) && dt.saturate) {
        sat = true;
        return 0;
      }
      return ud - p.lo;
    }
    default:
      have = false;
      return 0;
  }
}

int64_t clamp_int(int64_t v, uint32_t size_bits, bool is_signed, bool saturate,
                  bool* sat) {
  const __int128 wide = is_signed ? __int128(v) : __int128(uint64_t(v));
  return clamp_wide(wide, size_bits, is_signed, saturate, sat);
}

// Spec §4.1 integer rounding, d=1 discarded fraction bit. Arithmetic >> .
__int128 round_div2(__int128 v, uint8_t rm) {
  const __int128 q = v >> 1;
  const int bit0 = int(v & 1);
  const int bit1 = int((v >> 1) & 1);
  switch (rm & 3) {
    case 0: return q + bit0;                 // RNU
    case 1: return q + (bit0 && bit1);       // RNE (v[d-2:0] treated as 0)
    case 2: return q;                        // RDN
    default: return q | __int128(bit0 != 0); // ROD
  }
}

float32_t f32_neg(float32_t a) {
  return float32_t{a.v ^ 0x80000000u};
}

float32_t f32_canonical_abs(float32_t a) {
  if (isNaNF32UI(a.v))
    return float32_t{defaultNaNF32UI};
  return float32_t{a.v & 0x7fffffffu};
}

float32_t f32_ame_min(float32_t a, float32_t b) {
  if (isNaNF32UI(a.v) || isNaNF32UI(b.v)) {
    // A signaling NaN input quiets to the default NaN and raises NV; a
    // quiet NaN input propagates without NV (matches Sail/reference).
    if (softfloat_isSigNaNF32UI(a.v) || softfloat_isSigNaNF32UI(b.v))
      softfloat_exceptionFlags |= softfloat_flag_invalid;
    return float32_t{defaultNaNF32UI};
  }
  // Prefer -0 on a +0/-0 tie.
  if (((a.v | b.v) & 0x7fffffff) == 0)
    return float32_t{a.v | b.v}; // signed-or keeps -0 if either is -0
  return f32_lt_quiet(a, b) ? a : b;
}

float32_t f32_ame_max(float32_t a, float32_t b) {
  if (isNaNF32UI(a.v) || isNaNF32UI(b.v)) {
    if (softfloat_isSigNaNF32UI(a.v) || softfloat_isSigNaNF32UI(b.v))
      softfloat_exceptionFlags |= softfloat_flag_invalid;
    return float32_t{defaultNaNF32UI};
  }
  // Prefer +0 on a +0/-0 tie.
  if (((a.v | b.v) & 0x7fffffff) == 0)
    return float32_t{a.v & b.v}; // signed-and keeps +0 if either is +0
  return f32_lt_quiet(b, a) ? a : b;
}

float64_t f64_neg(float64_t a) {
  return float64_t{a.v ^ 0x8000000000000000ull};
}

float64_t f64_canonical_abs(float64_t a) {
  if (isNaNF64UI(a.v))
    return float64_t{defaultNaNF64UI};
  return float64_t{a.v & 0x7fffffffffffffffull};
}

float64_t f64_ame_min(float64_t a, float64_t b) {
  if (isNaNF64UI(a.v) || isNaNF64UI(b.v)) {
    if (softfloat_isSigNaNF64UI(a.v) || softfloat_isSigNaNF64UI(b.v))
      softfloat_exceptionFlags |= softfloat_flag_invalid;
    return float64_t{defaultNaNF64UI};
  }
  // Prefer -0 on a +0/-0 tie.
  if (((a.v | b.v) & 0x7fffffffffffffffull) == 0)
    return float64_t{a.v | b.v};
  return f64_lt_quiet(a, b) ? a : b;
}

float64_t f64_ame_max(float64_t a, float64_t b) {
  if (isNaNF64UI(a.v) || isNaNF64UI(b.v)) {
    if (softfloat_isSigNaNF64UI(a.v) || softfloat_isSigNaNF64UI(b.v))
      softfloat_exceptionFlags |= softfloat_flag_invalid;
    return float64_t{defaultNaNF64UI};
  }
  // Prefer +0 on a +0/-0 tie.
  if (((a.v | b.v) & 0x7fffffffffffffffull) == 0)
    return float64_t{a.v & b.v};
  return f64_lt_quiet(b, a) ? a : b;
}

uint8_t softfloat_to_ame_fflags(uint8_t sf) {
  // softfloat: inexact=1, underflow=2, overflow=4, infinite=8, invalid=16
  // RISC-V fflags: NX=1, UF=2, OF=4, DZ=8, NV=16 — same layout.
  return sf & 0x1f;
}

} // namespace

bool ame_host_ew(ame_ew_op_t op, uint32_t dtype, uint32_t nelem,
                 const uint8_t* a, const uint8_t* b, uint8_t* d,
                 uint8_t* fflags, uint8_t* xsat) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const uint32_t key = ame_dtype_family_key(dtype);
  const uint32_t eb = dt.size_bits / 8;
  if (dt.size_bits == 0 || nelem == 0)
    return false;
  if (!a || !d)
    return false;
  if (!ame_ew_is_unary(op) && !b)
    return false;
  if (dt.is_fp && dt.size_bits == 0)
    return false;

  if (ame_is_fp64(key)) {
    // Native f64 compute: routing FP64 through the f32 middle loses
    // precision and eats the overflow flags (f32 stage turns FP64 max into
    // Inf with no flags).
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
    const bool jam = ame_rm_is_rno(dt.fp_rm);
    const bool opcode_rm = (op == ame_ew_op_t::FRINTM ||
                            op == ame_ew_op_t::FRINTN ||
                            op == ame_ew_op_t::FRINTP ||
                            op == ame_ew_op_t::FRINTZ);
    const float64_t f64_half{0x3fe0000000000000ull};
    const float64_t f64_one{0x3ff0000000000000ull};
    uint8_t local_ff = 0;
    for (uint32_t i = 0; i < nelem; i++) {
      softfloat_exceptionFlags = 0;
      const float64_t av = load_f64(a + i * 8);
      const float64_t bv = b ? load_f64(b + i * 8) : float64_t{0};
      const float64_t destv = load_f64(d + i * 8);
      float64_t dv;
      switch (op) {
        case ame_ew_op_t::ADD: dv = f64_add(av, bv); break;
        case ame_ew_op_t::SUB: dv = f64_sub(bv, av); break; // B - A
        case ame_ew_op_t::MUL: dv = f64_mul(av, bv); break;
        case ame_ew_op_t::MIN: dv = f64_ame_min(av, bv); break;
        case ame_ew_op_t::MAX: dv = f64_ame_max(av, bv); break;
        case ame_ew_op_t::ABS: dv = f64_canonical_abs(av); break;
        case ame_ew_op_t::ABSDIFF: dv = f64_canonical_abs(f64_sub(av, bv)); break;
        case ame_ew_op_t::HDIFF: dv = f64_mul(f64_sub(bv, av), f64_half); break;
        case ame_ew_op_t::MEAN: dv = f64_mul(f64_add(av, bv), f64_half); break;
        case ame_ew_op_t::MULNEG: // negate-first, see the FP32 branch note
          dv = f64_mul(f64_neg(av), bv);
          if (isNaNF64UI(dv.v))
            dv = float64_t{defaultNaNF64UI};
          break;
        case ame_ew_op_t::MULACC: dv = f64_mulAdd(av, bv, destv); break;
        case ame_ew_op_t::MULACCNEG: dv = f64_mulAdd(f64_neg(av), bv, destv); break;
        case ame_ew_op_t::MULADD: dv = f64_mulAdd(bv, destv, av); break;
        case ame_ew_op_t::MULSUB: dv = f64_mulAdd(f64_neg(bv), destv, av); break;
        case ame_ew_op_t::SQRT: dv = f64_sqrt(av); break;
        case ame_ew_op_t::REC: dv = f64_div(f64_one, av); break;
        case ame_ew_op_t::RSQRT: dv = f64_div(f64_one, f64_sqrt(av)); break;
        // Spec 6.2.3 :252 ("Exact operations and exact conversions set none of
        // these flags") plus the mfrint* pages ("a representable integral
        // result is exact", instructions.adoc:1481/1516/1551/1586): removing
        // the input fraction is the operation itself, so it never sets NX.
        // exact=false keeps SoftFloat from signalling that rounding.
        case ame_ew_op_t::FRINTM:
          dv = f64_roundToInt(av, softfloat_round_min, false);
          break;
        case ame_ew_op_t::FRINTN:
          dv = f64_roundToInt(av, softfloat_round_near_even, false);
          break;
        case ame_ew_op_t::FRINTP:
          dv = f64_roundToInt(av, softfloat_round_max, false);
          break;
        case ame_ew_op_t::FRINTZ:
          dv = f64_roundToInt(av, softfloat_round_minMag, false);
          break;
        case ame_ew_op_t::COS:
        case ame_ew_op_t::SIN:
        case ame_ew_op_t::TANH:
        case ame_ew_op_t::EXP2:
        case ame_ew_op_t::LOG2: {
          if (isNaNF64UI(av.v)) {
            dv = float64_t{defaultNaNF64UI};
            break;
          }
          const bool inf =
              (av.v & 0x7fffffffffffffffull) == 0x7ff0000000000000ull;
          const bool neg = (av.v >> 63) != 0;
          const bool zero = (av.v & 0x7fffffffffffffffull) == 0;
          if (op == ame_ew_op_t::LOG2) {
            if (neg && !zero) {
              softfloat_exceptionFlags |= softfloat_flag_invalid;
              dv = float64_t{defaultNaNF64UI};
              break;
            }
            if (zero) {
              softfloat_exceptionFlags |= softfloat_flag_infinite;
              dv = float64_t{0xfff0000000000000ull}; // -inf
              break;
            }
          }
          if ((op == ame_ew_op_t::SIN || op == ame_ew_op_t::COS) && inf) {
            softfloat_exceptionFlags |= softfloat_flag_invalid;
            dv = float64_t{defaultNaNF64UI};
            break;
          }
          double hd, ho;
          std::memcpy(&hd, &av.v, 8);
          if (op == ame_ew_op_t::COS) ho = std::cos(hd);
          else if (op == ame_ew_op_t::SIN) ho = std::sin(hd);
          else if (op == ame_ew_op_t::TANH) ho = std::tanh(hd);
          else if (op == ame_ew_op_t::EXP2) ho = std::exp2(hd);
          else ho = std::log2(hd);
          std::memcpy(&dv.v, &ho, 8);
          // Spec 6.2.3 :248-250: NX covers "an implementation-defined
          // approximation that is not exact".  cos/sin are exact only at +-0;
          // tanh is also exact at +-inf (tanh(+-inf) = +-1).
          if ((op == ame_ew_op_t::COS || op == ame_ew_op_t::SIN ||
               op == ame_ew_op_t::TANH) &&
              !zero && !(op == ame_ew_op_t::TANH && inf))
            softfloat_exceptionFlags |= softfloat_flag_inexact;
          if (op == ame_ew_op_t::EXP2 && !inf &&
              (dv.v & 0x7fffffffffffffffull) == 0x7ff0000000000000ull &&
              (dv.v & 0x000fffffffffffffull) == 0) {
            // Host libm bypasses softfloat flags: restore OF|NX and the
            // directed / Inf-less-destination overflow result.
            softfloat_exceptionFlags |=
                uint8_t(softfloat_flag_overflow | softfloat_flag_inexact);
            if (!dt.has_inf || softfloat_roundingMode == softfloat_round_min ||
                softfloat_roundingMode == softfloat_round_max ||
                softfloat_roundingMode == softfloat_round_minMag)
              dv.v = (dv.v & 0x8000000000000000ull) | 0x7fefffffffffffffull;
          }
          if (isNaNF64UI(dv.v))
            dv = float64_t{defaultNaNF64UI};
          break;
        }
        default: return false;
      }
      if (ame_ew_reads_dest(op) && isNaNF64UI(dv.v))
        dv = float64_t{defaultNaNF64UI};
      // Spec 6.2.3: a required infinity with no dest Inf encoding saturates.
      if (!dt.has_inf &&
          (dv.v & 0x7fffffffffffffffull) == 0x7ff0000000000000ull &&
          (dv.v & 0x000fffffffffffffull) == 0)
        dv.v = (dv.v & 0x8000000000000000ull) | 0x7fefffffffffffffull;
      // RNO jam per element (opcode RM wins for mfrint*).
      if (!opcode_rm && jam &&
          (softfloat_exceptionFlags & softfloat_flag_inexact) &&
          !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
          !isNaNF64UI(dv.v))
        dv.v |= 1;
      local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      store_f64(d + i * 8, dv);
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (ame_is_bf16(key) &&
      (op == ame_ew_op_t::MULACC || op == ame_ew_op_t::MULACCNEG ||
       op == ame_ew_op_t::MULADD || op == ame_ew_op_t::MULSUB)) {
    // Ordinary FP32 rounding followed by BF16 rounding is not a fused result.
    // bf16_mulAdd uses an exact product and round-to-odd intermediate instead.
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
    const bool jam = ame_rm_is_rno(dt.fp_rm);
    uint8_t local_ff = 0;
    for (uint32_t i = 0; i < nelem; i++) {
      softfloat_exceptionFlags = 0;
      const bfloat16_t av{uint16_t(load_int_at(a, i, 16, false))};
      const bfloat16_t bv{uint16_t(load_int_at(b, i, 16, false))};
      const bfloat16_t old{uint16_t(load_int_at(d, i, 16, false))};
      bfloat16_t dv;
      switch (op) {
        case ame_ew_op_t::MULACC: dv = bf16_mulAdd(av, bv, old); break;
        case ame_ew_op_t::MULACCNEG:
          dv = bf16_mulAdd(bfloat16_t{uint16_t(av.v ^ 0x8000)}, bv, old);
          break;
        case ame_ew_op_t::MULADD: dv = bf16_mulAdd(bv, old, av); break;
        case ame_ew_op_t::MULSUB:
          dv = bf16_mulAdd(bfloat16_t{uint16_t(bv.v ^ 0x8000)}, old, av);
          break;
        default: return false;
      }
      if (isNaNBF16UI(dv.v))
        dv.v = defaultNaNBF16UI;
      dv.v = saturate_inf_raw(dv.v, key, dt.has_inf);
      if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
          !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
          !isNaNBF16UI(dv.v))
        dv.v |= 1;
      local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      store_int_at(d, i, 16, dv.v);
    }
    softfloat_exceptionFlags = local_ff;
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (is_fp_family(key) && !is_fp32(key)) {
    std::vector<uint8_t> fa(size_t(nelem) * 4), fd(size_t(nelem) * 4, 0);
    std::vector<uint8_t> fb;
    uint8_t conversion_ff = 0;
    auto convert_to_f32 = [&](const uint8_t* base, uint32_t index) {
      // Narrow-format conversion can itself raise NV for an sNaN.  Keep
      // that flag separate because the recursive FP32 path resets the
      // SoftFloat flags once per element.
      softfloat_exceptionFlags = 0;
      const float32_t value = raw_to_f32(
          uint64_t(load_int_at(base, index, dt.size_bits, false)), key);
      conversion_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      return value;
    };
    for (uint32_t i = 0; i < nelem; i++) {
      store_f32(fa.data() + i * 4, convert_to_f32(a, i));
      if (ame_ew_reads_dest(op))
        store_f32(fd.data() + i * 4, convert_to_f32(d, i));
    }
    const uint8_t* bptr = nullptr;
    if (b) {
      fb.assign(size_t(nelem) * 4, 0);
      for (uint32_t i = 0; i < nelem; i++)
        store_f32(fb.data() + i * 4, convert_to_f32(b, i));
      bptr = fb.data();
    }
    // Compute in f32 with the destination's rounding mode carried in the
    // synthetic FP32 descriptor (spec 6.2.3: rounding selects by dest dtype).
    const uint32_t f32dt = 0x20300120u | (uint32_t(dt.fp_rm & 0xf) << 22) | 0x00300000u;
    uint8_t compute_ff = 0;
    if (!ame_host_ew(op, f32dt, nelem, fa.data(), bptr, fd.data(),
                     &compute_ff, xsat))
      return false;
    // The f32->narrow store-back rounds and can raise NX/OF/UF; accrue them.
    const uint8_t store_ff = narrow_store_back(key, dt, nelem, fd.data(), d);
    const uint8_t local_ff = conversion_ff | compute_ff | store_ff;
    softfloat_exceptionFlags = local_ff;
    if (fflags)
      *fflags |= local_ff;
    return true;
  }

  uint8_t local_ff = 0;
  bool sat = false;
  const float32_t f32_half{0x3f000000u};

  if (is_fp32(key)) {
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
    softfloat_exceptionFlags = 0;
    for (uint32_t i = 0; i < nelem; i++) {
      softfloat_exceptionFlags = 0;
      const float32_t av = load_f32(a + i * eb);
      const float32_t bv = b ? load_f32(b + i * eb) : float32_t{0};
      const float32_t destv = load_f32(d + i * eb);
      float32_t dv;
      switch (op) {
        case ame_ew_op_t::ADD: dv = f32_add(av, bv); break;
        case ame_ew_op_t::SUB: dv = f32_sub(bv, av); break; // B - A
        case ame_ew_op_t::MUL: dv = f32_mul(av, bv); break;
        case ame_ew_op_t::MIN: dv = f32_ame_min(av, bv); break;
        case ame_ew_op_t::MAX: dv = f32_ame_max(av, bv); break;
        case ame_ew_op_t::ABS: dv = f32_canonical_abs(av); break;
        case ame_ew_op_t::ABSDIFF: dv = f32_canonical_abs(f32_sub(av, bv)); break;
        case ame_ew_op_t::HDIFF: dv = f32_mul(f32_sub(bv, av), f32_half); break;
        case ame_ew_op_t::MEAN: dv = f32_mul(f32_add(av, bv), f32_half); break;
        case ame_ew_op_t::MULNEG:
          // -(A*B) rounded once: negate an input (exact) so directed modes
          // see the sign of the true result; post-op negation rounds the
          // pre-negation value (BF16|RUP overflow gave -Inf, not -max).
          dv = f32_mul(f32_neg(av), bv);
          if (isNaNF32UI(dv.v))
            dv = float32_t{defaultNaNF32UI};
          break;
        case ame_ew_op_t::MULACC: dv = f32_mulAdd(av, bv, destv); break;
        case ame_ew_op_t::MULACCNEG: dv = f32_mulAdd(f32_neg(av), bv, destv); break;
        case ame_ew_op_t::MULADD: dv = f32_mulAdd(bv, destv, av); break;
        case ame_ew_op_t::MULSUB: dv = f32_mulAdd(f32_neg(bv), destv, av); break;
        case ame_ew_op_t::SQRT: dv = f32_sqrt(av); break;
        case ame_ew_op_t::REC: dv = f32_div(float32_t{0x3f800000u}, av); break;
        case ame_ew_op_t::RSQRT: {
          const float32_t s = f32_sqrt(av);
          dv = f32_div(float32_t{0x3f800000u}, s);
          break;
        }
        // Same as the FP64 FRINT* group above: the instruction's own rounding
        // to an integral value never sets NX (spec 6.2.3 :252).
        case ame_ew_op_t::FRINTM:
          dv = f32_roundToInt(av, softfloat_round_min, false);
          break;
        case ame_ew_op_t::FRINTN:
          dv = f32_roundToInt(av, softfloat_round_near_even, false);
          break;
        case ame_ew_op_t::FRINTP:
          dv = f32_roundToInt(av, softfloat_round_max, false);
          break;
        case ame_ew_op_t::FRINTZ:
          dv = f32_roundToInt(av, softfloat_round_minMag, false);
          break;
        case ame_ew_op_t::COS:
        case ame_ew_op_t::SIN:
        case ame_ew_op_t::TANH:
        case ame_ew_op_t::EXP2:
        case ame_ew_op_t::LOG2: {
          if (isNaNF32UI(av.v)) {
            if (softfloat_isSigNaNF32UI(av.v))
              softfloat_exceptionFlags |= softfloat_flag_invalid;
            dv = float32_t{defaultNaNF32UI};
            break;
          }
          const bool inf = (av.v & 0x7fffffff) == 0x7f800000u;
          const bool neg = (av.v >> 31) != 0;
          const bool zero = (av.v & 0x7fffffff) == 0;
          if (op == ame_ew_op_t::LOG2) {
            if (neg && !zero) {
              softfloat_exceptionFlags |= softfloat_flag_invalid;
              dv = float32_t{defaultNaNF32UI};
              break;
            }
            if (zero) {
              softfloat_exceptionFlags |= softfloat_flag_infinite;
              dv = float32_t{0xff800000u}; // -inf
              break;
            }
          }
          if ((op == ame_ew_op_t::SIN || op == ame_ew_op_t::COS) && inf) {
            softfloat_exceptionFlags |= softfloat_flag_invalid;
            dv = float32_t{defaultNaNF32UI};
            break;
          }
          float hf;
          std::memcpy(&hf, &av.v, 4);
          float ho;
          if (op == ame_ew_op_t::COS) ho = std::cos(hf);
          else if (op == ame_ew_op_t::SIN) ho = std::sin(hf);
          else if (op == ame_ew_op_t::TANH) ho = std::tanh(hf);
          else if (op == ame_ew_op_t::EXP2) ho = std::exp2(hf);
          else ho = std::log2(hf);
          std::memcpy(&dv.v, &ho, 4);
          // Spec 6.2.3 :248-250: NX covers "an implementation-defined
          // approximation that is not exact".  cos/sin are exact only at +-0;
          // tanh is also exact at +-inf (tanh(+-inf) = +-1).
          if ((op == ame_ew_op_t::COS || op == ame_ew_op_t::SIN ||
               op == ame_ew_op_t::TANH) &&
              !zero && !(op == ame_ew_op_t::TANH && inf))
            softfloat_exceptionFlags |= softfloat_flag_inexact;
          if (op == ame_ew_op_t::EXP2)
            dv = f32_host_overflow_fixup(dv, !inf);
          if (isNaNF32UI(dv.v))
            dv = float32_t{defaultNaNF32UI};
          break;
        }
        default: return false;
      }
      if (ame_ew_reads_dest(op) && isNaNF32UI(dv.v))
        dv = float32_t{defaultNaNF32UI};
      // Spec 6.2.3: a required infinity with no dest Inf encoding saturates.
      dv.v = saturate_inf_raw(dv.v, 0x20300120u, dt.has_inf);
      // RNO jam: an inexact FP32 result gets its LSB forced to 1 — except
      // under the mfrint* opcodes, whose rounding mode is opcode-selected
      // and whose result is integral by definition.
      const bool opcode_rm = (op == ame_ew_op_t::FRINTM ||
                              op == ame_ew_op_t::FRINTN ||
                              op == ame_ew_op_t::FRINTP ||
                              op == ame_ew_op_t::FRINTZ);
      if (!opcode_rm && ame_rm_is_rno(dt.fp_rm) &&
          (softfloat_exceptionFlags & softfloat_flag_inexact) &&
          !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
          !isNaNF32UI(dv.v))
        dv.v |= 1;
      local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      store_f32(d + i * eb, dv);
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (!is_int_family(key))
    return false;

  if (dt.size_bits == 128) {
    // Exact 128-bit lane: generic __int128 arithmetic wraps before the clamp
    // can see the true result, so saturating ops use magnitude arithmetic.
    // Bitwise ops are exact in any width and stay on the fast path here.
    const bool bitwise = op == ame_ew_op_t::AND || op == ame_ew_op_t::ANDNOT ||
                         op == ame_ew_op_t::OR || op == ame_ew_op_t::ORNOT ||
                         op == ame_ew_op_t::XOR;
    bool sat_any = false;
    for (uint32_t i = 0; i < nelem; i++) {
      const u128 ua = load_uwide_at(a, i, 128);
      const u128 ub = b ? load_uwide_at(b, i, 128) : 0;
      const u128 ud = load_uwide_at(d, i, 128);
      if (bitwise) {
        u128 r = 0;
        switch (op) {
          case ame_ew_op_t::AND: r = ua & ub; break;
          case ame_ew_op_t::ANDNOT: r = ua & ~ub; break;
          case ame_ew_op_t::OR: r = ua | ub; break;
          case ame_ew_op_t::ORNOT: r = ua | ~ub; break;
          case ame_ew_op_t::XOR: r = ua ^ ub; break;
          default: return false;
        }
        store_uwide_at(d, i, 128, r);
        continue;
      }
      bool sat = false, have = true;
      const u128 r = dt.is_signed
                         ? ew_int128_signed(op, dt, ua, ub, ud, sat, have)
                         : ew_int128_unsigned(op, dt, ua, ub, ud, sat, have);
      if (!have)
        return false;
      store_uwide_at(d, i, 128, r);
      sat_any |= sat;
    }
    if (xsat && sat_any && dt.saturate)
      *xsat |= 1;
    return true;
  }

  const bool bitwise =
      op == ame_ew_op_t::AND || op == ame_ew_op_t::ANDNOT ||
      op == ame_ew_op_t::OR || op == ame_ew_op_t::ORNOT ||
      op == ame_ew_op_t::XOR;
  if (bitwise) {
    const unsigned __int128 mask = dt.size_bits >= 128
        ? ~(unsigned __int128)(0)
        : ((unsigned __int128)(1) << dt.size_bits) - 1;
    for (uint32_t i = 0; i < nelem; i++) {
      const unsigned __int128 ua = load_uwide_at(a, i, dt.size_bits) & mask;
      const unsigned __int128 ub = load_uwide_at(b, i, dt.size_bits) & mask;
      unsigned __int128 ud = 0;
      switch (op) {
        case ame_ew_op_t::AND:    ud = ua & ub; break;
        case ame_ew_op_t::ANDNOT: ud = ua & (~ub & mask); break;
        case ame_ew_op_t::OR:     ud = ua | ub; break;
        case ame_ew_op_t::ORNOT:  ud = ua | (~ub & mask); break;
        case ame_ew_op_t::XOR:    ud = ua ^ ub; break;
        default: return false;
      }
      store_uwide_at(d, i, dt.size_bits, ud);
    }
    return true;
  }

  for (uint32_t i = 0; i < nelem; i++) {
    const __int128 av = dt.is_signed ? load_swide_at(a, i, dt.size_bits)
                                     : __int128(load_uwide_at(a, i, dt.size_bits));
    const __int128 bv = b ? (dt.is_signed ? load_swide_at(b, i, dt.size_bits)
                                          : __int128(load_uwide_at(b, i, dt.size_bits)))
                          : 0;
    const __int128 destv = dt.is_signed ? load_swide_at(d, i, dt.size_bits)
                                        : __int128(load_uwide_at(d, i, dt.size_bits));
    __int128 r = 0;
    bool have = true;
    if (!dt.is_signed) {
      const unsigned __int128 ua = (unsigned __int128)(av);
      const unsigned __int128 ub = (unsigned __int128)(bv);
      const unsigned __int128 udst = (unsigned __int128)(destv);
      switch (op) {
        case ame_ew_op_t::ADD: r = __int128(ua + ub); break;
        case ame_ew_op_t::SUB: r = __int128(ub - ua); break;
        case ame_ew_op_t::MUL: r = __int128(ua * ub); break;
        case ame_ew_op_t::MIN: r = ua < ub ? __int128(ua) : __int128(ub); break;
        case ame_ew_op_t::MAX: r = ua > ub ? __int128(ua) : __int128(ub); break;
        case ame_ew_op_t::ABS: r = __int128(ua); break;
        case ame_ew_op_t::ABSDIFF: r = ua > ub ? __int128(ua - ub) : __int128(ub - ua); break;
        case ame_ew_op_t::HDIFF:
          r = round_div2(__int128(ub - ua), dt.int_rm);
          break;
        case ame_ew_op_t::MEAN:
          r = round_div2(__int128(ua + ub), dt.int_rm);
          break;
        case ame_ew_op_t::MULNEG: r = -__int128(ua * ub); break;
        case ame_ew_op_t::MULACC: r = __int128(udst + ua * ub); break;
        case ame_ew_op_t::MULACCNEG: r = __int128(udst - ua * ub); break;
        case ame_ew_op_t::MULADD: r = __int128(ua + ub * udst); break;
        case ame_ew_op_t::MULSUB: r = __int128(ua - ub * udst); break;
        default: have = false; break;
      }
    } else {
      switch (op) {
        case ame_ew_op_t::ADD: r = av + bv; break;
        case ame_ew_op_t::SUB: r = bv - av; break;
        case ame_ew_op_t::MUL: r = av * bv; break;
        case ame_ew_op_t::MIN: r = av < bv ? av : bv; break;
        case ame_ew_op_t::MAX: r = av > bv ? av : bv; break;
        case ame_ew_op_t::ABS: r = av < 0 ? -av : av; break;
        case ame_ew_op_t::ABSDIFF: {
          const __int128 diff = av - bv;
          r = diff < 0 ? -diff : diff;
          break;
        }
        case ame_ew_op_t::HDIFF:
          r = round_div2(bv - av, dt.int_rm);
          break;
        case ame_ew_op_t::MEAN:
          r = round_div2(av + bv, dt.int_rm);
          break;
        case ame_ew_op_t::MULNEG: r = -(av * bv); break;
        case ame_ew_op_t::MULACC: r = destv + av * bv; break;
        case ame_ew_op_t::MULACCNEG: r = destv - av * bv; break;
        case ame_ew_op_t::MULADD: r = av + bv * destv; break;
        case ame_ew_op_t::MULSUB: r = av - bv * destv; break;
        default: have = false; break;
      }
    }
    if (!have)
      return false;
    const __int128 dv = clamp_wide128(r, dt.size_bits, dt.is_signed, dt.saturate, &sat);
    store_uwide_at(d, i, dt.size_bits, (unsigned __int128)(dv));
  }

  if (xsat && sat && dt.saturate)
    *xsat |= 1;
  return true;
}

namespace {

// Diagnostic FP32 exponent path.  Clamping at +/-512 preserves every FP32
// rounding decision, including addition of a finite FP32 old destination,
// while keeping the scaled operand normal and finite in binary64.
float32_t f32_scale_add_pow2(float32_t a, int64_t exp, float32_t old,
                            bool accumulate) {
  exp = std::max<int64_t>(-512, std::min<int64_t>(512, exp));
  const float64_t scale{uint64_t(exp + 1023) << 52};
  const float64_t av = f32_to_f64(a);
  const float64_t cv = accumulate ? f32_to_f64(old) : float64_t{0};
  const uint_fast8_t rm = softfloat_roundingMode;
  // Jam the wider intermediate so a tiny addend cannot create an erroneous
  // tie when the single architectural rounding to FP32 follows.
  softfloat_roundingMode = softfloat_round_odd;
  const float64_t wide = accumulate ? f64_mulAdd(av, scale, cv)
                                   : f64_mul(av, scale);
  softfloat_roundingMode = rm;
  if (accumulate && wide.v == 0 && rm == softfloat_round_min &&
      ((a.v ^ old.v) & 0x80000000u))
    return float32_t{0x80000000u};
  return f64_to_f32(wide);
}

float32_t f32_scale_pow2(float32_t a, int64_t exp) {
  if (isNaNF32UI(a.v))
    return float32_t{defaultNaNF32UI};
  if (exp == 0)
    return a;
  float hf;
  std::memcpy(&hf, &a.v, 4);
  hf = std::ldexp(hf, exp > 10000 ? 10000 : (exp < -10000 ? -10000 : int(exp)));
  float32_t r;
  std::memcpy(&r.v, &hf, 4);
  if (isNaNF32UI(r.v))
    r.v = defaultNaNF32UI;
  return r;
}

// Host-libm transcendentals and ldexp bypass softfloat flag accrual. Restore
// the spec's OF condition (finite inputs, infinite rounded result) and the
// IEEE directed-rounding overflow result: RDN clamps +overflow to max finite,
// RUP clamps -overflow to min finite; RNE/RTZ/RMM deliver the infinity.
float32_t f32_host_overflow_fixup(float32_t v, bool input_finite) {
  if (!input_finite)
    return v;
  if ((v.v & 0x7fffffffu) != 0x7f800000u || (v.v & 0x007fffffu) != 0)
    return v; // not ±Inf
  softfloat_exceptionFlags |=
      uint8_t(softfloat_flag_overflow | softfloat_flag_inexact);
  if (softfloat_roundingMode == softfloat_round_min ||
      softfloat_roundingMode == softfloat_round_max ||
      softfloat_roundingMode == softfloat_round_minMag) // RTZ/RNO: max finite
    return float32_t{(v.v & 0x80000000u) | 0x7f7fffffu};
  return v;
}

__int128 int_scale_pow2(int64_t a, int64_t exp, bool is_signed, uint32_t size_bits) {
  if (exp == 0)
    return is_signed ? __int128(a) : __int128(uint64_t(a));
  if (exp > 0) {
    const __int128 base = is_signed ? __int128(a) : __int128(uint64_t(a));
    if (exp >= 128)
      return base < 0 ? -(__int128(1) << 126) : (__int128(1) << 126);
    return base << int(exp);
  }
  int64_t sh = -exp;
  if (sh >= int64_t(size_bits)) {
    if (is_signed && a < 0)
      return -1;
    return 0;
  }
  if (is_signed)
    return a >> sh;
  return __int128(uint64_t(a) >> sh);
}

} // namespace

static bool ame_host_ldexp_impl(uint32_t data_dtype, uint32_t exp_dtype, uint32_t nelem,
                    const uint8_t* data, const uint8_t* expn, uint8_t* dest,
                    bool accumulate, bool negate_exp, uint8_t* fflags, uint8_t* xsat,
                    bool exact_fp32) {
  const ame_dtype_t dd = ame_parse_dtype(data_dtype);
  const ame_dtype_t de = ame_parse_dtype(exp_dtype);
  const uint32_t dk = ame_dtype_family_key(data_dtype);
  const uint32_t ek = ame_dtype_family_key(exp_dtype);
  if (dd.size_bits == 0 || de.size_bits == 0 || !is_int_family(ek))
    return false;
  if (!data || !expn || !dest || nelem == 0)
    return false;

  uint8_t local_ff = 0;
  bool sat = false;

  if (is_fp_family(dk) && !is_fp32(dk)) {
    std::vector<uint8_t> fa(size_t(nelem) * 4), fd(size_t(nelem) * 4, 0);
    for (uint32_t i = 0; i < nelem; i++) {
      store_f32(fa.data() + i * 4,
                raw_to_f32(uint64_t(load_int_at(data, i, dd.size_bits, false)), dk));
      if (accumulate)
        store_f32(fd.data() + i * 4,
                  raw_to_f32(uint64_t(load_int_at(dest, i, dd.size_bits, false)), dk));
    }
    // Keep the pre-existing narrow-format path outside this FP32 diagnosis.
    if (!ame_host_ldexp_impl(0x20300120u | (uint32_t(dd.fp_rm & 0xf) << 22) | 0x00300000u, exp_dtype,
                        nelem, fa.data(), expn, fd.data(),
                        accumulate, negate_exp, fflags, xsat, false))
      return false;
    // f32->narrow store-back rounding must accrue to amefflags too.
    if (fflags)
      *fflags |= narrow_store_back(dk, dd, nelem, fd.data(), dest);
    return true;
  }

  if (is_fp32(dk)) {
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dd.fp_rm);
    for (uint32_t i = 0; i < nelem; i++) {
      softfloat_exceptionFlags = 0;
      const float32_t av = load_f32(data + i * 4);
      int64_t exp = load_int_at(expn, i, de.size_bits, de.is_signed);
      if (exact_fp32)
        exp = std::max<int64_t>(-512, std::min<int64_t>(512, exp));
      if (negate_exp)
        exp = -exp;
      float32_t dv;
      if (exact_fp32)
        dv = f32_scale_add_pow2(av, exp, load_f32(dest + i * 4), accumulate);
      else {
        dv = f32_scale_pow2(av, exp);
        dv = f32_host_overflow_fixup(dv, (av.v & 0x7fffffffu) < 0x7f800000u);
        if (accumulate)
          dv = f32_add(load_f32(dest + i * 4), dv);
      }
      if (isNaNF32UI(dv.v))
        dv = float32_t{defaultNaNF32UI};
      dv.v = saturate_inf_raw(dv.v, 0x20300120u, dd.has_inf);
      if (ame_rm_is_rno(dd.fp_rm) &&
          (softfloat_exceptionFlags & softfloat_flag_inexact) &&
          !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
          !isNaNF32UI(dv.v))
        dv.v |= 1;
      local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      store_f32(dest + i * 4, dv);
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (!is_int_family(dk))
    return false;

  for (uint32_t i = 0; i < nelem; i++) {
    const int64_t av = load_int_at(data, i, dd.size_bits, dd.is_signed);
    int64_t exp = load_int_at(expn, i, de.size_bits, de.is_signed);
    if (negate_exp)
      exp = -exp;
    __int128 r = int_scale_pow2(av, exp, dd.is_signed, dd.size_bits);
    if (accumulate) {
      const int64_t destv = load_int_at(dest, i, dd.size_bits, dd.is_signed);
      r += dd.is_signed ? __int128(destv) : __int128(uint64_t(destv));
    }
    const int64_t dv = clamp_wide(r, dd.size_bits, dd.is_signed, dd.saturate, &sat);
    store_int_at(dest, i, dd.size_bits, dv);
  }
  if (xsat && sat && dd.saturate)
    *xsat |= 1;
  return true;
}

bool ame_host_ldexp(uint32_t data_dtype, uint32_t exp_dtype, uint32_t nelem,
                    const uint8_t* data, const uint8_t* expn, uint8_t* dest,
                    bool accumulate, bool negate_exp, uint8_t* fflags, uint8_t* xsat) {
  return ame_host_ldexp_impl(data_dtype, exp_dtype, nelem, data, expn, dest,
                            accumulate, negate_exp, fflags, xsat, true);
}

bool ame_host_ldexp_x(uint32_t data_dtype, uint32_t nelem, const uint8_t* data,
                      int64_t exp, uint8_t* dest, bool accumulate,
                      uint8_t* fflags, uint8_t* xsat) {
  // Broadcast the scalar exponent in the layout ame_host_ldexp reads back:
  // one element per AME_MAX_INT_DTYPE slot (spec 4.4 widest integer). The
  // store/load helpers both operate on the low 64 bits of a 128-bit slot,
  // so the buffer size must follow the descriptor, not stay at Int64.
  const uint32_t ebits = ame_parse_dtype(AME_MAX_INT_DTYPE).size_bits;
  std::vector<uint8_t> expn(ame_packed_bytes(nelem, ebits), 0);
  for (uint32_t i = 0; i < nelem; i++)
    store_int_at(expn.data(), i, ebits, exp);
  return ame_host_ldexp(data_dtype, AME_MAX_INT_DTYPE, nelem, data, expn.data(),
                        dest, accumulate, false, fflags, xsat);
}

bool ame_host_log2bias(uint32_t dtype, uint32_t nelem,
                       const uint8_t* a, const uint8_t* bias, uint8_t* dest,
                       bool bias_minus_log, uint8_t* fflags) {
  const uint32_t key = ame_dtype_family_key(dtype);
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  if (!is_fp_family(key) || !a || !bias || !dest || nelem == 0)
    return false;

  if (!is_fp32(key)) {
    std::vector<uint8_t> fa(size_t(nelem) * 4), fb(size_t(nelem) * 4), fd(size_t(nelem) * 4);
    for (uint32_t i = 0; i < nelem; i++) {
      store_f32(fa.data() + i * 4,
                raw_to_f32(uint64_t(load_int_at(a, i, dt.size_bits, false)), key));
      store_f32(fb.data() + i * 4,
                raw_to_f32(uint64_t(load_int_at(bias, i, dt.size_bits, false)), key));
    }
    if (!ame_host_log2bias(0x20300120u | (uint32_t(dt.fp_rm & 0xf) << 22) | 0x00300000u, nelem,
                           fa.data(), fb.data(), fd.data(),
                           bias_minus_log, fflags))
      return false;
    // f32->narrow store-back rounding must accrue to amefflags too.
    if (fflags)
      *fflags |= narrow_store_back(key, dt, nelem, fd.data(), dest);
    return true;
  }

  uint8_t local_ff = 0;
  softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
  for (uint32_t i = 0; i < nelem; i++) {
    softfloat_exceptionFlags = 0;
    const float32_t input = load_f32(a + i * 4);
    if (softfloat_isSigNaNF32UI(input.v))
      softfloat_exceptionFlags |= softfloat_flag_invalid;
    const float32_t av = f32_canonical_abs(input);
    float32_t lv;
    if (isNaNF32UI(av.v)) {
      lv = float32_t{defaultNaNF32UI};
    } else if ((av.v & 0x7fffffff) == 0) {
      softfloat_exceptionFlags |= softfloat_flag_infinite;
      lv = float32_t{0xff800000u};
    } else {
      float hf;
      std::memcpy(&hf, &av.v, 4);
      hf = std::log2(hf);
      std::memcpy(&lv.v, &hf, 4);
      if (isNaNF32UI(lv.v))
        lv = float32_t{defaultNaNF32UI};
    }
    const float32_t bv = load_f32(bias + i * 4);
    float32_t dv = bias_minus_log ? f32_sub(bv, lv) : f32_sub(lv, bv);
    if (isNaNF32UI(dv.v))
      dv = float32_t{defaultNaNF32UI};
    dv.v = saturate_inf_raw(dv.v, 0x20300120u, dt.has_inf);
    if (ame_rm_is_rno(dt.fp_rm) &&
        (softfloat_exceptionFlags & softfloat_flag_inexact) &&
        !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
        !isNaNF32UI(dv.v))
      dv.v |= 1;
    local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
    store_f32(dest + i * 4, dv);
  }
  if (fflags) *fflags |= local_ff;
  return true;
}

bool ame_host_ew_cmp(bool lt, uint32_t src_dtype, uint32_t dst_dtype, uint32_t nelem,
                     const uint8_t* a, const uint8_t* b, uint8_t* d, uint8_t* fflags) {
  const ame_dtype_t ds = ame_parse_dtype(src_dtype);
  const ame_dtype_t dd = ame_parse_dtype(dst_dtype);
  const uint32_t sk = ame_dtype_family_key(src_dtype);
  const uint32_t dk = ame_dtype_family_key(dst_dtype);
  if (ds.size_bits == 0 || dd.size_bits == 0 || dd.is_fp || !is_int_family(dk))
    return false;

  const unsigned __int128 ones = dd.size_bits >= 128
      ? ~(unsigned __int128)(0)
      : ((unsigned __int128)(1) << dd.size_bits) - 1;
  uint8_t local_ff = 0;

  if (is_fp_family(sk)) {
    for (uint32_t i = 0; i < nelem; i++) {
      softfloat_exceptionFlags = 0;
      const float32_t av = raw_to_f32(uint64_t(load_int_at(a, i, ds.size_bits, false)), sk);
      const float32_t bv = raw_to_f32(uint64_t(load_int_at(b, i, ds.size_bits, false)), sk);
      // Spec 6.2.3 :237-240: NV is set for a signaling NaN, but "Propagating a
      // quiet NaN alone does not set NV"; the ordered comparison is false for
      // any NaN operand.  The _quiet primitives already raise invalid for a
      // signaling NaN input and nothing for a quiet one, so no blanket NV here.
      const bool pass = lt ? f32_lt_quiet(av, bv) : f32_le_quiet(bv, av);
      local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      store_uwide_at(d, i, dd.size_bits, pass ? ones : 0);
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (!is_int_family(sk))
    return false;

  for (uint32_t i = 0; i < nelem; i++) {
    bool pass = false;
    if (!ds.is_signed) {
      const unsigned __int128 ua = load_uwide_at(a, i, ds.size_bits);
      const unsigned __int128 ub = load_uwide_at(b, i, ds.size_bits);
      pass = lt ? (ua < ub) : (ua >= ub);
    } else {
      const __int128 av = load_swide_at(a, i, ds.size_bits);
      const __int128 bv = load_swide_at(b, i, ds.size_bits);
      pass = lt ? (av < bv) : (av >= bv);
    }
    store_uwide_at(d, i, dd.size_bits, pass ? ones : 0);
  }
  return true;
}

bool ame_host_ew_sel(bool lt, bool cmov, uint32_t pred_dtype, uint32_t data_dtype,
                     uint32_t nelem, const uint8_t* pred, const uint8_t* src,
                     uint8_t* dest) {
  const ame_dtype_t dp = ame_parse_dtype(pred_dtype);
  const ame_dtype_t dd = ame_parse_dtype(data_dtype);
  const uint32_t pk = ame_dtype_family_key(pred_dtype);
  if (dp.size_bits == 0 || dd.size_bits == 0)
    return false;

  for (uint32_t i = 0; i < nelem; i++) {
    // Spec cmov/sel: numeric compare of Md[ms1] against zero, not the MSB.
    // The comparison is ordered, matching ame_host_ew_cmp: a NaN element makes
    // both directions false and -0 compares equal to +0.  Unsigned integers are
    // never < 0; FP uses the predicate datatype's order.
    bool neg = false;
    bool nonneg = true;
    if (is_fp_family(pk)) {
      const float32_t av =
          raw_to_f32(uint64_t(load_int_at(pred, i, dp.size_bits, false)), pk);
      const float32_t z = i32_to_f32(0);
      neg = f32_lt_quiet(av, z);
      nonneg = f32_le_quiet(z, av);
    } else if (dp.is_signed) {
      if (dp.size_bits == 128) {
        // load_int_at truncates to 64 bits; the sign of a 128-bit
        // element lives in bit 127.
        neg = (load_uwide_at(pred, i, dp.size_bits) >> 127) != 0;
      } else {
        neg = load_int_at(pred, i, dp.size_bits, true) < 0;
      }
      nonneg = !neg;
    }
    const bool take = lt ? neg : nonneg;
    if (take) {
      ame_copy_elem(dest, i, src, i, dd.size_bits);
    } else if (!cmov) {
      store_int_at(dest, i, dd.size_bits, 0);
    }
  }
  return true;
}

static bool shift_one(ame_shift_op_t op, const ame_dtype_t& dt,
                      const uint8_t* data, uint32_t sh, uint8_t* out) {
  const uint32_t ew = dt.size_bits;
  const uint64_t mask = ew == 64 ? ~uint64_t(0) : ((uint64_t(1) << ew) - 1);
  sh &= (ew - 1);
  if (op == ame_shift_op_t::SRA) {
    if (!dt.is_signed || dt.is_fp)
      return false;
    const int64_t v = load_int_elem(data, ew, true);
    store_int_elem(out, ew, v >> sh);
    return true;
  }
  const uint64_t u = uint64_t(load_int_elem(data, ew, false)) & mask;
  uint64_t r = (op == ame_shift_op_t::SLL) ? ((u << sh) & mask) : (u >> sh);
  store_int_elem(out, ew, int64_t(r));
  return true;
}

bool ame_host_ew_shift(ame_shift_op_t op, uint32_t data_dtype, uint32_t amt_dtype,
                       uint32_t nelem, const uint8_t* data, const uint8_t* amt,
                       uint8_t* d) {
  const ame_dtype_t dd = ame_parse_dtype(data_dtype);
  const ame_dtype_t da = ame_parse_dtype(amt_dtype);
  if (dd.is_fp || da.is_fp || !is_int_family(ame_dtype_family_key(data_dtype)) ||
      !is_int_family(ame_dtype_family_key(amt_dtype)))
    return false;
  const uint32_t ew = dd.size_bits;
  if (ew == 0 || (ew & (ew - 1)) != 0)
    return false;
  if (da.size_bits == 0)
    return false;
  for (uint32_t i = 0; i < nelem; i++) {
    const uint64_t raw = uint64_t(load_int_at(amt, i, da.size_bits, false));
    uint8_t tmp_in[8] = {}, tmp_out[8] = {};
    ame_copy_elem(tmp_in, 0, data, i, ew);
    if (!shift_one(op, dd, tmp_in, uint32_t(raw), tmp_out))
      return false;
    ame_copy_elem(d, i, tmp_out, 0, ew);
  }
  return true;
}

bool ame_host_ew_shift_x(ame_shift_op_t op, uint32_t data_dtype, uint64_t shamt,
                         uint32_t nelem, const uint8_t* data, uint8_t* d) {
  const ame_dtype_t dd = ame_parse_dtype(data_dtype);
  if (dd.is_fp || !is_int_family(ame_dtype_family_key(data_dtype)))
    return false;
  const uint32_t ew = dd.size_bits;
  if (ew == 0 || (ew & (ew - 1)) != 0)
    return false;
  for (uint32_t i = 0; i < nelem; i++) {
    uint8_t tmp_in[8] = {}, tmp_out[8] = {};
    ame_copy_elem(tmp_in, 0, data, i, ew);
    if (!shift_one(op, dd, tmp_in, uint32_t(shamt), tmp_out))
      return false;
    ame_copy_elem(d, i, tmp_out, 0, ew);
  }
  return true;
}

namespace {

enum class ame_fp_kind_t { NONE, F16, BF16, F32, F64, E4M3, E5M2 };

ame_fp_kind_t fp_kind_of(uint32_t key) {
  if (ame_is_fp16(key)) return ame_fp_kind_t::F16;
  if (ame_is_bf16(key)) return ame_fp_kind_t::BF16;
  if (ame_is_fp32(key)) return ame_fp_kind_t::F32;
  if (ame_is_fp64(key)) return ame_fp_kind_t::F64;
  if (ame_is_e4m3(key)) return ame_fp_kind_t::E4M3;
  if (ame_is_e5m2(key)) return ame_fp_kind_t::E5M2;
  return ame_fp_kind_t::NONE;
}

// RNO (0101) has no softfloat mode: it is the jam rounding of the integer ROD
// rule applied to FP — truncate (RTZ) and, when the discarded bits are
// nonzero, force the LSB to 1. Callers apply the sticky fixup by testing
// ame_rm_is_rno together with a per-rounding INEXACT delta.
bool ame_rm_is_rno(uint8_t fp_rm) {
  return (fp_rm & 0xf) == 5;
}

uint8_t ame_fp_rm_to_softfloat(uint8_t fp_rm) {
  switch (fp_rm & 0xf) {
    case 0: return softfloat_round_near_even;
    case 1: return softfloat_round_minMag;
    case 2: return softfloat_round_min;
    case 3: return softfloat_round_max;
    case 4: return softfloat_round_near_maxMag;
    case 5: return softfloat_round_minMag; // RNO: RTZ + sticky LSB (jam)
    default: return softfloat_round_near_even;
  }
}

uint8_t ame_int_rm_to_softfloat(uint8_t int_rm) {
  switch (int_rm & 3) {
    case 0: return softfloat_round_near_maxMag; // RNU ≈ ties away
    case 1: return softfloat_round_near_even;   // RNE
    case 2: return softfloat_round_min;         // RDN
    default: return softfloat_round_minMag;     // ROD ≈ RTZ + jam later
  }
}

float32_t load_as_f32(const uint8_t* p, ame_fp_kind_t k) {
  if (k == ame_fp_kind_t::F32)
    return load_f32(p);
  if (k == ame_fp_kind_t::F16) {
    float16_t h;
    std::memcpy(&h.v, p, 2);
    return f16_to_f32(h);
  }
  if (k == ame_fp_kind_t::BF16) {
    bfloat16_t h;
    std::memcpy(&h.v, p, 2);
    return bf16_to_f32(h);
  }
  if (k == ame_fp_kind_t::F64) {
    float64_t f;
    std::memcpy(&f.v, p, 8);
    return f64_to_f32(f);
  }
  if (k == ame_fp_kind_t::E4M3) {
    e4m3_t e;
    e.v = p[0];
    return f16_to_f32(e4m3_to_f16(e));
  }
  e5m2_t e;
  e.v = p[0];
  return f16_to_f32(e5m2_to_f16(e));
}

void store_from_f32(uint8_t* p, ame_fp_kind_t k, float32_t f, bool dest_inf) {
  if (k == ame_fp_kind_t::F32) {
    store_f32(p, f);
    return;
  }
  if (k == ame_fp_kind_t::F16) {
    const float16_t h = f32_to_f16(f);
    std::memcpy(p, &h.v, 2);
    return;
  }
  if (k == ame_fp_kind_t::BF16) {
    const bfloat16_t h = f32_to_bf16(f);
    std::memcpy(p, &h.v, 2);
    return;
  }
  if (k == ame_fp_kind_t::F64) {
    const float64_t d = f32_to_f64(f);
    std::memcpy(p, &d.v, 8);
    return;
  }
  if (k == ame_fp_kind_t::E4M3) {
    p[0] = f32_to_e4m3(f, true).v;
    return;
  }
  p[0] = f32_to_e5m2(f, !dest_inf).v;
}

} // namespace

bool ame_host_convert_elem(uint32_t src_dtype, const uint8_t* src,
                           uint32_t dst_dtype, uint8_t* dst,
                           uint8_t* fflags, uint8_t* xsat) {
  if (src_dtype == 0 || dst_dtype == 0)
    return false;
  const ame_dtype_t sd = ame_parse_dtype(src_dtype);
  const ame_dtype_t td = ame_parse_dtype(dst_dtype);
  if (sd.size_bits == 0 || td.size_bits == 0 || sd.is_custom || td.is_custom)
    return false;

  const uint32_t sk = ame_dtype_family_key(src_dtype);
  const uint32_t tk = ame_dtype_family_key(dst_dtype);
  const ame_fp_kind_t sfk = fp_kind_of(sk);
  const ame_fp_kind_t tfk = fp_kind_of(tk);
  const bool sint = is_int_family(sk);
  const bool dint = is_int_family(tk);
  if ((!sint && sfk == ame_fp_kind_t::NONE) || (!dint && tfk == ame_fp_kind_t::NONE))
    return false;

  uint8_t local_ff = 0;
  bool sat = false;

  if (sint && dint) {
    const u128 raw = load_uwide_at(src, 0, sd.size_bits);
    const bool neg = sd.is_signed && raw_is_negative(raw, sd.size_bits);
    const u128 mag = neg ? raw_magnitude(raw, sd.size_bits) : raw;
    const u128 c = convert_int_magnitude(neg, u256w{0, mag}, td, sat);
    store_uwide_at(dst, 0, td.size_bits, c);
    if (xsat && sat && td.saturate)
      *xsat |= 1;
    return true;
  }

  if (sfk != ame_fp_kind_t::NONE && tfk != ame_fp_kind_t::NONE) {
    if (sfk == tfk) {
      ame_copy_elem(dst, 0, src, 0, td.size_bits);
      return true;
    }
    softfloat_roundingMode = ame_fp_rm_to_softfloat(td.fp_rm);
    softfloat_exceptionFlags = 0;
    store_from_f32(dst, tfk, load_as_f32(src, sfk), td.has_inf);
    local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (sint && tfk != ame_fp_kind_t::NONE) {
    const int64_t v = load_int_elem(src, sd.size_bits, sd.is_signed);
    softfloat_roundingMode = ame_fp_rm_to_softfloat(td.fp_rm);
    softfloat_exceptionFlags = 0;
    const float32_t f = sd.is_signed ? i64_to_f32(v) : ui64_to_f32(uint64_t(v));
    store_from_f32(dst, tfk, f, td.has_inf);
    local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (sfk != ame_fp_kind_t::NONE && dint) {
    softfloat_roundingMode = ame_int_rm_to_softfloat(td.int_rm);
    softfloat_exceptionFlags = 0;
    const float32_t f = load_as_f32(src, sfk);
    if (td.is_signed) {
      int64_t v = f32_to_i64(f, softfloat_roundingMode, false);
      if ((td.int_rm & 3) == 3 && (softfloat_exceptionFlags & softfloat_flag_inexact))
        v |= 1; // ROD jam
      v = clamp_int(v, td.size_bits, true, td.saturate, &sat);
      store_int_elem(dst, td.size_bits, v);
    } else {
      uint64_t u = f32_to_ui64(f, softfloat_roundingMode, false);
      if ((td.int_rm & 3) == 3 && (softfloat_exceptionFlags & softfloat_flag_inexact))
        u |= 1;
      int64_t v = int64_t(u);
      v = clamp_int(v, td.size_bits, false, td.saturate, &sat);
      store_int_elem(dst, td.size_bits, v);
    }
    local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
    if (fflags) *fflags |= local_ff;
    if (xsat && sat && td.saturate)
      *xsat |= 1;
    return true;
  }

  return false;
}

bool ame_host_convert(uint32_t src_dtype, uint32_t dst_dtype, uint32_t nelem,
                      const uint8_t* src, uint8_t* dst,
                      uint8_t* fflags, uint8_t* xsat) {
  const ame_dtype_t sd = ame_parse_dtype(src_dtype);
  const ame_dtype_t td = ame_parse_dtype(dst_dtype);
  if (sd.size_bits == 0 || td.size_bits == 0 || !src || !dst)
    return false;
  for (uint32_t i = 0; i < nelem; i++) {
    uint8_t sbuf[16] = {}, dbuf[16] = {};
    ame_copy_elem(sbuf, 0, src, i, sd.size_bits);
    if (!ame_host_convert_elem(src_dtype, sbuf, dst_dtype, dbuf, fflags, xsat))
      return false;
    ame_copy_elem(dst, i, dbuf, 0, td.size_bits);
  }
  return true;
}

bool ame_convert_scalar(uint32_t scalar_dtype, uint64_t xval, uint32_t xlen,
                        uint32_t tensor_dtype, uint8_t* out_elem) {
  if (scalar_dtype == 0 || tensor_dtype == 0)
    return false;
  const ame_dtype_t sd = ame_parse_dtype(scalar_dtype);
  if (sd.size_bits == 0)
    return false;

  // Spec 6.2.4: low sizeof(scalar_dtype) bits when that fits in XLEN; otherwise
  // zero-extend the X-register pattern (including Int/Uint128).
  unsigned __int128 bits = xval;
  // RV32 host register storage may contain sign-extension above XLEN.
  if (xlen < 64)
    bits &= ((unsigned __int128)(1) << xlen) - 1;
  if (sd.size_bits < xlen && sd.size_bits < 128)
    bits &= ((unsigned __int128)(1) << sd.size_bits) - 1;

  uint8_t tmp[16] = {};
  store_uwide_at(tmp, 0, sd.size_bits, bits);
  return ame_host_convert_elem(scalar_dtype, tmp, tensor_dtype, out_elem, nullptr, nullptr);
}

namespace {

const uint8_t* elem_ptr(const uint8_t* base, uint32_t n, uint32_t eb,
                        uint32_t i, uint32_t j) {
  return base + (i * n + j) * eb;
}

uint8_t* elem_ptr(uint8_t* base, uint32_t n, uint32_t eb,
                  uint32_t i, uint32_t j) {
  return base + (i * n + j) * eb;
}

// A fold feeds the destination-domain result of each step into the next one.
// Keep raw patterns for comparisons and sign/magnitude for an exact addition.
bool fold_step_int(ame_fold_op_t op, const ame_dtype_t& dt,
                   u128& acc, u128 v, bool& sat) {
  const u128 bias = dt.is_signed ? u128(1) << (dt.size_bits - 1) : 0;
  switch (op) {
    case ame_fold_op_t::ADD: {
      const bool aneg = dt.is_signed && raw_is_negative(acc, dt.size_bits);
      const bool vneg = dt.is_signed && raw_is_negative(v, dt.size_bits);
      const u256w am{0, aneg ? raw_magnitude(acc, dt.size_bits) : acc};
      const u256w vm{0, vneg ? raw_magnitude(v, dt.size_bits) : v};
      bool neg;
      u256w mag;
      add_mag256(aneg, am, vneg, vm, neg, mag);
      acc = convert_int_magnitude(neg, mag, dt, sat);
      return true;
    }
    case ame_fold_op_t::MIN:
      acc = (acc ^ bias) < (v ^ bias) ? acc : v;
      return true;
    case ame_fold_op_t::MAX:
      acc = (acc ^ bias) > (v ^ bias) ? acc : v;
      return true;
    default: return false;
  }
}

bool fold_step_f32(ame_fold_op_t op, float32_t& acc, float32_t v, uint8_t* ff) {
  softfloat_exceptionFlags = 0;
  float32_t next;
  switch (op) {
    case ame_fold_op_t::ADD: next = f32_add(acc, v); break;
    case ame_fold_op_t::MIN: next = f32_ame_min(acc, v); break;
    case ame_fold_op_t::MAX: next = f32_ame_max(acc, v); break;
    default: return false;
  }
  *ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
  acc = next;
  return true;
}

} // namespace

bool ame_host_reduce(ame_fold_op_t op, bool along_row, uint32_t dtype, uint32_t n,
                     const uint8_t* src, uint8_t* dst, uint8_t* fflags, uint8_t* xsat) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const uint32_t key = ame_dtype_family_key(dtype);
  if (dt.size_bits == 0 || n == 0)
    return false;

  uint8_t local_ff = 0;
  bool sat = false;

  if (is_fp_family(key)) {
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
    const bool jam = ame_rm_is_rno(dt.fp_rm);
    for (uint32_t outer = 0; outer < n; outer++) {
      const uint32_t first = along_row ? square_index(n, outer, 0)
                                       : square_index(n, 0, outer);
      float32_t acc = raw_to_f32(uint64_t(load_int_at(src, first, dt.size_bits, false)), key);
      for (uint32_t k = 1; k < n; k++) {
        const uint32_t ix = along_row ? square_index(n, outer, k)
                                      : square_index(n, k, outer);
        const float32_t v =
            raw_to_f32(uint64_t(load_int_at(src, ix, dt.size_bits, false)), key);
        if (!fold_step_f32(op, acc, v, &local_ff))
          return false;
        if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
            !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
            !isNaNF32UI(acc.v))
          acc.v |= 1;
      }
      for (uint32_t inner = 0; inner < n; inner++) {
        const uint32_t ix = along_row ? square_index(n, outer, inner)
                                      : square_index(n, inner, outer);
        softfloat_exceptionFlags = 0;
        uint64_t raw = f32_to_raw(acc, dtype);
        raw = saturate_inf_raw(raw, key, dt.has_inf);
        if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
            !(softfloat_exceptionFlags & softfloat_flag_overflow))
          raw |= 1;
        store_int_at(dst, ix, dt.size_bits, int64_t(raw));
        local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      }
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (!is_int_family(key))
    return false;

  for (uint32_t outer = 0; outer < n; outer++) {
    const uint32_t first = along_row ? square_index(n, outer, 0)
                                     : square_index(n, 0, outer);
    u128 acc = load_uwide_at(src, first, dt.size_bits);
    for (uint32_t k = 1; k < n; k++) {
      const uint32_t ix = along_row ? square_index(n, outer, k)
                                    : square_index(n, k, outer);
      const u128 v = load_uwide_at(src, ix, dt.size_bits);
      if (!fold_step_int(op, dt, acc, v, sat))
        return false;
    }
    for (uint32_t inner = 0; inner < n; inner++) {
      const uint32_t ix = along_row ? square_index(n, outer, inner)
                                    : square_index(n, inner, outer);
      store_uwide_at(dst, ix, dt.size_bits, acc);
    }
  }
  if (xsat && sat && dt.saturate)
    *xsat |= 1;
  return true;
}

bool ame_host_prefix(ame_fold_op_t op, bool along_row, uint32_t dtype, uint32_t n,
                     const uint8_t* src, uint8_t* dst, uint8_t* fflags, uint8_t* xsat) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const uint32_t key = ame_dtype_family_key(dtype);
  if (dt.size_bits == 0 || n == 0)
    return false;

  uint8_t local_ff = 0;
  bool sat = false;

  if (is_fp_family(key)) {
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
    const bool jam = ame_rm_is_rno(dt.fp_rm);
    for (uint32_t outer = 0; outer < n; outer++) {
      const uint32_t first = along_row ? square_index(n, outer, 0)
                                       : square_index(n, 0, outer);
      float32_t acc = raw_to_f32(uint64_t(load_int_at(src, first, dt.size_bits, false)), key);
      for (uint32_t k = 0; k < n; k++) {
        if (k > 0) {
          const uint32_t ix = along_row ? square_index(n, outer, k)
                                        : square_index(n, k, outer);
          const float32_t v =
              raw_to_f32(uint64_t(load_int_at(src, ix, dt.size_bits, false)), key);
          if (!fold_step_f32(op, acc, v, &local_ff))
            return false;
          if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
              !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
              !isNaNF32UI(acc.v))
            acc.v |= 1;
        }
        const uint32_t ix = along_row ? square_index(n, outer, k)
                                      : square_index(n, k, outer);
        softfloat_exceptionFlags = 0;
        uint64_t raw = f32_to_raw(acc, dtype);
        raw = saturate_inf_raw(raw, key, dt.has_inf);
        if (jam && (softfloat_exceptionFlags & softfloat_flag_inexact) &&
            !(softfloat_exceptionFlags & softfloat_flag_overflow))
          raw |= 1;
        store_int_at(dst, ix, dt.size_bits, int64_t(raw));
        local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
      }
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (!is_int_family(key))
    return false;

  for (uint32_t outer = 0; outer < n; outer++) {
    const uint32_t first = along_row ? square_index(n, outer, 0)
                                     : square_index(n, 0, outer);
    u128 acc = load_uwide_at(src, first, dt.size_bits);
    for (uint32_t k = 0; k < n; k++) {
      if (k > 0) {
        const uint32_t ix = along_row ? square_index(n, outer, k)
                                      : square_index(n, k, outer);
        const u128 v = load_uwide_at(src, ix, dt.size_bits);
        if (!fold_step_int(op, dt, acc, v, sat))
          return false;
      }
      const uint32_t ix = along_row ? square_index(n, outer, k)
                                    : square_index(n, k, outer);
      store_uwide_at(dst, ix, dt.size_bits, acc);
    }
  }
  if (xsat && sat && dt.saturate)
    *xsat |= 1;
  return true;
}

bool ame_host_bcast(bool along_row, uint32_t dtype, uint32_t n,
                    const uint8_t* src, uint8_t* dst, uint32_t idx) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  // Spec 6.24/6.106: D[i,j] = A[i,c] (column) or A[c,j] (row). Bit-copy;
  // c is already range-checked by the caller as an illegal-instruction case.
  if (dt.size_bits == 0 || n == 0 || idx >= n)
    return false;
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const uint32_t src_i = along_row ? square_index(n, idx, j)
                                       : square_index(n, i, idx);
      ame_copy_elem(dst, square_index(n, i, j), src, src_i, dt.size_bits);
    }
  }
  return true;
}

bool ame_host_shift(bool along_row, uint32_t dtype, uint32_t n,
                    const uint8_t* src, uint8_t* dst, int32_t offset) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  // Spec 6.27/6.109: D[i,j] = A[i, j+c] or A[i+c, j]; out-of-range fills 0,
  // no wrap, no trap. c is a signed 32-bit control scalar.
  if (dt.size_bits == 0 || n == 0)
    return false;

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t si = along_row ? int32_t(i) + offset : int32_t(i);
      const int32_t sj = along_row ? int32_t(j) : int32_t(j) + offset;
      const uint32_t di = square_index(n, i, j);
      if (si < 0 || sj < 0 || uint32_t(si) >= n || uint32_t(sj) >= n)
        store_uwide_at(dst, di, dt.size_bits, 0);
      else
        ame_copy_elem(dst, di, src,
                      square_index(n, uint32_t(si), uint32_t(sj)),
                      dt.size_bits);
    }
  }
  return true;
}

bool ame_host_gather(bool along_row, uint32_t dtype, uint32_t idx_dtype, uint32_t n,
                     const uint8_t* src, const uint8_t* idx, uint8_t* dst) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const ame_dtype_t idt = ame_parse_dtype(idx_dtype);
  if (dt.size_bits == 0 || idt.size_bits == 0 ||
      !is_int_family(ame_dtype_family_key(idx_dtype)))
    return false;

  uint32_t nbits = 0;
  for (uint32_t t = n; t > 1; t >>= 1)
    nbits++;
  const uint32_t mask = (1u << nbits) - 1;

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const uint32_t raw = uint32_t(load_uwide_at(idx, square_index(n, i, j),
                                                 idt.size_bits));
      const uint32_t g = raw & mask;
      const uint32_t src_i = along_row ? square_index(n, g, j)
                                       : square_index(n, i, g);
      ame_copy_elem(dst, square_index(n, i, j), src, src_i, dt.size_bits);
    }
  }
  return true;
}

bool ame_host_write_ids(bool row_ids, uint32_t dtype, uint32_t n, uint8_t* dst) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const uint32_t key = ame_dtype_family_key(dtype);
  if (dt.size_bits == 0 || !is_int_family(key))
    return false;
  const int64_t max_idx = int64_t(n) - 1;
  bool sat = false;
  const int64_t clamped = clamp_int(max_idx, dt.size_bits, dt.is_signed, false, &sat);
  if (clamped != max_idx)
    return false;
  if (dt.is_signed) {
    if (dt.size_bits < 64) {
      const int64_t hi = (int64_t(1) << (dt.size_bits - 1)) - 1;
      if (max_idx > hi)
        return false;
    }
  } else if (dt.size_bits < 64) {
    const uint64_t hi = (uint64_t(1) << dt.size_bits) - 1;
    if (uint64_t(max_idx) > hi)
      return false;
  }

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int64_t v = row_ids ? int64_t(i) : int64_t(j);
      store_uwide_at(dst, square_index(n, i, j), dt.size_bits,
                     (unsigned __int128)(uint64_t(v)));
    }
  }
  return true;
}

bool ame_host_zip2(bool along_row, bool unzip, uint32_t dtype, uint32_t n,
                   const uint8_t* a, const uint8_t* b,
                   uint8_t* out1, uint8_t* out2) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const uint32_t ew = dt.size_bits;
  if (ew == 0 || n < 2 || (n & 1))
    return false;
  const uint32_t half = n / 2;
  auto cpy = [&](uint8_t* dst, uint32_t di, uint32_t dj,
                 const uint8_t* src, uint32_t si, uint32_t sj) {
    ame_copy_elem(dst, square_index(n, di, dj), src, square_index(n, si, sj), ew);
  };

  if (along_row) {
    for (uint32_t k = 0; k < half; k++) {
      for (uint32_t j = 0; j < n; j++) {
        if (!unzip) {
          cpy(out1, 2 * k, j, a, k, j);
          cpy(out1, 2 * k + 1, j, b, k, j);
          cpy(out2, 2 * k, j, a, half + k, j);
          cpy(out2, 2 * k + 1, j, b, half + k, j);
        } else {
          cpy(out1, k, j, a, 2 * k, j);
          cpy(out1, half + k, j, b, 2 * k, j);
          cpy(out2, k, j, a, 2 * k + 1, j);
          cpy(out2, half + k, j, b, 2 * k + 1, j);
        }
      }
    }
  } else {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < half; j++) {
        if (!unzip) {
          cpy(out1, i, 2 * j, a, i, j);
          cpy(out1, i, 2 * j + 1, b, i, j);
          cpy(out2, i, 2 * j, a, i, half + j);
          cpy(out2, i, 2 * j + 1, b, i, half + j);
        } else {
          cpy(out1, i, j, a, i, 2 * j);
          cpy(out1, i, half + j, b, i, 2 * j);
          cpy(out2, i, j, a, i, 2 * j + 1);
          cpy(out2, i, half + j, b, i, 2 * j + 1);
        }
      }
    }
  }
  return true;
}

bool ame_host_scatter(bool along_row, ame_scatter_op_t op,
                      uint32_t dtype, uint32_t idx_dtype, uint32_t n,
                      const uint8_t* src, const uint8_t* idx, uint8_t* dest,
                      uint8_t* fflags, uint8_t* xsat) {
  const ame_dtype_t dt = ame_parse_dtype(dtype);
  const ame_dtype_t idt = ame_parse_dtype(idx_dtype);
  const uint32_t key = ame_dtype_family_key(dtype);
  if (dt.size_bits == 0 || idt.size_bits == 0 ||
      !is_int_family(ame_dtype_family_key(idx_dtype)))
    return false;

  uint32_t nbits = 0;
  for (uint32_t t = n; t > 1; t >>= 1)
    nbits++;
  const uint32_t mask = (1u << nbits) - 1;

  bool sat = false;
  uint8_t local_ff = 0;
  auto dest_ix = [&](uint32_t i, uint32_t j, uint32_t g) {
    return along_row ? square_index(n, g, j) : square_index(n, i, g);
  };

  if (is_fp_family(key)) {
    softfloat_roundingMode = ame_fp_rm_to_softfloat(dt.fp_rm);
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        const uint32_t raw = uint32_t(uint64_t(load_int_at(idx, square_index(n, i, j),
                                                           idt.size_bits, false)));
        const uint32_t g = raw & mask;
        const uint32_t dix = dest_ix(i, j, g);
        const float32_t av =
            raw_to_f32(uint64_t(load_int_at(src, square_index(n, i, j), dt.size_bits, false)), key);
        const float32_t dv =
            raw_to_f32(uint64_t(load_int_at(dest, dix, dt.size_bits, false)), key);
        softfloat_exceptionFlags = 0;
        float32_t out = (op == ame_scatter_op_t::ADD) ? f32_add(dv, av)
                                                      : f32_ame_max(dv, av);
        if (ame_rm_is_rno(dt.fp_rm) &&
            (softfloat_exceptionFlags & softfloat_flag_inexact) &&
            !(softfloat_exceptionFlags & softfloat_flag_overflow) &&
            !isNaNF32UI(out.v))
          out.v |= 1;
        local_ff |= softfloat_to_ame_fflags(softfloat_exceptionFlags);
        uint64_t sraw = f32_to_raw(out, dtype);
        sraw = saturate_inf_raw(sraw, key, dt.has_inf);
        store_int_at(dest, dix, dt.size_bits, int64_t(sraw));
      }
    }
    if (fflags) *fflags |= local_ff;
    return true;
  }

  if (!is_int_family(key))
    return false;

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const uint32_t raw = uint32_t(load_uwide_at(idx, square_index(n, i, j),
                                                 idt.size_bits));
      const uint32_t g = raw & mask;
      const uint32_t dix = dest_ix(i, j, g);
      const unsigned __int128 ua = load_uwide_at(src, square_index(n, i, j), dt.size_bits);
      const unsigned __int128 ud = load_uwide_at(dest, dix, dt.size_bits);
      unsigned __int128 out = 0;
      if (op == ame_scatter_op_t::MAX) {
        if (!dt.is_signed) {
          out = ua > ud ? ua : ud;
        } else {
          const __int128 sa = load_swide_at(src, square_index(n, i, j), dt.size_bits);
          const __int128 sd = load_swide_at(dest, dix, dt.size_bits);
          out = (unsigned __int128)(sa > sd ? sa : sd);
        }
      } else {
        bool local_sat = false;
        if (!dt.is_signed) {
          out = (unsigned __int128)clamp_wide128(__int128(ua) + __int128(ud),
                                                 dt.size_bits, false, dt.saturate, &local_sat);
        } else {
          const __int128 sa = load_swide_at(src, square_index(n, i, j), dt.size_bits);
          const __int128 sd = load_swide_at(dest, dix, dt.size_bits);
          out = (unsigned __int128)clamp_wide128(sd + sa, dt.size_bits, true, dt.saturate, &local_sat);
        }
        sat = sat || local_sat;
      }
      store_uwide_at(dest, dix, dt.size_bits, out);
    }
  }
  if (xsat && sat && dt.saturate && op == ame_scatter_op_t::ADD)
    *xsat |= 1;
  return true;
}

bool ame_mmove_legal(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits) {
  if (n == 0 || udsz_bits == 0 || (elem_bits != 8 && elem_bits != 16 &&
                                   elem_bits != 32 && elem_bits != 64))
    return false;
  const uint32_t row_bits = n * udsz_bits;
  return row_bits >= elem_bits && (row_bits % elem_bits) == 0;
}

uint32_t ame_mmove_count(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits) {
  if (!ame_mmove_legal(n, udsz_bits, elem_bits))
    return 0;
  const uint32_t row_elements = (n * udsz_bits) / elem_bits;
  return n * row_elements;
}

void ame_host_mmove_store(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits,
                          uint8_t* mreg, uint32_t pos, uint64_t data) {
  const uint32_t count = ame_mmove_count(n, udsz_bits, elem_bits);
  const uint32_t eb = elem_bits / 8;
  pos &= count - 1;
  uint8_t* dst = mreg + size_t(pos) * eb;
  for (uint32_t i = 0; i < eb; i++)
    dst[i] = uint8_t(data >> (8 * i));
}

uint64_t ame_host_mmove_load(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits,
                             const uint8_t* mreg, uint32_t pos) {
  const uint32_t count = ame_mmove_count(n, udsz_bits, elem_bits);
  const uint32_t eb = elem_bits / 8;
  pos &= count - 1;
  const uint8_t* src = mreg + size_t(pos) * eb;
  uint64_t u = 0;
  for (uint32_t i = 0; i < eb; i++)
    u |= uint64_t(src[i]) << (8 * i);
  return u;
}
