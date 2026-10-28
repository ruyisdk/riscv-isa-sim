#include "isa_parser.h"
#include "ame_encoding.h"
#include "ame_ops.h"
#include "decode.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-fp-unary-math: %s\n", msg);
    std::exit(1);
  }
}

static uint32_t fbits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

static float bitsf(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static void test_encodings() {
  expect((MATCH_MSQRT_EW & MASK_MSQRT_EW) == MATCH_MSQRT_EW, "msqrt");
  expect((MATCH_MREC_EW & MASK_MREC_EW) == MATCH_MREC_EW, "mrec");
  expect((MATCH_MEXP2_EW & MASK_MEXP2_EW) == MATCH_MEXP2_EW, "mexp2");
  expect((MATCH_MLOG2_EW & MASK_MLOG2_EW) == MATCH_MLOG2_EW, "mlog2");
  expect((MATCH_MSIN_EW & MASK_MSIN_EW) == MATCH_MSIN_EW, "msin");
  expect((MATCH_MCOS_EW & MASK_MCOS_EW) == MATCH_MCOS_EW, "mcos");
  expect((MATCH_MTANH_EW & MASK_MTANH_EW) == MATCH_MTANH_EW, "mtanh");
  expect((MATCH_MRSQRT_EW & MASK_MRSQRT_EW) == MATCH_MRSQRT_EW, "mrsqrt");
  expect((MATCH_MFRINTZ_EW & MASK_MFRINTZ_EW) == MATCH_MFRINTZ_EW, "mfrintz");
  expect((MATCH_MLDEXP_EW & MASK_MLDEXP_EW) == MATCH_MLDEXP_EW, "mldexp");
  expect((MATCH_MLDEXP_EW_X & MASK_MLDEXP_EW_X) == MATCH_MLDEXP_EW_X, "mldexp.x");
  expect((MATCH_MLOG2SUB_EW & MASK_MLOG2SUB_EW) == MATCH_MLOG2SUB_EW, "mlog2sub");
  expect((MATCH_MSUBLOG2_EW_X & MASK_MSUBLOG2_EW_X) == MATCH_MSUBLOG2_EW_X, "msublog2.x");

  expect((MATCH_MCOS_EW & MASK_MCONV_EW) != MATCH_MCONV_EW, "mcos != mconv");
  expect((MATCH_MTANH_EW & MASK_MLS_1R) != MATCH_MLS_1R, "mtanh != mls.1r");
  expect((MATCH_MLDEXP_EW & MASK_MUNPACK_EW_X) != MATCH_MUNPACK_EW_X, "mldexp != munpack");
  expect((MATCH_MSUBLOG2_EW_X & MASK_MMULACC_2D) != MATCH_MMULACC_2D, "msublog2.x != mmulacc");
  expect((MATCH_MFRINTM_EW & MASK_MABS_EW) != MATCH_MABS_EW, "frintm != mabs");

  insn_t s(insn_bits_t(MATCH_MSQRT_EW | (1 << 15) | (2 << 7)));
  expect(s.rd() == 2 && s.rs1() == 1, "msqrt regs");

  insn_t l(insn_bits_t(MATCH_MLDEXP_EW | (3 << 20) | (1 << 15) | (2 << 7)));
  expect(l.rd() == 2 && l.rs1() == 1 && l.rs2() == 3, "mldexp regs");
}

static void test_fp_unops() {
  const uint32_t f32 = 0x20300120;
  uint32_t a[4] = {fbits(4.0f), fbits(8.0f), fbits(2.0f), fbits(1.0f)};
  uint32_t d[4] = {};
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ew(ame_ew_op_t::SQRT, f32, 1,
                     reinterpret_cast<uint8_t*>(a), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "sqrt");
  expect(bitsf(d[0]) == 2.0f, "sqrt(4)");

  expect(ame_host_ew(ame_ew_op_t::REC, f32, 1,
                     reinterpret_cast<uint8_t*>(a + 2), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "rec");
  expect(bitsf(d[0]) == 0.5f, "1/2");

  uint32_t three = fbits(3.0f);
  expect(ame_host_ew(ame_ew_op_t::EXP2, f32, 1,
                     reinterpret_cast<uint8_t*>(&three), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "exp2");
  expect(bitsf(d[0]) == 8.0f, "exp2(3)");

  expect(ame_host_ew(ame_ew_op_t::LOG2, f32, 1,
                     reinterpret_cast<uint8_t*>(a + 1), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "log2");
  expect(bitsf(d[0]) == 3.0f, "log2(8)");

  uint32_t z = fbits(0.0f);
  expect(ame_host_ew(ame_ew_op_t::SIN, f32, 1,
                     reinterpret_cast<uint8_t*>(&z), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "sin");
  expect(bitsf(d[0]) == 0.0f, "sin(0)");
  expect(ame_host_ew(ame_ew_op_t::COS, f32, 1,
                     reinterpret_cast<uint8_t*>(&z), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "cos");
  expect(bitsf(d[0]) == 1.0f, "cos(0)");
  expect(ame_host_ew(ame_ew_op_t::TANH, f32, 1,
                     reinterpret_cast<uint8_t*>(&z), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "tanh");
  expect(bitsf(d[0]) == 0.0f, "tanh(0)");

  expect(ame_host_ew(ame_ew_op_t::RSQRT, f32, 1,
                     reinterpret_cast<uint8_t*>(&a[0]), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "rsqrt");
  expect(bitsf(d[0]) == 0.5f, "rsqrt(4)");

  uint32_t frac = fbits(1.9f);
  expect(ame_host_ew(ame_ew_op_t::FRINTZ, f32, 1,
                     reinterpret_cast<uint8_t*>(&frac), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "frintz");
  expect(bitsf(d[0]) == 1.0f, "trunc(1.9)");

  uint32_t nfrac = fbits(-1.1f);
  expect(ame_host_ew(ame_ew_op_t::FRINTM, f32, 1,
                     reinterpret_cast<uint8_t*>(&nfrac), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "frintm");
  expect(bitsf(d[0]) == -2.0f, "floor(-1.1)");

  uint32_t pfrac = fbits(1.1f);
  expect(ame_host_ew(ame_ew_op_t::FRINTP, f32, 1,
                     reinterpret_cast<uint8_t*>(&pfrac), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "frintp");
  expect(bitsf(d[0]) == 2.0f, "ceil(1.1)");

  uint32_t tie = fbits(2.5f);
  expect(ame_host_ew(ame_ew_op_t::FRINTN, f32, 1,
                     reinterpret_cast<uint8_t*>(&tie), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "frintn");
  expect(bitsf(d[0]) == 2.0f, "rne(2.5)");
}

static void test_ldexp_and_logbias() {
  const uint32_t f32 = 0x20300120;
  const uint32_t i32 = 0x40000020;
  uint32_t a[1] = {fbits(3.0f)};
  int32_t e[1] = {2};
  uint32_t d[1] = {};
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ldexp(f32, i32, 1,
                        reinterpret_cast<uint8_t*>(a),
                        reinterpret_cast<uint8_t*>(e),
                        reinterpret_cast<uint8_t*>(d), false, false, &ff, &xs),
         "ldexp");
  expect(bitsf(d[0]) == 12.0f, "3*2^2");

  expect(ame_host_ldexp(f32, i32, 1,
                        reinterpret_cast<uint8_t*>(a),
                        reinterpret_cast<uint8_t*>(e),
                        reinterpret_cast<uint8_t*>(d), false, true, &ff, &xs),
         "rdexp");
  expect(bitsf(d[0]) == 0.75f, "3*2^-2");

  uint32_t acc[1] = {fbits(1.0f)};
  expect(ame_host_ldexp(f32, i32, 1,
                        reinterpret_cast<uint8_t*>(a),
                        reinterpret_cast<uint8_t*>(e),
                        reinterpret_cast<uint8_t*>(acc), true, false, &ff, &xs),
         "ldexpacc");
  expect(bitsf(acc[0]) == 13.0f, "1+12");

  uint32_t dx[1] = {};
  expect(ame_host_ldexp_x(f32, 1, reinterpret_cast<uint8_t*>(a), 2,
                          reinterpret_cast<uint8_t*>(dx), false, &ff, &xs),
         "ldexp.x");
  expect(bitsf(dx[0]) == 12.0f, "ldexp.x 3*4");

  int32_t ia[1] = {5};
  int32_t ie[1] = {2};
  int32_t id[1] = {};
  expect(ame_host_ldexp(i32, i32, 1,
                        reinterpret_cast<uint8_t*>(ia),
                        reinterpret_cast<uint8_t*>(ie),
                        reinterpret_cast<uint8_t*>(id), false, false, &ff, &xs),
         "int ldexp");
  expect(id[0] == 20, "5<<2");

  uint32_t eight = fbits(8.0f);
  uint32_t one = fbits(1.0f);
  uint32_t out = 0;
  expect(ame_host_log2bias(f32, 1,
                           reinterpret_cast<uint8_t*>(&eight),
                           reinterpret_cast<uint8_t*>(&one),
                           reinterpret_cast<uint8_t*>(&out), false, &ff),
         "log2sub");
  expect(bitsf(out) == 2.0f, "log2(8)-1");

  expect(ame_host_log2bias(f32, 1,
                           reinterpret_cast<uint8_t*>(&eight),
                           reinterpret_cast<uint8_t*>(&one),
                           reinterpret_cast<uint8_t*>(&out), true, &ff),
         "sublog2");
  expect(bitsf(out) == -2.0f, "1-log2(8)");
}

static void test_e5m2_signaling_nan_flags() {
  const uint32_t e5m2 = 0x14300108;
  uint8_t a[1] = {0x7d};
  uint8_t b[1] = {0x3c};
  uint8_t d[1] = {0x3c};
  uint8_t ff = 0, xs = 0;

  // Default (QEMU-aligned): OCP OFP8 Rev 1.0 §5.1 does not interpret the three
  // E5M2 NaN mantissa encodings, so the model treats every one of them as quiet
  // and raises no NV.  The result is the canonical quiet NaN either way.
  expect(!ame_e5m2_nan_ieee(), "e5m2 NaN interpretation defaults to QEMU-aligned");
  expect(ame_host_ew(ame_ew_op_t::MULADD, e5m2, 1, a, b, d, &ff, &xs),
         "e5m2 madd signaling NaN");
  // Spec 6.2.3 :222-224: a quieted NaN input produces the destination
  // datatype's canonical quiet NaN.  E5M2 canonical qNaN is S.11111.10 = 0x7E
  // (0x7F is the non-canonical payload S.11111.11).
  expect(d[0] == 0x7e, "e5m2 signaling NaN result");
  expect(ff == 0x00, "e5m2 quiet NaN encoding raises no NV");

  // IEEE reading: fraction 01 is the signaling encoding (quiet bit clear), so
  // NV is raised and the delivered result is still 0x7E.
  ame_set_e5m2_nan_ieee(true);
  ff = 0;
  d[0] = 0x3c;
  expect(ame_host_ew(ame_ew_op_t::MULADD, e5m2, 1, a, b, d, &ff, &xs),
         "e5m2 madd signaling NaN (IEEE reading)");
  expect(d[0] == 0x7e, "e5m2 signaling NaN result (IEEE reading)");
  expect(ff == 0x10, "e5m2 signaling NaN invalid flag");

  // The knob is E5M2-only: an E4M3 NaN still carries no NV under it.
  const uint32_t e4m3 = 0x10100108;
  const uint8_t e4_nan[1] = {0x7f};
  ff = 0;
  d[0] = 0x3c;
  expect(ame_host_ew(ame_ew_op_t::MULADD, e4m3, 1, e4_nan, b, d, &ff, &xs),
         "e4m3 madd NaN under the IEEE E5M2 reading");
  expect(d[0] == 0x7f, "e4m3 NaN result");
  expect(ff == 0x00, "e4m3 NaN raises no NV");

  ame_set_e5m2_nan_ieee(false);
}

// Spec 6.2.3 :237-238 raises NV for a signaling NaN input and :222-224
// quiet it, so a signaling boxed NaN must come back as the canonical quiet NaN
// with NV set.  The FP32 branch always did this; the FP64 branch of
// cos/sin/tanh/exp2/log2 dropped the flag until it was made to match.
static void test_fp64_signaling_nan_sets_nv() {
  const uint32_t fp64 = 0x2c300140;
  const uint32_t fp32 = 0x20300120;
  const uint8_t snan64[8] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f};
  const uint8_t qnan64[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};
  const uint8_t snan32[4] = {0x01, 0x00, 0x80, 0x7f};
  const uint8_t qnan32[4] = {0x00, 0x00, 0xc0, 0x7f};
  const ame_ew_op_t ops[5] = {ame_ew_op_t::COS, ame_ew_op_t::SIN,
                              ame_ew_op_t::TANH, ame_ew_op_t::EXP2,
                              ame_ew_op_t::LOG2};
  const char* names[5] = {"mcos", "msin", "mtanh", "mexp2", "mlog2"};
  for (int op = 0; op < 5; op++) {
    uint8_t d[8] = {};
    uint8_t ff = 0, xs = 0;
    expect(ame_host_ew(ops[op], fp64, 1, snan64, nullptr, d, &ff, &xs),
           "fp64 signaling NaN runs");
    if (std::memcmp(d, qnan64, 8) != 0 || ff != 0x10)
      fprintf(stderr, "ame-fp-unary-math: %s fp64 sNaN -> %016llx flags %02x\n",
              names[op], (unsigned long long)*(const uint64_t*)d, ff);
    expect(std::memcmp(d, qnan64, 8) == 0, "fp64 signaling NaN result");
    expect(ff == 0x10, "fp64 signaling NaN NV");
    expect(xs == 0, "fp64 signaling NaN no saturation");

    // A quiet NaN propagates without NV.
    std::memset(d, 0, 8);
    ff = 0;
    expect(ame_host_ew(ops[op], fp64, 1, qnan64, nullptr, d, &ff, &xs),
           "fp64 quiet NaN runs");
    expect(std::memcmp(d, qnan64, 8) == 0, "fp64 quiet NaN result");
    expect(ff == 0x00, "fp64 quiet NaN raises no NV");

    // FP32 is the control: it has always set NV here.
    uint8_t d32[4] = {};
    ff = 0;
    expect(ame_host_ew(ops[op], fp32, 1, snan32, nullptr, d32, &ff, &xs),
           "fp32 signaling NaN runs");
    expect(std::memcmp(d32, qnan32, 4) == 0, "fp32 signaling NaN result");
    expect(ff == 0x10, "fp32 signaling NaN NV");
  }
}

// Manual pole case: 1/(±0) and 1/sqrt(±0) saturate to the same-sign max
// finite E4M3 encoding (0x7e / 0xfe) and must accrue DZ|OF|NX = 0xd.
static void test_e4m3_pole_flags() {
  const uint32_t e4m3 = 0x10100108;
  const uint8_t zero = 0x00;
  const uint8_t nzero = 0x80;
  uint8_t d[1] = {};
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ew(ame_ew_op_t::REC, e4m3, 1, &zero, nullptr, d, &ff, &xs),
         "e4m3 rec(+0)");
  expect(d[0] == 0x7e, "e4m3 rec(+0) result");
  expect(ff == 0xd, "e4m3 rec(+0) flags DZ|OF|NX");

  ff = 0;
  expect(ame_host_ew(ame_ew_op_t::REC, e4m3, 1, &nzero, nullptr, d, &ff, &xs),
         "e4m3 rec(-0)");
  expect(d[0] == 0xfe, "e4m3 rec(-0) result");
  expect(ff == 0xd, "e4m3 rec(-0) flags DZ|OF|NX");

  ff = 0;
  expect(ame_host_ew(ame_ew_op_t::RSQRT, e4m3, 1, &zero, nullptr, d, &ff, &xs),
         "e4m3 rsqrt(+0)");
  expect(d[0] == 0x7e, "e4m3 rsqrt(+0) result");
  expect(ff == 0xd, "e4m3 rsqrt(+0) flags DZ|OF|NX");

  ff = 0;
  expect(ame_host_ew(ame_ew_op_t::RSQRT, e4m3, 1, &nzero, nullptr, d, &ff, &xs),
         "e4m3 rsqrt(-0)");
  expect(d[0] == 0xfe, "e4m3 rsqrt(-0) result");
  expect(ff == 0xd, "e4m3 rsqrt(-0) flags DZ|OF|NX");
}

// A mixed-format rounding case: the destination field must not reach the
// narrowed result.  2049.0 is integral but not representable in FP16, so the
// representational rounding of the stored element is the only observable
// difference between reading the field and reading the opcode.
static void test_mixed_rounding_field_is_ignored() {
  const uint32_t fp64 = 0x2c300140;
  const uint32_t fp16 = 0x14300110;
  const uint32_t rno = 0x01400000;
  const uint64_t values[] = {0x40a0020000000000ull, 0x40a0060000000000ull};
  const uint16_t by_direction[4][2] = {
      {0x6800, 0x6801}, // FRINTM: 2048, 2050
      {0x6800, 0x6802}, // FRINTN: 2048, 2052
      {0x6801, 0x6802}, // FRINTP: 2050, 2052
      {0x6800, 0x6801}, // FRINTZ: 2048, 2050
  };
  const ame_ew_op_t ops[4] = {ame_ew_op_t::FRINTM, ame_ew_op_t::FRINTN,
                              ame_ew_op_t::FRINTP, ame_ew_op_t::FRINTZ};
  const char* names[4] = {"mfrintm", "mfrintn", "mfrintp", "mfrintz"};
  for (int op = 0; op < 4; op++) {
    for (uint32_t field = 0; field < 2; field++) {
      for (int i = 0; i < 2; i++) {
        uint8_t a[8] = {0}, d[2] = {0};
        uint8_t ff = 0, xs = 0;
        std::memcpy(a, &values[i], 8);
        const uint32_t dst = fp16 | (field ? rno : 0);
        expect(ame_host_ew_mixed(ops[op], dst, fp64, 1, a, d, &ff, &xs),
               "mixed mfrint runs");
        if (d[0] != uint8_t(by_direction[op][i] & 0xff) ||
            d[1] != uint8_t(by_direction[op][i] >> 8))
          fprintf(stderr,
                  "ame-fp-unary-math: %s field=%s value=%llu -> %02x%02x "
                  "expected %04x\n",
                  names[op], field ? "RNO" : "RNE",
                  (unsigned long long)values[i], d[1], d[0], by_direction[op][i]);
        expect(d[0] == uint8_t(by_direction[op][i] & 0xff) &&
                   d[1] == uint8_t(by_direction[op][i] >> 8),
               "mixed mfrint ignores the destination rounding field");
        expect(ff == 0x01, "mixed mfrint NX");
        expect(xs == 0, "mixed mfrint no saturation");
      }
    }
  }
  // An operation that does read the field still narrows in the field's
  // direction.  sqrt(3) is 1.7320508..., whose FP16 neighbours are 0x3eed and
  // 0x3eee: RNE stores the upper one, the RNO emulation (RTZ plus jam) the
  // lower one, so the control proves the field still reaches this path.
  uint8_t a[8] = {0}, d[2] = {0};
  uint8_t ff = 0, xs = 0;
  const uint64_t three = 0x4008000000000000ull;
  std::memcpy(a, &three, 8);
  expect(ame_host_ew_mixed(ame_ew_op_t::SQRT, fp16, fp64, 1, a, d, &ff, &xs),
         "mixed sqrt runs");
  expect(d[0] == 0xee && d[1] == 0x3e, "mixed sqrt RNE narrows to nearest");
  expect(ff == 0x01, "mixed sqrt NX");
  ff = 0;
  d[0] = 0;
  d[1] = 0;
  expect(ame_host_ew_mixed(ame_ew_op_t::SQRT, fp16 | rno, fp64, 1, a, d, &ff, &xs),
         "mixed sqrt runs under RNO");
  expect(d[0] == 0xed && d[1] == 0x3e, "mixed sqrt RNO narrows toward zero");
  expect(ff == 0x01, "mixed sqrt NX under RNO");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_fp_unops();
  test_ldexp_and_logbias();
  test_e5m2_signaling_nan_flags();
  test_fp64_signaling_nan_sets_nv();
  test_e4m3_pole_flags();
  test_mixed_rounding_field_is_ignored();
  fprintf(stderr, "ame-fp-unary-math: ok\n");
  return 0;
}
