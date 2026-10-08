#include "isa_parser.h"
#include "ame_encoding.h"
#include "ame_ops.h"
#include "decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-elementwise-convert: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MCONV_EW & MASK_MCONV_EW) == MATCH_MCONV_EW, "mconv");
  expect((MATCH_MBCAST_M_X & MASK_MBCAST_M_X) == MATCH_MBCAST_M_X, "mbcast");
  expect((MATCH_MPACK_EW_X & MASK_MPACK_EW_X) == MATCH_MPACK_EW_X, "mpack");
  expect((MATCH_MUNPACK_EW_X & MASK_MUNPACK_EW_X) == MATCH_MUNPACK_EW_X, "munpack");

  expect((MATCH_MCONV_EW & MASK_MABS_EW) != MATCH_MABS_EW, "mconv != mabs");
  expect((MATCH_MPACK_EW_X & MASK_MCOLSCATMAX_EW) != MATCH_MCOLSCATMAX_EW,
         "mpack != scatmax");
  expect((MATCH_MUNPACK_EW_X & MASK_MPACK_EW_X) != MATCH_MPACK_EW_X,
         "munpack != mpack");

  insn_t c(insn_bits_t(MATCH_MCONV_EW | (1 << 15) | (2 << 7)));
  expect(c.rd() == 2 && c.rs1() == 1, "mconv regs");

  insn_t b(insn_bits_t(MATCH_MBCAST_M_X | (3 << 20) | (1 << 15) | (2 << 7)));
  expect(b.rd() == 2 && b.rs1() == 1 && b.rs2() == 3, "mbcast regs");

  insn_t p(insn_bits_t(MATCH_MPACK_EW_X | (4 << 20) | (1 << 15) | (0 << 7)));
  expect(p.rd() == 0 && p.rs1() == 1 && p.rs2() == 4, "mpack rs1=X rs2=ms2");
}

static void test_int_widen_narrow() {
  const uint32_t i8 = 0x40000008;
  const uint32_t i32 = 0x40000020;
  const uint32_t i8sat = 0x60000008;
  const uint32_t nelem = 4;
  int8_t a[4] = {1, -2, 127, -128};
  int32_t d[4] = {};
  uint8_t ff = 0, xs = 0;

  expect(ame_host_convert(i8, i32, nelem,
                          reinterpret_cast<uint8_t*>(a),
                          reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "i8->i32");
  expect(d[0] == 1 && d[1] == -2 && d[2] == 127 && d[3] == -128, "i8->i32 vals");

  int8_t back[4] = {};
  expect(ame_host_convert(i32, i8, nelem,
                          reinterpret_cast<uint8_t*>(d),
                          reinterpret_cast<uint8_t*>(back), &ff, &xs),
         "i32->i8 wrap");
  expect(back[0] == 1 && back[1] == -2 && back[2] == 127 && back[3] == -128,
         "i32->i8 vals");

  int32_t wide[1] = {200};
  int8_t sat[1] = {0};
  xs = 0;
  expect(ame_host_convert(i32, i8sat, 1,
                          reinterpret_cast<uint8_t*>(wide),
                          reinterpret_cast<uint8_t*>(sat), &ff, &xs),
         "i32->i8sat");
  expect(sat[0] == 127, "sat clamp");
  expect(xs == 1, "amexsat");
}

static void test_int_fp32() {
  const uint32_t i32 = 0x40000020;
  const uint32_t f32 = 0x20300120;
  int32_t a[2] = {3, -4};
  uint32_t bits[2] = {};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_convert(i32, f32, 2,
                          reinterpret_cast<uint8_t*>(a),
                          reinterpret_cast<uint8_t*>(bits), &ff, &xs),
         "i32->f32");
  float f0, f1;
  std::memcpy(&f0, &bits[0], 4);
  std::memcpy(&f1, &bits[1], 4);
  expect(f0 == 3.0f && f1 == -4.0f, "i32->f32 vals");

  int32_t back[2] = {};
  expect(ame_host_convert(f32, i32, 2,
                          reinterpret_cast<uint8_t*>(bits),
                          reinterpret_cast<uint8_t*>(back), &ff, &xs),
         "f32->i32");
  expect(back[0] == 3 && back[1] == -4, "f32->i32 vals");
}

static void test_scalar_and_pack_slots() {
  const uint32_t i32 = 0x40000020;
  const uint32_t i8 = 0x40000008;
  uint8_t out[4] = {};
  expect(ame_convert_scalar(i32, uint64_t(-5), 64, i8, out), "scalar i32->i8");
  expect(int8_t(out[0]) == -5, "scalar val");

  // Pack slot 1 of 4: convert one square of 4 int32s into int8s at offset 4.
  const uint32_t n = 4;
  std::vector<int32_t> src = {10, 20, 30, 40};
  std::vector<int8_t> packed(n * 4, int8_t(7));
  uint8_t ff = 0, xs = 0;
  expect(ame_host_convert(i32, i8, n,
                          reinterpret_cast<uint8_t*>(src.data()),
                          reinterpret_cast<uint8_t*>(packed.data() + n),
                          &ff, &xs),
         "pack slot convert");
  expect(packed[0] == 7 && packed[n] == 10 && packed[n + 3] == 40, "pack slot");

  std::vector<int32_t> unpacked(n, 0);
  expect(ame_host_convert(i8, i32, n,
                          reinterpret_cast<uint8_t*>(packed.data() + n),
                          reinterpret_cast<uint8_t*>(unpacked.data()),
                          &ff, &xs),
         "unpack slot convert");
  expect(unpacked == src, "unpack inverse");
}

// Spec 6.2.4 / manual instructions.json: when sizeof(src_dtype) is greater
// than XLEN, X[rs1] is zero-extended to that width. On RV32 the host register
// storage may carry sign-extension above XLEN, which must not leak into an
// Int64/Uint64/FP64 (or wider) scalar.
static void test_scalar_xlen_mask() {
  const uint32_t i64 = 0x40000040; // Int64
  const uint32_t f64 = 0x2C300140; // FP64 (exp_width 11)

  uint8_t out[8] = {};

  // RV32 Int64: only the low 32 bits of X[rs1] are defined.
  expect(ame_convert_scalar(i64, uint64_t(0xFFFFFFFFFFFFFFFFull), 32, i64, out),
         "scalar i64->i64 xlen32");
  const uint8_t i64_x32[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
  expect(std::memcmp(out, i64_x32, sizeof(out)) == 0,
         "scalar i64->i64 xlen32 val");

  // RV32 FP64: the register holds 0x0000000080000000 (-0.0), not the
  // sign-extended NaN pattern 0xFFFFFFFF80000000.
  std::memset(out, 0, sizeof(out));
  expect(ame_convert_scalar(f64, uint64_t(0xFFFFFFFF80000000ull), 32, f64, out),
         "scalar f64->f64 xlen32");
  const uint8_t f64_x32[8] = {0, 0, 0, 0x80, 0, 0, 0, 0};
  expect(std::memcmp(out, f64_x32, sizeof(out)) == 0,
         "scalar f64->f64 xlen32 val");

  // RV64 boundary: no XLEN masking, 0xFFFFFFFFFFFFFFFF stays -1.
  std::memset(out, 0, sizeof(out));
  expect(ame_convert_scalar(i64, uint64_t(0xFFFFFFFFFFFFFFFFull), 64, i64, out),
         "scalar i64->i64 xlen64");
  const uint8_t i64_x64[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  expect(std::memcmp(out, i64_x64, sizeof(out)) == 0,
         "scalar i64->i64 xlen64 val");

  // RV64 boundary: FP64 patterns pass through unchanged.
  std::memset(out, 0, sizeof(out));
  expect(ame_convert_scalar(f64, uint64_t(0x3FF0000000000000ull), 64, f64, out),
         "scalar f64->f64 xlen64");
  const uint8_t f64_x64[8] = {0, 0, 0, 0, 0, 0, 0xF0, 0x3F};
  expect(std::memcmp(out, f64_x64, sizeof(out)) == 0,
         "scalar f64->f64 xlen64 val");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_int_widen_narrow();
  test_int_fp32();
  test_scalar_and_pack_slots();
  test_scalar_xlen_mask();
  fprintf(stderr, "ame-elementwise-convert: ok\n");
  return 0;
}
