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
    fprintf(stderr, "ame-elementwise-arith: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MADD_EW & MASK_MADD_EW) == MATCH_MADD_EW, "madd.ew");
  expect((MATCH_MADD_EW_X & MASK_MADD_EW_X) == MATCH_MADD_EW_X, "madd.ew.x");
  expect((MATCH_MSUB_EW & MASK_MSUB_EW) == MATCH_MSUB_EW, "msub.ew");
  expect((MATCH_MSUB_EW_X & MASK_MSUB_EW_X) == MATCH_MSUB_EW_X, "msub.ew.x");
  expect((MATCH_MMUL_EW & MASK_MMUL_EW) == MATCH_MMUL_EW, "mmul.ew");
  expect((MATCH_MMUL_EW_X & MASK_MMUL_EW_X) == MATCH_MMUL_EW_X, "mmul.ew.x");
  expect((MATCH_MMIN_EW & MASK_MMIN_EW) == MATCH_MMIN_EW, "mmin.ew");
  expect((MATCH_MMIN_EW_X & MASK_MMIN_EW_X) == MATCH_MMIN_EW_X, "mmin.ew.x");
  expect((MATCH_MMAX_EW & MASK_MMAX_EW) == MATCH_MMAX_EW, "mmax.ew");
  expect((MATCH_MMAX_EW_X & MASK_MMAX_EW_X) == MATCH_MMAX_EW_X, "mmax.ew.x");

  expect((MATCH_MADD_EW & MASK_MADD_EW) != (MATCH_MADD_EW_X & MASK_MADD_EW_X),
         "madd.ew != madd.ew.x");
  expect((MATCH_MSUB_EW & MASK_MSUB_EW) != (MATCH_MMUL_EW & MASK_MMUL_EW),
         "msub.ew != mmul.ew");

  // madd.ew md=m0, ms1=m1, ms2=m2
  insn_t add(insn_bits_t(MATCH_MADD_EW | (2 << 20) | (1 << 15) | (0 << 7)));
  expect((add.bits() & MASK_MADD_EW) == MATCH_MADD_EW, "madd.ew fields");
  expect(add.rd() == 0 && add.rs1() == 1 && add.rs2() == 2, "madd.ew regs");

  // madd.ew.x md=m3, xs1=a1, ms2=m4
  insn_t addx(insn_bits_t(MATCH_MADD_EW_X | (4 << 20) | (11 << 15) | (3 << 7)));
  expect((addx.bits() & MASK_MADD_EW_X) == MATCH_MADD_EW_X, "madd.ew.x fields");
  expect(addx.rd() == 3 && addx.rs1() == 11 && addx.rs2() == 4, "madd.ew.x regs");
}

static void test_int32_ew() {
  const uint32_t dtype = 0x40000020; // Int32
  const uint32_t nelem = 4;
  std::vector<int32_t> a = {1, -2, 3, -4};
  std::vector<int32_t> b = {10, 20, 30, 40};
  std::vector<int32_t> d(nelem);
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ew(ame_ew_op_t::ADD, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "add");
  expect(d[0] == 11 && d[1] == 18 && d[2] == 33 && d[3] == 36, "add vals");

  expect(ame_host_ew(ame_ew_op_t::SUB, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "sub");
  // D = B - A
  expect(d[0] == 9 && d[1] == 22 && d[2] == 27 && d[3] == 44, "sub vals");

  expect(ame_host_ew(ame_ew_op_t::MUL, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mul");
  expect(d[0] == 10 && d[1] == -40 && d[2] == 90 && d[3] == -160, "mul vals");

  expect(ame_host_ew(ame_ew_op_t::MIN, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "min");
  expect(d[0] == 1 && d[1] == -2 && d[2] == 3 && d[3] == -4, "min vals");

  expect(ame_host_ew(ame_ew_op_t::MAX, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "max");
  expect(d[0] == 10 && d[1] == 20 && d[2] == 30 && d[3] == 40, "max vals");
}

static void test_int8_sat() {
  // Int8 saturating: size=8, signed, sat bit set → 0x60000008
  const uint32_t dtype = 0x60000008;
  const uint32_t nelem = 2;
  int8_t a[2] = {100, -100};
  int8_t b[2] = {100, -100};
  int8_t d[2] = {0, 0};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_ew(ame_ew_op_t::ADD, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a),
                     reinterpret_cast<uint8_t*>(b),
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "sat add");
  expect(d[0] == 127 && d[1] == -128, "sat clamp");
  expect(xs == 1, "amexsat set");
}

static void test_uint4_ew() {
  const uint32_t dtype = 0x00000004; // Uint4
  const uint32_t nelem = 4;
  // Packed little-endian nibbles: [1,2] [3,4] and [5,6] [7,8]
  uint8_t a[2] = {uint8_t(1 | (2 << 4)), uint8_t(3 | (4 << 4))};
  uint8_t b[2] = {uint8_t(5 | (6 << 4)), uint8_t(7 | (8 << 4))};
  uint8_t d[2] = {};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_ew(ame_ew_op_t::ADD, dtype, nelem, a, b, d, &ff, &xs), "u4 add");
  expect(d[0] == uint8_t(6 | (8 << 4)) && d[1] == uint8_t(10 | (12 << 4)),
         "u4 add vals");
  expect(ame_host_ew(ame_ew_op_t::MUL, dtype, nelem, a, b, d, &ff, &xs), "u4 mul");
  // 1*5=5, 2*6=12, 3*7=21 -> 5, 4*8=32 -> 0
  expect(d[0] == uint8_t(5 | (12 << 4)) && d[1] == uint8_t(5 | (0 << 4)),
         "u4 mul wrap");
}

static void test_scalar_convert() {
  uint8_t out[4] = {};
  expect(ame_convert_scalar(0x40000020, 7, 64, 0x40000020, out), "i32 scalar");
  int32_t v = 0;
  std::memcpy(&v, out, 4);
  expect(v == 7, "scalar value");
  uint8_t nibble[1] = {0xff};
  expect(ame_convert_scalar(0x00000004, 0xb, 64, 0x00000004, nibble), "u4 scalar");
  expect((nibble[0] & 0xf) == 0xb, "u4 scalar low nibble");
  uint8_t wide[16] = {};
  expect(ame_convert_scalar(0x00000080, 0x11, 64, 0x00000080, wide), "u128 scalar");
  expect(wide[0] == 0x11 && wide[8] == 0, "u128 zero-extends XLEN");
}

static void test_uint64_ew() {
  const uint32_t dtype = 0x00000040; // Uint64
  const uint32_t nelem = 2;
  uint64_t a[2] = {10, 1};
  uint64_t b[2] = {5, 2};
  uint64_t d[2] = {0xaaaaaaaaaaaaaaaaull, 0xaaaaaaaaaaaaaaaaull};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_ew(ame_ew_op_t::ADD, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a),
                     reinterpret_cast<uint8_t*>(b),
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "u64 add");
  expect(d[0] == 15 && d[1] == 3, "u64 add vals");
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(d);
  expect(raw[0] == 15 && raw[4] == 0 && raw[7] == 0, "u64 stored as 8 bytes");
}

static void test_uint64_sat() {
  const uint32_t dtype = 0x20000040; // Uint64 SAT
  const uint32_t nelem = 2;
  uint64_t a[2] = {10, ~uint64_t(0)};
  uint64_t b[2] = {5, 1};
  uint64_t d[2] = {0xaaaaaaaaaaaaaaaaull, 0};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_ew(ame_ew_op_t::ADD, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a),
                     reinterpret_cast<uint8_t*>(b),
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "u64 sat add");
  expect(d[0] == 15 && d[1] == ~uint64_t(0), "u64 sat add vals");
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(d);
  expect(raw[0] == 15 && raw[4] == 0, "u64 sat 15 is 8-byte");
  expect(xs == 1, "u64 sat sets amexsat");
}

static void test_cmp_dest_width() {
  const uint32_t u8 = 0x00000008, i32 = 0x40000020;
  const uint32_t nelem = 2;
  uint8_t a[2] = {1, 5}, b[2] = {2, 3};
  int32_t d[2] = {0, 0};
  uint8_t ff = 0;
  expect(ame_host_ew_cmp(true, u8, i32, nelem, a, b,
                         reinterpret_cast<uint8_t*>(d), &ff),
         "cmp dest i32");
  expect(d[0] == -1 && d[1] == 0, "cmp lt ones in dest width");
}

static void test_uint8_cmovge_unsigned() {
  // UINT8 200 has MSB set; spec compares numerically against 0, so >= 0
  // still selects src (not the old dest).
  const uint32_t u8 = 0x00000008;
  const uint32_t nelem = 2;
  uint8_t pred[2] = {200, 3};
  uint8_t src[2] = {11, 12};
  uint8_t dest[2] = {99, 98};
  expect(ame_host_ew_sel(false, true, u8, u8, nelem, pred, src, dest),
         "u8 cmovge");
  expect(dest[0] == 11 && dest[1] == 12, "u8 cmovge takes src for unsigned");
}

static void test_int8_cmovge() {
  const uint32_t i8 = 0x40000008;
  const uint32_t nelem = 2;
  int8_t pred[2] = {-1, 3};
  uint8_t src[2] = {11, 12};
  uint8_t dest[2] = {99, 98};
  expect(ame_host_ew_sel(false, true, i8, i8, nelem,
                         reinterpret_cast<uint8_t*>(pred), src, dest),
         "i8 cmovge");
  expect(dest[0] == 99 && dest[1] == 12, "i8 cmovge keeps dest when pred < 0");
}

static void test_mixed_madd() {
  // Int8 source converted to Int32 dest, then add (spec 2.3 convert-to-dest).
  const uint32_t i8 = 0x40000008, i32 = 0x40000020;
  const uint32_t nelem = 2;
  int8_t a8[2] = {10, -2};
  int32_t b32[2] = {5, 7};
  int32_t d32[2] = {};
  uint8_t conv[8] = {};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_convert(i8, i32, nelem, reinterpret_cast<uint8_t*>(a8), conv, &ff, &xs),
         "i8 to i32");
  expect(ame_host_ew(ame_ew_op_t::ADD, i32, nelem, conv,
                     reinterpret_cast<uint8_t*>(b32),
                     reinterpret_cast<uint8_t*>(d32), &ff, &xs),
         "mixed add");
  expect(d32[0] == 15 && d32[1] == 5, "mixed add vals");
}

static void test_int128_convert() {
  const uint32_t i64 = 0x40000040, i128 = 0x40000080;
  int64_t src = -3;
  uint8_t dst[16] = {};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_convert(i64, i128, 1, reinterpret_cast<uint8_t*>(&src), dst, &ff, &xs),
         "i64 to i128");
  expect(dst[0] == 0xfd && dst[1] == 0xff && dst[8] == 0xff && dst[15] == 0xff,
         "i64 sign-extends to i128");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_int32_ew();
  test_int8_sat();
  test_uint4_ew();
  test_scalar_convert();
  test_uint64_ew();
  test_uint64_sat();
  test_cmp_dest_width();
  test_uint8_cmovge_unsigned();
  test_int8_cmovge();
  test_mixed_madd();
  test_int128_convert();
  fprintf(stderr, "ame-elementwise-arith: ok\n");
  return 0;
}
