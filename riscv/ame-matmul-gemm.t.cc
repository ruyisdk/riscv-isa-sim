#include "isa_parser.h"
#include "ame_encoding.h"
#include "ame_dtypes.h"
#include "ame_ops.h"
#include "decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-matmul-gemm: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MLS_1R & MASK_MLS_1R) == MATCH_MLS_1R, "mls.1r match");
  expect((MATCH_MLS_RM & MASK_MLS_RM) == MATCH_MLS_RM, "mls.rm match");
  expect((MATCH_MSS_1R & MASK_MSS_1R) == MATCH_MSS_1R, "mss.1r match");
  expect((MATCH_MSS_RM & MASK_MSS_RM) == MATCH_MSS_RM, "mss.rm match");
  expect((MATCH_MMULACC_2D & MASK_MMULACC_2D) == MATCH_MMULACC_2D, "mmulacc.2d match");
  expect((MATCH_MMOV_M_A & MASK_MMOV_M_A) == MATCH_MMOV_M_A, "mmov.m.a match");
  expect((MATCH_MZERO_2D_ACC & MASK_MZERO_2D_ACC) == MATCH_MZERO_2D_ACC, "mzero.acc match");
  expect((MATCH_MZERO_2D_M & MASK_MZERO_2D_M) == MATCH_MZERO_2D_M, "mzero.m match");

  expect((MATCH_MLS_1R & MASK_MLS_1R) != (MATCH_MSETTYP & MASK_MSETTYP), "mls.1r != msettyp");
  expect((MATCH_MMULACC_2D & MASK_MMULACC_2D) != (MATCH_MMULACCNEG_2D & MASK_MMULACCNEG_2D),
         "mmulacc != mmulaccneg");
}

static void test_int32_gemm() {
  const uint32_t n = 8;
  const uint32_t n_sq = 1;
  const uint32_t dtype = 0x40000020; // Int32
  std::vector<uint8_t> a(n * n * 4, 0), b(n * n * 4, 0), c(n * n * 4, 0);

  // A = I, B = 2 on diagonal -> C = 2I
  for (uint32_t i = 0; i < n; i++) {
    int32_t one = 1, two = 2;
    std::memcpy(a.data() + (i * n + i) * 4, &one, 4);
    std::memcpy(b.data() + (i * n + i) * 4, &two, 4);
  }

  expect(ame_matmul_supported(dtype, dtype, dtype, 32), "int32 matmul supported");
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(), false),
         "int32 matmul ran");

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      int32_t v = 0;
      std::memcpy(&v, c.data() + (i * n + j) * 4, 4);
      const int32_t expect_v = (i == j) ? 2 : 0;
      if (v != expect_v) {
        fprintf(stderr, "ame-matmul-gemm: C[%u,%u]=%d want %d\n", i, j, v, expect_v);
        std::exit(1);
      }
    }
  }
}

static void test_int8_widen_gemm() {
  // pack_factor=4, N=8 => K = 32. A is 8x32 int8, B is 32x8 int8, C is 8x8 int32.
  const uint32_t n = 8;
  const uint32_t n_sq = 4;
  const uint32_t a_dt = 0x40000008; // Int8
  const uint32_t c_dt = 0x40000020; // Int32
  std::vector<uint8_t> a(n * n_sq * n, 0), b(n_sq * n * n, 0);
  std::vector<uint8_t> c(n * n * 4, 0);

  // A all 1s, B all 1s => C[i,j] = K = 32
  std::memset(a.data(), 1, a.size());
  std::memset(b.data(), 1, b.size());

  expect(ame_matmul_supported(a_dt, a_dt, c_dt, 32), "int8->int32 supported");
  expect(ame_host_matmul(n, n_sq, a_dt, a.data(), a_dt, b.data(), c_dt, c.data(), false),
         "int8 matmul ran");

  for (uint32_t i = 0; i < n * n; i++) {
    int32_t v = 0;
    std::memcpy(&v, c.data() + i * 4, 4);
    expect(v == 32, "int8 widen C element == 32");
  }

  // accumulate again => 64
  expect(ame_host_matmul(n, n_sq, a_dt, a.data(), a_dt, b.data(), c_dt, c.data(), true),
         "int8 matmulacc ran");
  for (uint32_t i = 0; i < n * n; i++) {
    int32_t v = 0;
    std::memcpy(&v, c.data() + i * 4, 4);
    expect(v == 64, "int8 matmulacc C element == 64");
  }
}

static void test_uint4_widen_gemm() {
  // pack_factor=8, N=4 => K=32. A is 4x32 uint4, B is 32x4 uint4, C is 4x4 uint32.
  const uint32_t n = 4;
  const uint32_t n_sq = 8;
  const uint32_t a_dt = 0x00000004; // Uint4
  const uint32_t c_dt = 0x00000020; // Uint32
  std::vector<uint8_t> a(ame_packed_bytes(n * n_sq * n, 4), 0);
  std::vector<uint8_t> b(ame_packed_bytes(n_sq * n * n, 4), 0);
  std::vector<uint8_t> c(n * n * 4, 0);

  uint8_t one[1] = {1};
  for (uint32_t i = 0; i < n * n_sq * n; i++) {
    ame_copy_elem(a.data(), i, one, 0, 4);
    ame_copy_elem(b.data(), i, one, 0, 4);
  }

  expect(ame_matmul_supported(a_dt, a_dt, c_dt, 32), "uint4->uint32 supported");
  expect(ame_host_matmul(n, n_sq, a_dt, a.data(), a_dt, b.data(), c_dt, c.data(), false),
         "uint4 matmul ran");
  for (uint32_t i = 0; i < n * n; i++) {
    uint32_t v = 0;
    std::memcpy(&v, c.data() + i * 4, 4);
    expect(v == 32, "uint4 widen C element == 32");
  }
}

static void test_unsupported_tuple() {
  expect(ame_matmul_supported(0x40000008, 0x40000008, 0x40000008, 32),
         "int8->int8 Acc supported");
  expect(!ame_matmul_supported(0x40000008, 0x40000020, 0x40000020, 32),
         "int8 x int32 mismatch unsupported");
  expect(ame_matmul_supported(0x40000008, 0x40000008, 0x40000080, 32),
         "int8 x int8 -> int128 Acc supported");
}

static void test_mixed_signedness_multiple_square() {
  const uint32_t n = 4;
  const uint32_t n_sq = 2;
  const uint32_t int4 = 0x40000004;
  const uint32_t uint4 = 0x00000004;
  const uint32_t int8 = 0x40000008;
  std::vector<uint8_t> a(ame_packed_bytes(n * n_sq * n, 4), 0);
  std::vector<uint8_t> b(ame_packed_bytes(n * n_sq * n, 4), 0);
  std::vector<uint8_t> c(n * n, 0);
  const uint8_t lhs_neg_one[1] = {0xf};
  const uint8_t lhs_two[1] = {2};
  const uint8_t rhs_two[1] = {2};
  const uint8_t rhs_three[1] = {3};

  for (uint32_t i = 0; i < n * n; i++) {
    ame_copy_elem(a.data(), i, lhs_neg_one, 0, 4);
    ame_copy_elem(b.data(), i, rhs_two, 0, 4);
    ame_copy_elem(a.data(), n * n + i, lhs_two, 0, 4);
    ame_copy_elem(b.data(), n * n + i, rhs_three, 0, 4);
  }

  expect(ame_matmul_supported(int4, uint4, int8, 8),
         "mixed signedness int4 matmul supported");
  expect(ame_host_matmul(n, n_sq, int4, a.data(), uint4, b.data(), int8,
                         c.data(), false),
         "mixed signedness multiple-square matmul ran");
  // Four (-1 * 2) terms plus four (2 * 3) terms = 16 in every cell.
  for (uint8_t value : c)
    expect(value == 16, "mixed signedness multiple-square result == 16");
}

static void test_narrow_acc_saturation() {
  const uint32_t u8_sat = 0x20000008; // Uint8 + SAT
  const uint32_t i8_sat = 0x60000008; // Int8 + SAT
  const uint32_t n = 2;
  const uint32_t n_sq = 1;
  std::vector<uint8_t> ua(n * n, 1), ub(n * n, 255), uc(n * n, 0);
  uint8_t ff = 0, xs = 0;

  // Each output is 1*255 + 1*255 = 510, which clamps to Uint8::MAX.
  expect(ame_host_matmul(n, n_sq, u8_sat, ua.data(), u8_sat, ub.data(),
                         u8_sat, uc.data(), false, false, false, false,
                         &ff, &xs),
         "uint8 narrow Acc saturation ran");
  for (uint8_t value : uc)
    expect(value == 255, "uint8 narrow Acc clamps to max");
  expect(xs == 1, "uint8 narrow Acc sets amexsat");

  std::vector<uint8_t> ia(n * n, 100), ib(n * n, 2), ic(n * n, 0);
  ff = 0;
  xs = 0;
  // Each output is 100*2 + 100*2 = 400, which clamps to Int8::MAX.
  expect(ame_host_matmul(n, n_sq, i8_sat, ia.data(), i8_sat, ib.data(),
                         i8_sat, ic.data(), false, false, false, false,
                         &ff, &xs),
         "int8 positive narrow Acc saturation ran");
  for (uint8_t value : ic)
    expect(value == 0x7f, "int8 positive narrow Acc clamps to max");
  expect(xs == 1, "int8 positive narrow Acc sets amexsat");

  std::fill(ia.begin(), ia.end(), 100);
  std::fill(ib.begin(), ib.end(), 0xfe); // -2 in Int8
  std::fill(ic.begin(), ic.end(), 0);
  xs = 0;
  // Each output is -400, which clamps to Int8::MIN.
  expect(ame_host_matmul(n, n_sq, i8_sat, ia.data(), i8_sat, ib.data(),
                         i8_sat, ic.data(), false, false, false, false,
                         &ff, &xs),
         "int8 negative narrow Acc saturation ran");
  for (uint8_t value : ic)
    expect(value == 0x80, "int8 negative narrow Acc clamps to min");
  expect(xs == 1, "int8 negative narrow Acc sets amexsat");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");

  test_encodings();
  test_int32_gemm();
  test_int8_widen_gemm();
  test_uint4_widen_gemm();
  test_unsupported_tuple();
  test_mixed_signedness_multiple_square();
  test_narrow_acc_saturation();

  fprintf(stderr, "ame-matmul-gemm: ok\n");
  return 0;
}
