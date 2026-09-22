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
    fprintf(stderr, "ame-matmul-transpose-accumulate: %s\n", msg);
    std::exit(1);
  }
}

static int32_t load_i32(const uint8_t* p) {
  int32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

static void store_i32(uint8_t* p, int32_t v) {
  std::memcpy(p, &v, 4);
}

static void test_encodings() {
  expect((MATCH_MMULATACC_2D & MASK_MMULATACC_2D) == MATCH_MMULATACC_2D, "mmulatacc");
  expect((MATCH_MMULATACCNEG_2D & MASK_MMULATACCNEG_2D) == MATCH_MMULATACCNEG_2D,
         "mmulataccneg");
  expect((MATCH_MMULBTACC_2D & MASK_MMULBTACC_2D) == MATCH_MMULBTACC_2D, "mmulbtacc");
  expect((MATCH_MMULBTACCNEG_2D & MASK_MMULBTACCNEG_2D) == MATCH_MMULBTACCNEG_2D,
         "mmulbtaccneg");
  expect((MATCH_MMULACCNEG_2D & MASK_MMULACCNEG_2D) == MATCH_MMULACCNEG_2D, "mmulaccneg");
  expect((MATCH_MMULACC_2D & MASK_MMULACCNEG_2D) != MATCH_MMULACCNEG_2D,
         "mmulacc != mmulaccneg");
  expect((MATCH_MMULATACCNEG_2D & MASK_AME_ACQUIRE) != MATCH_AME_ACQUIRE,
         "mmulataccneg != acquire");
  expect((MATCH_MMULBTACCNEG_2D & MASK_MMULBTACC_2D) != MATCH_MMULBTACC_2D,
         "mmulbtaccneg != mmulbtacc");

  insn_t at(insn_bits_t(MATCH_MMULATACC_2D | (2 << 20) | (1 << 15) | (0 << 7)));
  expect(at.rd() == 0 && at.rs1() == 1 && at.rs2() == 2, "mmulatacc regs");
}

static void test_int32_variants() {
  const uint32_t n = 4;
  const uint32_t n_sq = 1;
  const uint32_t dtype = 0x40000020;
  std::vector<uint8_t> a(n * n * 4), b(n * n * 4), c(n * n * 4), old(n * n * 4);

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      store_i32(a.data() + (i * n + j) * 4, int32_t(10 * i + j));
      store_i32(b.data() + (i * n + j) * 4, i == j ? 1 : 0);
      store_i32(old.data() + (i * n + j) * 4, 3);
    }
  }

  // AT × I from zero Acc = A^T. n_sq=1 stacked layout equals the same N×N buffer.
  std::memset(c.data(), 0, c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, false, true, false),
         "mmulatacc from zero");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      const int32_t want = int32_t(10 * j + i); // A^T
      if (v != want) {
        fprintf(stderr, "ame-matmul-transpose-accumulate: AT C[%u,%u]=%d want %d\n", i, j, v, want);
        std::exit(1);
      }
    }
  }

  // A × I^T from zero = A (I^T=I)
  std::memset(c.data(), 0, c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, false, false, true),
         "mmulbtacc I");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      const int32_t want = int32_t(10 * i + j);
      expect(v == want, "mmulbtacc I elem");
    }
  }

  // A × A^T from zero: C[i,j] = sum_k A[i,k] A[j,k]
  std::memset(c.data(), 0, c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, a.data(), dtype, c.data(),
                         true, false, false, true),
         "mmulbtacc A A^T");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      int32_t want = 0;
      for (uint32_t k = 0; k < n; k++)
        want += int32_t(10 * i + k) * int32_t(10 * j + k);
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      if (v != want) {
        fprintf(stderr, "ame-matmul-transpose-accumulate: AAT C[%u,%u]=%d want %d\n", i, j, v, want);
        std::exit(1);
      }
    }
  }

  // -(A × I) from zero = -A
  std::memset(c.data(), 0, c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, true, false, false),
         "mmulaccneg from zero");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      expect(v == -int32_t(10 * i + j), "mmulaccneg-from-zero elem");
    }
  }

  // AT accumulate: C = 3 + A^T
  std::memcpy(c.data(), old.data(), c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, false, true, false),
         "mmulatacc");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      expect(v == 3 + int32_t(10 * j + i), "mmulatacc elem");
    }
  }

  // C = 3 - A
  std::memcpy(c.data(), old.data(), c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, true, false, false),
         "mmulaccneg");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      expect(v == 3 - int32_t(10 * i + j), "mmulaccneg elem");
    }
  }

  // AT accumulate-negate: C = 3 - A^T
  std::memcpy(c.data(), old.data(), c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, true, true, false),
         "mmulataccneg");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      expect(v == 3 - int32_t(10 * j + i), "mmulataccneg elem");
    }
  }

  // BT accumulate-negate: C = 3 - A (I^T = I)
  std::memcpy(c.data(), old.data(), c.size());
  expect(ame_host_matmul(n, n_sq, dtype, a.data(), dtype, b.data(), dtype, c.data(),
                         true, true, false, true),
         "mmulbtaccneg");
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      const int32_t v = load_i32(c.data() + (i * n + j) * 4);
      expect(v == 3 - int32_t(10 * i + j), "mmulbtaccneg elem");
    }
  }
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_int32_variants();
  fprintf(stderr, "ame-matmul-transpose-accumulate: ok\n");
  return 0;
}
