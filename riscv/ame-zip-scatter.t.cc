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
    fprintf(stderr, "ame-zip-scatter: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MCOLZIP_EW & MASK_MCOLZIP_EW) == MATCH_MCOLZIP_EW, "colzip match");
  expect((MATCH_MCOLUNZIP_EW & MASK_MCOLUNZIP_EW) == MATCH_MCOLUNZIP_EW, "colunzip match");
  expect((MATCH_MROWZIP_EW & MASK_MROWZIP_EW) == MATCH_MROWZIP_EW, "rowzip match");
  expect((MATCH_MROWUNZIP_EW & MASK_MROWUNZIP_EW) == MATCH_MROWUNZIP_EW, "rowunzip match");
  expect((MATCH_MCOLSCATADD_EW & MASK_MCOLSCATADD_EW) == MATCH_MCOLSCATADD_EW, "colscatadd");
  expect((MATCH_MROWSCATADD_EW & MASK_MROWSCATADD_EW) == MATCH_MROWSCATADD_EW, "rowscatadd");
  expect((MATCH_MCOLSCATMAX_EW & MASK_MCOLSCATMAX_EW) == MATCH_MCOLSCATMAX_EW, "colscatmax");
  expect((MATCH_MROWSCATMAX_EW & MASK_MROWSCATMAX_EW) == MATCH_MROWSCATMAX_EW, "rowscatmax");

  // Zip encodings fix rd=0.
  expect((MASK_MCOLZIP_EW & 0xf80) == 0xf80, "zip masks rd");
  insn_t z(insn_bits_t(MATCH_MCOLZIP_EW | (1 << 20) | (2 << 15)));
  expect(z.rd() == 0 && z.rs1() == 2 && z.rs2() == 1, "colzip regs");

  insn_t s(insn_bits_t(MATCH_MCOLSCATADD_EW | (3 << 20) | (1 << 15) | (2 << 7)));
  expect(s.rd() == 2 && s.rs1() == 1 && s.rs2() == 3, "scatadd regs");

  expect((MATCH_MCOLZIP_EW & MASK_MCOLUNZIP_EW) != MATCH_MCOLUNZIP_EW, "zip != unzip");
  expect((MATCH_MCOLSCATADD_EW & MASK_MCOLSCATMAX_EW) != MATCH_MCOLSCATMAX_EW,
         "scatadd != scatmax");
}

static void fill_ij(std::vector<int32_t>& m, uint32_t n, int32_t bias) {
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < n; j++)
      m[i * n + j] = bias + int32_t(i * 10 + j);
}

static void test_zip_unzip_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t n = 4;
  std::vector<int32_t> a(n * n), b(n * n), o1(n * n), o2(n * n);
  fill_ij(a, n, 0);
  fill_ij(b, n, 100);

  expect(ame_host_zip2(false, false, dtype, n,
                       reinterpret_cast<uint8_t*>(a.data()),
                       reinterpret_cast<uint8_t*>(b.data()),
                       reinterpret_cast<uint8_t*>(o1.data()),
                       reinterpret_cast<uint8_t*>(o2.data())),
         "col zip");
  // row0: [A00,B00,A01,B01]=[0,100,1,101] and [A02,B02,A03,B03]=[2,102,3,103]
  expect(o1[0] == 0 && o1[1] == 100 && o1[2] == 1 && o1[3] == 101, "col zip o1 row0");
  expect(o2[0] == 2 && o2[1] == 102 && o2[2] == 3 && o2[3] == 103, "col zip o2 row0");

  std::vector<int32_t> r1(n * n), r2(n * n);
  expect(ame_host_zip2(false, true, dtype, n,
                       reinterpret_cast<uint8_t*>(o1.data()),
                       reinterpret_cast<uint8_t*>(o2.data()),
                       reinterpret_cast<uint8_t*>(r1.data()),
                       reinterpret_cast<uint8_t*>(r2.data())),
         "col unzip of zip");
  expect(r1 == a && r2 == b, "col zip/unzip inverse");

  expect(ame_host_zip2(true, false, dtype, n,
                       reinterpret_cast<uint8_t*>(a.data()),
                       reinterpret_cast<uint8_t*>(b.data()),
                       reinterpret_cast<uint8_t*>(o1.data()),
                       reinterpret_cast<uint8_t*>(o2.data())),
         "row zip");
  // even rows from A[k], odd from B[k]: o1 row0=A row0, o1 row1=B row0
  expect(o1[0] == 0 && o1[1] == 1 && o1[4] == 100 && o1[5] == 101, "row zip o1");
  expect(o2[0] == 20 && o2[1] == 21 && o2[4] == 120 && o2[5] == 121, "row zip o2");

  expect(ame_host_zip2(true, true, dtype, n,
                       reinterpret_cast<uint8_t*>(o1.data()),
                       reinterpret_cast<uint8_t*>(o2.data()),
                       reinterpret_cast<uint8_t*>(r1.data()),
                       reinterpret_cast<uint8_t*>(r2.data())),
         "row unzip of zip");
  expect(r1 == a && r2 == b, "row zip/unzip inverse");
}

static void test_zip_unzip_uint32() {
  const uint32_t dtype = 0x00000020;
  const uint32_t n = 4;
  std::vector<uint32_t> a(n * n), b(n * n), o1(n * n), o2(n * n), r1(n * n), r2(n * n);
  for (uint32_t i = 0; i < n * n; i++) {
    a[i] = i + 1;
    b[i] = 100 + i;
  }
  expect(ame_host_zip2(false, false, dtype, n,
                       reinterpret_cast<uint8_t*>(a.data()),
                       reinterpret_cast<uint8_t*>(b.data()),
                       reinterpret_cast<uint8_t*>(o1.data()),
                       reinterpret_cast<uint8_t*>(o2.data())),
         "u32 col zip");
  expect(o1[0] == 1 && o1[1] == 100 && o1[2] == 2 && o1[3] == 101, "u32 col zip o1");
  expect(ame_host_zip2(false, true, dtype, n,
                       reinterpret_cast<uint8_t*>(o1.data()),
                       reinterpret_cast<uint8_t*>(o2.data()),
                       reinterpret_cast<uint8_t*>(r1.data()),
                       reinterpret_cast<uint8_t*>(r2.data())),
         "u32 col unzip");
  expect(r1 == a && r2 == b, "u32 zip inverse");
}

static void test_scatter_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t n = 4;
  std::vector<int32_t> a(n * n), idx(n * n), d(n * n);
  fill_ij(a, n, 0);
  for (uint32_t i = 0; i < n * n; i++)
    idx[i] = 1;
  for (uint32_t i = 0; i < n * n; i++)
    d[i] = 5;

  uint8_t ff = 0, xs = 0;
  expect(ame_host_scatter(false, ame_scatter_op_t::ADD, dtype, dtype, n,
                          reinterpret_cast<uint8_t*>(a.data()),
                          reinterpret_cast<uint8_t*>(idx.data()),
                          reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "col scatadd");
  // D[i,1] += sum of row i; other columns stay 5. row0 sum=0+1+2+3=6 → 11
  expect(d[1] == 11 && d[0] == 5 && d[2] == 5, "col scatadd row0");
  expect(d[4 + 1] == 5 + (10 + 11 + 12 + 13), "col scatadd row1");

  fill_ij(a, n, 0);
  for (uint32_t i = 0; i < n * n; i++)
    d[i] = 0;
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < n; j++)
      idx[i * n + j] = int32_t(j); // identity columns → D[i,j]+=A[i,j]
  expect(ame_host_scatter(false, ame_scatter_op_t::ADD, dtype, dtype, n,
                          reinterpret_cast<uint8_t*>(a.data()),
                          reinterpret_cast<uint8_t*>(idx.data()),
                          reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "col scatadd identity");
  expect(d[0] == 0 && d[1] == 1 && d[4] == 10 && d[15] == 33, "identity scatadd");

  for (uint32_t i = 0; i < n * n; i++) {
    d[i] = 7;
    idx[i] = 2;
  }
  expect(ame_host_scatter(false, ame_scatter_op_t::MAX, dtype, dtype, n,
                          reinterpret_cast<uint8_t*>(a.data()),
                          reinterpret_cast<uint8_t*>(idx.data()),
                          reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "col scatmax");
  // D[0,2] = max(7, 0,1,2,3) = 7; D[1,2] = max(7, 10,11,12,13) = 13
  expect(d[2] == 7 && d[4 + 2] == 13, "col scatmax vals");
  expect(d[0] == 7 && d[1] == 7, "untouched cols");

  for (uint32_t i = 0; i < n * n; i++) {
    d[i] = 1;
    idx[i] = 0;
  }
  expect(ame_host_scatter(true, ame_scatter_op_t::ADD, dtype, dtype, n,
                          reinterpret_cast<uint8_t*>(a.data()),
                          reinterpret_cast<uint8_t*>(idx.data()),
                          reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "row scatadd");
  // D[0,j] += sum of column j. col0: 0+10+20+30=60 → 61
  expect(d[0] == 61 && d[1] == 1 + (1 + 11 + 21 + 31), "row scatadd");
  expect(d[4] == 1, "other rows untouched");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_zip_unzip_int32();
  test_zip_unzip_uint32();
  test_scatter_int32();
  fprintf(stderr, "ame-zip-scatter: ok\n");
  return 0;
}
