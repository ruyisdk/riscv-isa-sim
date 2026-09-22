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
    fprintf(stderr, "ame-reduce-prefix-permute: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MREDUCEADD_COL & MASK_MREDUCEADD_COL) == MATCH_MREDUCEADD_COL, "reduceadd.col");
  expect((MATCH_MREDUCEADD_ROW & MASK_MREDUCEADD_ROW) == MATCH_MREDUCEADD_ROW, "reduceadd.row");
  expect((MATCH_MPREFIXADD_ROW & MASK_MPREFIXADD_ROW) == MATCH_MPREFIXADD_ROW, "prefixadd.row");
  expect((MATCH_MPREFIXADD_ROW & MASK_AME_ACQUIRE) != MATCH_AME_ACQUIRE,
         "prefixadd.row != acquire");
  expect((MATCH_MCOLID_EW & MASK_MCOLID_EW) == MATCH_MCOLID_EW, "mcolid");
  expect((MATCH_MROWID_EW & MASK_MROWID_EW) == MATCH_MROWID_EW, "mrowid");
  expect((MATCH_MCOLBCAST_EW_X & MASK_MCOLBCAST_EW_X) == MATCH_MCOLBCAST_EW_X, "colbcast");
  expect((MATCH_MCOLGATHER_EW & MASK_MCOLGATHER_EW) == MATCH_MCOLGATHER_EW, "colgather");
  expect((MATCH_MCOLSHIFT_EW_X & MASK_MCOLSHIFT_EW_X) == MATCH_MCOLSHIFT_EW_X, "colshift");

  insn_t red(insn_bits_t(MATCH_MREDUCEADD_COL | (1 << 15) | (2 << 7)));
  expect(red.rd() == 2 && red.rs1() == 1, "reduce regs");

  insn_t bc(insn_bits_t(MATCH_MCOLBCAST_EW_X | (11 << 20) | (1 << 15) | (2 << 7)));
  expect(bc.rd() == 2 && bc.rs1() == 1 && bc.rs2() == 11, "bcast md, rs1, ms2");
}

static void test_reduce_prefix_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t n = 4;
  std::vector<int32_t> a(n * n);
  std::vector<int32_t> d(n * n);
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < n; j++)
      a[i * n + j] = int32_t(i * 10 + j);

  uint8_t ff = 0, xs = 0;
  expect(ame_host_reduce(ame_fold_op_t::ADD, true, dtype, n,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "row reduce add");
  // row0 sum = 0+1+2+3 = 6
  expect(d[0] == 6 && d[1] == 6 && d[2] == 6 && d[3] == 6, "row0 reduce");
  // row1 sum = 10+11+12+13 = 46
  expect(d[4] == 46, "row1 reduce");

  expect(ame_host_reduce(ame_fold_op_t::ADD, false, dtype, n,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "col reduce add");
  // col0 = 0+10+20+30 = 60
  expect(d[0] == 60 && d[4] == 60 && d[8] == 60 && d[12] == 60, "col0 reduce");

  expect(ame_host_prefix(ame_fold_op_t::ADD, true, dtype, n,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "row prefix add");
  expect(d[0] == 0 && d[1] == 1 && d[2] == 3 && d[3] == 6, "row0 prefix");

  expect(ame_host_reduce(ame_fold_op_t::MIN, true, dtype, n,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "row min");
  expect(d[0] == 0 && d[4] == 10, "row min vals");

  expect(ame_host_reduce(ame_fold_op_t::MAX, false, dtype, n,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "col max");
  expect(d[0] == 30 && d[1] == 31, "col max vals");

  // N=4 UINT8 prefix-max along columns (spec 6.95); AME_NELEM=16 is a legal power of 4.
  const uint32_t u8 = 0x00000008;
  uint8_t au[16] = {
    1, 9, 3, 4,
    8, 2, 7, 0,
    5, 6, 1, 9,
    3, 4, 8, 2
  };
  uint8_t du[16] = {};
  expect(ame_host_prefix(ame_fold_op_t::MAX, false, u8, n,
                         au, du, &ff, &xs),
         "uint8 col prefix max");
  expect(du[0] == 1 && du[4] == 8 && du[8] == 8 && du[12] == 8, "uint8 col0 prefixmax");
  expect(du[1] == 9 && du[5] == 9 && du[9] == 9 && du[13] == 9, "uint8 col1 prefixmax");

  expect(ame_host_prefix(ame_fold_op_t::ADD, true, u8, n, au, du, &ff, &xs),
         "uint8 row prefix add");
  expect(du[0] == 1 && du[1] == 10 && du[2] == 13 && du[3] == 17, "uint8 row0 prefixadd");

  const uint32_t u4 = 0x00000004;
  uint8_t au4[8] = {};
  uint8_t du4[8] = {};
  uint8_t raw[16] = {
    1, 9, 3, 4,
    8, 2, 7, 0,
    5, 6, 1, 9,
    3, 4, 8, 2
  };
  for (uint32_t i = 0; i < 16; i++) {
    uint8_t tmp[1] = {raw[i]};
    ame_copy_elem(au4, i, tmp, 0, 4);
  }
  expect(ame_host_prefix(ame_fold_op_t::MAX, false, u4, n,
                         au4, du4, &ff, &xs),
         "uint4 col prefix max");
  uint8_t got[16] = {};
  for (uint32_t i = 0; i < 16; i++) {
    uint8_t tmp[1] = {};
    ame_copy_elem(tmp, 0, du4, i, 4);
    got[i] = tmp[0];
  }
  expect(got[0] == 1 && got[4] == 8 && got[8] == 8 && got[12] == 8, "uint4 col0 prefixmax");
  expect(got[1] == 9 && got[5] == 9 && got[9] == 9 && got[13] == 9, "uint4 col1 prefixmax");

  uint8_t addsrc[16] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };
  uint8_t adddst[16] = {};
  expect(ame_host_prefix(ame_fold_op_t::ADD, true, u8, n,
                         addsrc, adddst, &ff, &xs),
         "uint8 row prefix add");
  expect(adddst[0] == 1 && adddst[1] == 3 && adddst[2] == 6 && adddst[3] == 10,
         "uint8 row0 prefixadd");
  expect(ame_host_reduce(ame_fold_op_t::ADD, true, u8, n,
                         addsrc, adddst, &ff, &xs),
         "uint8 row reduce add");
  expect(adddst[0] == 10 && adddst[3] == 10 && adddst[4] == 26,
         "uint8 row reduce add broadcast");

  const uint32_t n16 = 16;
  std::vector<uint8_t> a16(n16 * n16), d16(n16 * n16);
  for (uint32_t i = 0; i < n16 * n16; i++)
    a16[i] = uint8_t((i * 3) & 0xff);
  expect(ame_host_prefix(ame_fold_op_t::ADD, true, u8, n16,
                         a16.data(), d16.data(), &ff, &xs),
         "uint8 n16 row prefix add");
  uint32_t run = 0;
  for (uint32_t j = 0; j < n16; j++) {
    run = uint32_t(uint8_t(run + a16[j]));
    expect(d16[j] == uint8_t(run), "uint8 n16 row0 prefix");
  }
  expect(ame_host_reduce(ame_fold_op_t::MIN, false, u8, n16,
                         a16.data(), d16.data(), &ff, &xs),
         "uint8 n16 col min");
  uint8_t mn = a16[0];
  for (uint32_t i = 1; i < n16; i++)
    if (a16[i * n16] < mn)
      mn = a16[i * n16];
  expect(d16[0] == mn && d16[n16] == mn, "uint8 n16 col0 min broadcast");
  expect(ame_host_reduce(ame_fold_op_t::MIN, true, u8, n16,
                         a16.data(), d16.data(), &ff, &xs),
         "uint8 n16 row min");
  uint8_t rmn = a16[0];
  for (uint32_t j = 1; j < n16; j++)
    if (a16[j] < rmn)
      rmn = a16[j];
  expect(d16[0] == rmn && d16[n16 - 1] == rmn, "uint8 n16 row0 min broadcast");
}

static void test_permute_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t n = 4;
  std::vector<int32_t> a(n * n), d(n * n), idx(n * n);
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < n; j++)
      a[i * n + j] = int32_t(i * 10 + j);

  expect(ame_host_bcast(false, dtype, n,
                        reinterpret_cast<uint8_t*>(a.data()),
                        reinterpret_cast<uint8_t*>(d.data()), 2),
         "col bcast");
  // column 2 of each row broadcast: row0 -> 2, row1 -> 12
  expect(d[0] == 2 && d[1] == 2 && d[4] == 12 && d[5] == 12, "col bcast vals");

  expect(ame_host_bcast(true, dtype, n,
                        reinterpret_cast<uint8_t*>(a.data()),
                        reinterpret_cast<uint8_t*>(d.data()), 1),
         "row bcast");
  expect(d[0] == 10 && d[1] == 11 && d[4] == 10 && d[5] == 11, "row bcast vals");

  expect(ame_host_shift(false, dtype, n,
                        reinterpret_cast<uint8_t*>(a.data()),
                        reinterpret_cast<uint8_t*>(d.data()), 1),
         "col shift");
  // D[i,j] = A[i,j+1]; D[0,0]=1, D[0,3]=0 fill
  expect(d[0] == 1 && d[1] == 2 && d[2] == 3 && d[3] == 0, "col shift vals");

  for (uint32_t i = 0; i < n * n; i++)
    idx[i] = 1; // gather column/row 1
  expect(ame_host_gather(false, dtype, dtype, n,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(idx.data()),
                         reinterpret_cast<uint8_t*>(d.data())),
         "col gather");
  expect(d[0] == 1 && d[4] == 11, "col gather vals");

  expect(ame_host_write_ids(false, dtype, n, reinterpret_cast<uint8_t*>(d.data())),
         "colid");
  expect(d[0] == 0 && d[1] == 1 && d[2] == 2 && d[3] == 3, "colid row0");
  expect(d[4] == 0 && d[5] == 1, "colid row1");

  expect(ame_host_write_ids(true, dtype, n, reinterpret_cast<uint8_t*>(d.data())),
         "rowid");
  expect(d[0] == 0 && d[4] == 1 && d[8] == 2 && d[12] == 3, "rowid col0");
}

// Spec 6.24 case 0: N=4 UINT8 unit dtype, .rm 01..10, c=2.
// Column 2 is 3,7,11,15 so dest[0]==3, dest[4]==7, dest[15]==15.
static void test_permute_uint8_n4() {
  const uint32_t u8 = 0x00000008;
  const uint32_t n = 4;
  uint8_t a[16] = {
    0x01, 0x02, 0x03, 0x04,
    0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x0c,
    0x0d, 0x0e, 0x0f, 0x10
  };
  uint8_t d[16];

  expect(ame_pack_factor(8, 8) == 1, "UINT8 is unit at U=8");
  expect(ame_pack_factor(8, 16) == 2, "UINT8 is packed at U=16");
  expect(ame_pack_factor(8, 32) == 4, "UINT8 is packed at U=32");

  expect(ame_host_bcast(false, u8, n, a, d, 2), "uint8 col bcast");
  expect(d[0] == 3 && d[1] == 3 && d[2] == 3 && d[3] == 3, "uint8 col bcast row0");
  expect(d[4] == 7 && d[8] == 11 && d[15] == 15, "uint8 col bcast dest[4,8,15]");

  expect(ame_host_bcast(true, u8, n, a, d, 1), "uint8 row bcast");
  expect(d[0] == 5 && d[1] == 6 && d[2] == 7 && d[3] == 8, "uint8 row bcast row0");
  expect(d[4] == 5 && d[12] == 5, "uint8 row bcast later rows");

  expect(ame_host_shift(false, u8, n, a, d, 1), "uint8 col shift");
  expect(d[0] == 2 && d[1] == 3 && d[2] == 4 && d[3] == 0, "uint8 col shift row0");
  expect(d[4] == 6 && d[7] == 0, "uint8 col shift row1");

  expect(ame_host_shift(true, u8, n, a, d, 1), "uint8 row shift");
  expect(d[0] == 5 && d[3] == 8 && d[12] == 0, "uint8 row shift");

  expect(ame_host_shift(false, u8, n, a, d, -1), "uint8 col shift neg");
  expect(d[0] == 0 && d[1] == 1 && d[2] == 2 && d[3] == 3, "uint8 col shift -1");
}

// Packed UINT8 (U=32, P=4): each square is still a 4x4, computed independently.
static void test_permute_uint8_packed() {
  const uint32_t u8 = 0x00000008;
  const uint32_t n = 4, p = 4;
  std::vector<uint8_t> src(size_t(p) * n * n), dst(size_t(p) * n * n);
  for (uint32_t s = 0; s < p; s++)
    for (uint32_t i = 0; i < n * n; i++)
      src[s * n * n + i] = uint8_t(s * 16 + i + 1);
  for (uint32_t s = 0; s < p; s++) {
    expect(ame_host_bcast(false, u8, n, src.data() + s * n * n,
                          dst.data() + s * n * n, 2),
           "packed uint8 col bcast");
  }
  expect(dst[0] == 3 && dst[4] == 7 && dst[15] == 15, "packed sq0 col2");
  expect(dst[16] == 19 && dst[20] == 23 && dst[31] == 31, "packed sq1 col2");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_reduce_prefix_int32();
  test_permute_int32();
  test_permute_uint8_n4();
  test_permute_uint8_packed();
  fprintf(stderr, "ame-reduce-prefix-permute: ok\n");
  return 0;
}
