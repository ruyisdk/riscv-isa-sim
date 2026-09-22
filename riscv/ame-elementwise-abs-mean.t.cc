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
    fprintf(stderr, "ame-elementwise-abs-mean: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MABS_EW & MASK_MABS_EW) == MATCH_MABS_EW, "mabs");
  expect((MATCH_MABSDIFF_EW & MASK_MABSDIFF_EW) == MATCH_MABSDIFF_EW, "mabsdiff");
  expect((MATCH_MABSDIFF_EW_X & MASK_MABSDIFF_EW_X) == MATCH_MABSDIFF_EW_X, "mabsdiff.x");
  expect((MATCH_MHDIFF_EW & MASK_MHDIFF_EW) == MATCH_MHDIFF_EW, "mhdiff");
  expect((MATCH_MHDIFF_EW_X & MASK_MHDIFF_EW_X) == MATCH_MHDIFF_EW_X, "mhdiff.x");
  expect((MATCH_MMEAN_EW & MASK_MMEAN_EW) == MATCH_MMEAN_EW, "mmean");
  expect((MATCH_MMEAN_EW_X & MASK_MMEAN_EW_X) == MATCH_MMEAN_EW_X, "mmean.x");
  expect((MATCH_MMULACC_EW & MASK_MMULACC_EW) == MATCH_MMULACC_EW, "mmulacc");
  expect((MATCH_MMULACC_EW_X & MASK_MMULACC_EW_X) == MATCH_MMULACC_EW_X, "mmulacc.x");
  expect((MATCH_MMULACCNEG_EW & MASK_MMULACCNEG_EW) == MATCH_MMULACCNEG_EW, "mmulaccneg");
  expect((MATCH_MMULADD_EW & MASK_MMULADD_EW) == MATCH_MMULADD_EW, "mmuladd");
  expect((MATCH_MMULSUB_EW & MASK_MMULSUB_EW) == MATCH_MMULSUB_EW, "mmulsub");
  expect((MATCH_MMULNEG_EW & MASK_MMULNEG_EW) == MATCH_MMULNEG_EW, "mmulneg");

  expect((MATCH_MABS_EW & MASK_MABS_EW) != (MATCH_MSETTYP & MASK_MSETTYP),
         "mabs != msettyp");
  expect((MATCH_MABSDIFF_EW & MASK_MABSDIFF_EW) != (MATCH_MADD_EW & MASK_MADD_EW),
         "mabsdiff != madd");
  expect((MATCH_MMEAN_EW & MASK_MMEAN_EW) != (MATCH_MMEAN_EW_X & MASK_MMEAN_EW_X),
         "mmean != mmean.x");
  expect((MATCH_MMULACC_EW & MASK_MMULACC_EW) != (MATCH_MMUL_EW & MASK_MMUL_EW),
         "mmulacc != mmul");

  insn_t abs(insn_bits_t(MATCH_MABS_EW | (1 << 15) | (2 << 7)));
  expect(abs.rd() == 2 && abs.rs1() == 1, "mabs regs");

  insn_t mean(insn_bits_t(MATCH_MMEAN_EW | (2 << 20) | (1 << 15) | (0 << 7)));
  expect(mean.rd() == 0 && mean.rs1() == 1 && mean.rs2() == 2, "mmean regs");
}

static void test_abs_absdiff() {
  const uint32_t dtype = 0x40000020;
  const uint32_t nelem = 4;
  std::vector<int32_t> a = {1, -2, 3, -4};
  std::vector<int32_t> b = {10, 20, 30, 40};
  std::vector<int32_t> d(nelem);
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ew(ame_ew_op_t::ABS, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()), nullptr,
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "abs");
  expect(d[0] == 1 && d[1] == 2 && d[2] == 3 && d[3] == 4, "abs vals");

  expect(ame_host_ew(ame_ew_op_t::ABSDIFF, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "absdiff");
  expect(d[0] == 9 && d[1] == 22 && d[2] == 27 && d[3] == 44, "absdiff vals");
}

static void test_mean_hdiff_rm() {
  const uint32_t rnu = 0x40000020; // Int32 RNU
  const uint32_t rne = 0x48000020; // Int32 RNE
  const uint32_t rdn = 0x50000020; // Int32 RDN
  const uint32_t rod = 0x58000020; // Int32 ROD
  const uint32_t nelem = 2;
  std::vector<int32_t> a = {0, 1};
  std::vector<int32_t> b = {1, 2};
  std::vector<int32_t> d(nelem);
  uint8_t ff = 0, xs = 0;

  // mean(0,1)=0.5; mean(1,2)=1.5
  expect(ame_host_ew(ame_ew_op_t::MEAN, rnu, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mean rnu");
  expect(d[0] == 1 && d[1] == 2, "mean rnu vals");

  expect(ame_host_ew(ame_ew_op_t::MEAN, rne, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mean rne");
  expect(d[0] == 0 && d[1] == 2, "mean rne vals");

  expect(ame_host_ew(ame_ew_op_t::MEAN, rdn, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mean rdn");
  expect(d[0] == 0 && d[1] == 1, "mean rdn vals");

  expect(ame_host_ew(ame_ew_op_t::MEAN, rod, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mean rod");
  expect(d[0] == 1 && d[1] == 1, "mean rod vals");

  // hdiff = (B-A)/2: (1-0)/2=0.5, (2-1)/2=0.5
  expect(ame_host_ew(ame_ew_op_t::HDIFF, rnu, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "hdiff rnu");
  expect(d[0] == 1 && d[1] == 1, "hdiff rnu vals");
}

static void test_fused() {
  const uint32_t dtype = 0x40000020;
  const uint32_t nelem = 2;
  std::vector<int32_t> a = {2, 3};
  std::vector<int32_t> b = {4, 5};
  std::vector<int32_t> d = {10, 20};
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ew(ame_ew_op_t::MULNEG, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mulneg");
  expect(d[0] == -8 && d[1] == -15, "mulneg vals");

  d = {10, 20};
  expect(ame_host_ew(ame_ew_op_t::MULACC, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mulacc");
  expect(d[0] == 18 && d[1] == 35, "mulacc vals");

  d = {10, 20};
  expect(ame_host_ew(ame_ew_op_t::MULACCNEG, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mulaccneg");
  expect(d[0] == 2 && d[1] == 5, "mulaccneg vals");

  d = {10, 20};
  expect(ame_host_ew(ame_ew_op_t::MULADD, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "muladd");
  expect(d[0] == 42 && d[1] == 103, "muladd vals");

  d = {10, 20};
  expect(ame_host_ew(ame_ew_op_t::MULSUB, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "mulsub");
  expect(d[0] == -38 && d[1] == -97, "mulsub vals");
}

static void test_abs_sat() {
  const uint32_t dtype = 0x60000008; // Int8 saturating
  int8_t a[1] = {-128};
  int8_t d[1] = {0};
  uint8_t ff = 0, xs = 0;
  expect(ame_host_ew(ame_ew_op_t::ABS, dtype, 1,
                     reinterpret_cast<uint8_t*>(a), nullptr,
                     reinterpret_cast<uint8_t*>(d), &ff, &xs),
         "abs sat");
  expect(d[0] == 127, "abs INT_MIN sat");
  expect(xs == 1, "amexsat set");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_abs_absdiff();
  test_mean_hdiff_rm();
  test_fused();
  test_abs_sat();
  fprintf(stderr, "ame-elementwise-abs-mean: ok\n");
  return 0;
}
