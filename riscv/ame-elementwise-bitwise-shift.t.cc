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
    fprintf(stderr, "ame-elementwise-bitwise-shift: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MAND_EW & MASK_MAND_EW) == MATCH_MAND_EW, "mand");
  expect((MATCH_MANDNOT_EW & MASK_MANDNOT_EW) == MATCH_MANDNOT_EW, "mandnot");
  expect((MATCH_MOR_EW & MASK_MOR_EW) == MATCH_MOR_EW, "mor");
  expect((MATCH_MORNOT_EW & MASK_MORNOT_EW) == MATCH_MORNOT_EW, "mornot");
  expect((MATCH_MXOR_EW & MASK_MXOR_EW) == MATCH_MXOR_EW, "mxor");
  expect((MATCH_MSLL_EW & MASK_MSLL_EW) == MATCH_MSLL_EW, "msll");
  expect((MATCH_MSRL_EW & MASK_MSRL_EW) == MATCH_MSRL_EW, "msrl");
  expect((MATCH_MSRA_EW & MASK_MSRA_EW) == MATCH_MSRA_EW, "msra");
  expect((MATCH_MCMPGE_EW & MASK_MCMPGE_EW) == MATCH_MCMPGE_EW, "mcmpge");
  expect((MATCH_MCMPLT_EW & MASK_MCMPLT_EW) == MATCH_MCMPLT_EW, "mcmplt");
  expect((MATCH_MSELGE_EW & MASK_MSELGE_EW) == MATCH_MSELGE_EW, "mselge");
  expect((MATCH_MSELLT_EW & MASK_MSELLT_EW) == MATCH_MSELLT_EW, "msellt");
  expect((MATCH_MCMOVGE_EW & MASK_MCMOVGE_EW) == MATCH_MCMOVGE_EW, "mcmovge");
  expect((MATCH_MCMOVLT_EW & MASK_MCMOVLT_EW) == MATCH_MCMOVLT_EW, "mcmovlt");

  expect((MATCH_MAND_EW & MASK_MAND_EW_X) != MATCH_MAND_EW_X, "mand != mand.x");
  expect((MATCH_MADD_EW & MASK_MAND_EW) != MATCH_MAND_EW, "madd != mand");
  expect((MATCH_MSELGE_EW & MASK_MCOLBCAST_EW_X) != MATCH_MCOLBCAST_EW_X,
         "mselge != mcolbcast");

  insn_t a(insn_bits_t(MATCH_MAND_EW | (2 << 20) | (1 << 15) | (0 << 7)));
  expect(a.rd() == 0 && a.rs1() == 1 && a.rs2() == 2, "mand regs");
}

static void test_bitwise_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t nelem = 4;
  std::vector<int32_t> a = {0x0f0f0f0f, -1, 0, 0x12345678};
  std::vector<int32_t> b = {0x00ff00ff, 0x0f0f0f0f, -1, 0x11111111};
  std::vector<int32_t> d(nelem);
  uint8_t ff = 0, xs = 0;

  expect(ame_host_ew(ame_ew_op_t::AND, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "and");
  expect(d[0] == int32_t(0x000f000f) && d[1] == 0x0f0f0f0f && d[2] == 0 &&
             d[3] == 0x10101010,
         "and vals");

  expect(ame_host_ew(ame_ew_op_t::ANDNOT, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "andnot");
  expect(d[0] == int32_t(0x0f000f00) && d[2] == 0, "andnot vals");

  expect(ame_host_ew(ame_ew_op_t::OR, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "or");
  expect(d[0] == int32_t(0x0fff0fff) && d[2] == -1, "or vals");

  expect(ame_host_ew(ame_ew_op_t::XOR, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "xor");
  expect(d[0] == int32_t(0x0ff00ff0), "xor vals");

  expect(ame_host_ew(ame_ew_op_t::ORNOT, dtype, nelem,
                     reinterpret_cast<uint8_t*>(a.data()),
                     reinterpret_cast<uint8_t*>(b.data()),
                     reinterpret_cast<uint8_t*>(d.data()), &ff, &xs),
         "ornot");
  expect(d[2] == 0, "ornot zero | ~(-1)");
}

static void test_cmp_sel_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t nelem = 4;
  std::vector<int32_t> a = {1, -2, 3, -4};
  std::vector<int32_t> b = {1, 20, -5, -4};
  std::vector<int32_t> d(nelem);
  uint8_t ff = 0;

  expect(ame_host_ew_cmp(false, dtype, dtype, nelem,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(b.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff),
         "cmpge");
  expect(d[0] == -1 && d[1] == 0 && d[2] == -1 && d[3] == -1, "cmpge vals");

  expect(ame_host_ew_cmp(true, dtype, dtype, nelem,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(b.data()),
                         reinterpret_cast<uint8_t*>(d.data()), &ff),
         "cmplt");
  expect(d[0] == 0 && d[1] == -1 && d[2] == 0 && d[3] == 0, "cmplt vals");

  std::vector<int32_t> src = {10, 20, 30, 40};
  std::vector<int32_t> dest = {7, 7, 7, 7};
  expect(ame_host_ew_sel(false, false, dtype, dtype, nelem,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(src.data()),
                         reinterpret_cast<uint8_t*>(dest.data())),
         "selge");
  expect(dest[0] == 10 && dest[1] == 0 && dest[2] == 30 && dest[3] == 0, "selge vals");

  dest = {7, 7, 7, 7};
  expect(ame_host_ew_sel(true, true, dtype, dtype, nelem,
                         reinterpret_cast<uint8_t*>(a.data()),
                         reinterpret_cast<uint8_t*>(src.data()),
                         reinterpret_cast<uint8_t*>(dest.data())),
         "cmovlt");
  expect(dest[0] == 7 && dest[1] == 20 && dest[2] == 7 && dest[3] == 40, "cmovlt vals");
}

static void test_sel_uint8() {
  const uint32_t u8 = 0x00000008;
  const uint32_t nelem = 2;
  uint8_t pred[2] = {0x80, 0x01};
  uint8_t src[2] = {10, 20};
  uint8_t dest[2] = {7, 7};
  expect(ame_host_ew_sel(false, false, u8, u8, nelem, pred, src, dest), "u8 selge");
  expect(dest[0] == 10 && dest[1] == 20, "u8 selge: unsigned never < 0");
}

static void test_shift_int32() {
  const uint32_t dtype = 0x40000020;
  const uint32_t nelem = 4;
  std::vector<int32_t> data = {1, -8, 0x40000000, -1};
  std::vector<int32_t> amt = {1, 2, 1, 33}; // 33 ≡ 1 (mod 32)
  std::vector<int32_t> d(nelem);

  expect(ame_host_ew_shift(ame_shift_op_t::SLL, dtype, dtype, nelem,
                           reinterpret_cast<uint8_t*>(data.data()),
                           reinterpret_cast<uint8_t*>(amt.data()),
                           reinterpret_cast<uint8_t*>(d.data())),
         "sll");
  expect(d[0] == 2 && d[1] == -32 && d[3] == -2, "sll vals");

  expect(ame_host_ew_shift(ame_shift_op_t::SRL, dtype, dtype, nelem,
                           reinterpret_cast<uint8_t*>(data.data()),
                           reinterpret_cast<uint8_t*>(amt.data()),
                           reinterpret_cast<uint8_t*>(d.data())),
         "srl");
  expect(d[1] == int32_t(uint32_t(-8) >> 2) && d[3] == int32_t(0x7fffffffu), "srl vals");

  expect(ame_host_ew_shift(ame_shift_op_t::SRA, dtype, dtype, nelem,
                           reinterpret_cast<uint8_t*>(data.data()),
                           reinterpret_cast<uint8_t*>(amt.data()),
                           reinterpret_cast<uint8_t*>(d.data())),
         "sra");
  expect(d[1] == -2 && d[3] == -1, "sra vals");

  expect(ame_host_ew_shift_x(ame_shift_op_t::SLL, dtype, 2, nelem,
                             reinterpret_cast<uint8_t*>(data.data()),
                             reinterpret_cast<uint8_t*>(d.data())),
         "sll.x");
  expect(d[0] == 4 && d[1] == -32, "sll.x vals");

  const uint32_t udtype = 0x00000020; // Uint32: SRA unsupported
  expect(!ame_host_ew_shift(ame_shift_op_t::SRA, udtype, udtype, nelem,
                            reinterpret_cast<uint8_t*>(data.data()),
                            reinterpret_cast<uint8_t*>(amt.data()),
                            reinterpret_cast<uint8_t*>(d.data())),
         "sra unsigned UN");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_bitwise_int32();
  test_cmp_sel_int32();
  test_sel_uint8();
  test_shift_int32();
  fprintf(stderr, "ame-elementwise-bitwise-shift: ok\n");
  return 0;
}
