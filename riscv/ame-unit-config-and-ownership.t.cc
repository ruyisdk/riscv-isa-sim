#include "isa_parser.h"
#include "ame_encoding.h"
#include "ame_unit.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-unit-config-and-ownership: %s\n", msg);
    std::exit(1);
  }
}

static void test_params() {
  reg_t n = 0;
  expect(ame_params_legal(64, 32, 32, 4, &n) && n == 8, "default params");
  expect(ame_params_legal(256, 16, 16, 2, &n) && n == 16, "alt params");
  expect(!ame_params_legal(32, 32, 32, 4), "nelem not power of 4");
  expect(!ame_params_legal(64, 24, 32, 4), "bad udsz");
  expect(!ame_params_legal(64, 32, 8, 4), "bad mregs");
  expect(!ame_params_legal(64, 32, 32, 3), "bad accregs");
  expect(ame_params_legal(64, 32, 32, 8, &n) && n == 8, "accregs 8");
  expect(ame_params_legal(64, 32, 32, 16, &n) && n == 8, "accregs 16");
}

static void test_backend() {
  ame_backend_t be;
  int hart_a = 1, hart_b = 2;
  bool newly = false;

  expect(be.acquire(&hart_a, 0 /* TRY */, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_GRANTED, true) && newly,
         "A TRY grant");
  expect(be.owned_by(&hart_a), "A owns");

  expect(be.acquire(&hart_a, 0, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_GRANTED, true) && !newly,
         "A re-acquire");

  expect(be.acquire(&hart_b, 0, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_BUSY, false) && !newly,
         "B TRY busy");

  expect(be.acquire(&hart_b, 1 /* BOUNDED_WAIT SHORT */, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_TIMEOUT, false),
         "B BOUNDED_WAIT timeout");

  expect(be.acquire(&hart_b, 1 | (0xf << 2), &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_TIMEOUT, false),
         "B BOUNDED_WAIT class 0xF");

  expect(be.acquire(&hart_b, 2, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_BAD_DESC, false),
         "reserved WAIT_MODE");
  expect(be.acquire(&hart_b, 1 | (0x4 << 2), &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_BAD_DESC, false),
         "reserved TIMEOUT_CLASS");
  expect(be.acquire(&hart_b, 1ul << 6, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_BAD_DESC, false),
         "reserved high bits");

  be.release(&hart_b);
  expect(be.owned_by(&hart_a), "foreign release ignored");

  be.release(&hart_a);
  expect(be.owner() == nullptr, "released");

  expect(be.acquire(&hart_b, 0, &newly) ==
             AME_ACQUIRE_RESULT(AME_ACQUIRE_GRANTED, true) && newly,
         "B grant after release");
  be.release(&hart_b);
}

static void test_dump() {
  ame_unit_t amu;
  amu.ame_nelem = 64;
  amu.ame_n = 8;
  amu.ame_udsz = 32;
  amu.num_m_regs = 32;
  amu.num_acc_regs = 4;
  amu.m_data.assign(amu.num_m_regs * amu.bytes_per_m_reg(), 0);
  amu.md_desc.assign(amu.num_m_regs, 0);
  amu.acc_data.assign(amu.num_acc_regs * amu.bytes_per_acc_reg(), 0);
  amu.ad_desc.assign(amu.num_acc_regs, 0);
  amu.md_desc[0] = 0x0c000000;
  amu.m_data[0] = 0x11;
  amu.m_data[1] = 0x22;
  amu.m_data[2] = 0x33;
  amu.m_data[3] = 0x44;

  std::ostringstream os;
  amu.dump_summary(os);
  amu.dump_m(os, 0);
  amu.dump_acc(os, 0);
  const std::string s = os.str();
  expect(s.find("N=8") != std::string::npos, "dump N");
  expect(s.find("M0") != std::string::npos, "dump M0");
  expect(s.find("44332211") != std::string::npos, "little-endian word");
  expect(s.find("A0") != std::string::npos, "dump A0");
}

static void test_m32_span() {
  ame_unit_t amu;
  amu.ame_nelem = 16;
  amu.ame_n = 4;
  amu.ame_udsz = 8;
  amu.num_m_regs = 32;
  const uint32_t u8 = 0x00000008, u64 = 0x00000040;
  expect(amu.m_span_legal(16, 1, u8), "u8 m16 with 32 regs");
  expect(amu.m_span_legal(24, 1, u8), "u8 m24 with 32 regs");
  expect(amu.m_span_legal(16, 1, u64), "u64 group at m16");
  expect(amu.m_span_legal(24, 1, u64), "u64 group at m24");
  expect(!amu.m_span_legal(17, 1, u64), "u64 misaligned m17");
  amu.num_m_regs = 16;
  expect(!amu.m_span_legal(16, 1, u8), "u8 m16 illegal with 16 regs");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_params();
  test_backend();
  test_dump();
  test_m32_span();
  fprintf(stderr, "ame-unit-config-and-ownership: ok\n");
  return 0;
}
