#include "isa_parser.h"
#include "encoding.h"
#include "ame_encoding.h"
#include "ame_ops.h"
#include "decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-mmove-and-ms-off: %s\n", msg);
    std::exit(1);
  }
}

static void test_encodings() {
  expect((MATCH_MMOVE8_M_X & MASK_MMOVE8_M_X) == MATCH_MMOVE8_M_X, "mmove8.m.x");
  expect((MATCH_MMOVE16_M_X & MASK_MMOVE16_M_X) == MATCH_MMOVE16_M_X, "mmove16.m.x");
  expect((MATCH_MMOVE32_M_X & MASK_MMOVE32_M_X) == MATCH_MMOVE32_M_X, "mmove32.m.x");
  expect((MATCH_MMOVE64_M_X & MASK_MMOVE64_M_X) == MATCH_MMOVE64_M_X, "mmove64.m.x");
  expect((MATCH_MMOVE8_X_M & MASK_MMOVE8_X_M) == MATCH_MMOVE8_X_M, "mmove8.x.m");
  expect((MATCH_MMOVE16_X_M & MASK_MMOVE16_X_M) == MATCH_MMOVE16_X_M, "mmove16.x.m");
  expect((MATCH_MMOVE32_X_M & MASK_MMOVE32_X_M) == MATCH_MMOVE32_X_M, "mmove32.x.m");
  expect((MATCH_MMOVE64_X_M & MASK_MMOVE64_X_M) == MATCH_MMOVE64_X_M, "mmove64.x.m");

  expect((MATCH_MMOVE8_M_X & MASK_MBCAST_M_X) != MATCH_MBCAST_M_X, "mmove8 != mbcast");
  expect((MATCH_MMOVE64_X_M & MASK_MCOLUNZIP_EW) != MATCH_MCOLUNZIP_EW,
         "mmove64.x.m != mcolunzip");

  // mmove32.m.x md=2, xs2=x12 (data=rs1), xs1=x11 (pos=rs2)
  insn_t st(insn_bits_t(MATCH_MMOVE32_M_X | (11 << 20) | (12 << 15) | (2 << 7)));
  expect(st.rd() == 2 && st.rs1() == 12 && st.rs2() == 11, "mmove32.m.x regs");

  // mmove32.x.m xd=10, ms2=0, xs1=11
  insn_t ld(insn_bits_t(MATCH_MMOVE32_X_M | (11 << 20) | (0 << 15) | (10 << 7)));
  expect(ld.rd() == 10 && ld.rs1() == 0 && ld.rs2() == 11, "mmove32.x.m regs");
}

static void test_mmove_host() {
  const uint32_t n = 8, udsz = 32;
  const uint32_t bytes = n * n * (udsz / 8);
  expect(ame_mmove_legal(n, udsz, 8), "8-bit legal");
  expect(ame_mmove_legal(n, udsz, 16), "16-bit legal");
  expect(ame_mmove_legal(n, udsz, 32), "32-bit legal");
  expect(ame_mmove_legal(n, udsz, 64), "64-bit legal");
  expect(!ame_mmove_legal(4, 8, 64), "narrow row illegal");
  expect(ame_mmove_count(n, udsz, 32) == 64, "32-bit count");
  expect(ame_mmove_count(n, udsz, 8) == 256, "8-bit count");

  std::vector<uint8_t> m(bytes, 0);
  ame_host_mmove_store(n, udsz, 32, m.data(), 0, 0x11111111u);
  ame_host_mmove_store(n, udsz, 32, m.data(), 1, 0x22222222u);
  ame_host_mmove_store(n, udsz, 32, m.data(), 8, 0x33333333u); // row 1, col 0
  expect(ame_host_mmove_load(n, udsz, 32, m.data(), 0) == 0x11111111u, "load 0");
  expect(ame_host_mmove_load(n, udsz, 32, m.data(), 1) == 0x22222222u, "load 1");
  expect(ame_host_mmove_load(n, udsz, 32, m.data(), 8) == 0x33333333u, "load row1");
  expect(ame_host_mmove_load(n, udsz, 32, m.data(), 64) == 0x11111111u, "pos wrap");

  ame_host_mmove_store(n, udsz, 8, m.data(), 0, 0xab);
  expect(ame_host_mmove_load(n, udsz, 8, m.data(), 0) == 0xab, "u8");
  ame_host_mmove_store(n, udsz, 16, m.data(), 3, 0xbeef);
  expect(ame_host_mmove_load(n, udsz, 16, m.data(), 3) == 0xbeef, "u16");
  ame_host_mmove_store(n, udsz, 64, m.data(), 2, 0x0102030405060708ull);
  expect(ame_host_mmove_load(n, udsz, 64, m.data(), 2) == 0x0102030405060708ull, "u64");
}

static void test_ms_off_rule() {
  expect(!AME_OWNED_MS_OFF_ILLEGAL(false, 0), "not owned Off ok");
  expect(!AME_OWNED_MS_OFF_ILLEGAL(true, 1ul << MSTATUS_AME_MS_SHIFT),
         "owned Initial ok");
  expect(AME_OWNED_MS_OFF_ILLEGAL(true, 0), "owned Off illegal");
  expect(AME_OWNED_MS_OFF_ILLEGAL(true, ~SSTATUS_AME_MS), "owned MS cleared illegal");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");
  test_encodings();
  test_mmove_host();
  test_ms_off_rule();
  fprintf(stderr, "ame-mmove-and-ms-off: ok\n");
  return 0;
}
