#include "isa_parser.h"
#include "ame_encoding.h"
#include "decode.h"

#include <cstdio>
#include <cstdlib>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-acquire-release-encoding: %s\n", msg);
    std::exit(1);
  }
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");

  expect(isa.extension_enabled(EXT_ZTT), "ztt extension not enabled");

  // ame.acquire a0, a1 : xfunct5=0x09, rs1=a1, rd=a0 (spec 8.1)
  insn_t acquire(insn_bits_t(0xa495852b));
  expect((acquire.bits() & MASK_AME_ACQUIRE) == MATCH_AME_ACQUIRE,
         "ame.acquire encoding mismatch");
  expect(acquire.rs1() == 11, "ame.acquire rs1 is a1");
  expect(acquire.rd() == 10, "ame.acquire rd is a0");

  insn_t release(insn_bits_t(MATCH_AME_RELEASE));
  expect((release.bits() & MASK_AME_RELEASE) == MATCH_AME_RELEASE,
         "ame.release encoding mismatch");

  fprintf(stderr, "ame-acquire-release-encoding: ok\n");
  return 0;
}
