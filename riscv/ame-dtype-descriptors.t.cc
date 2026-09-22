#include "isa_parser.h"
#include "ame_encoding.h"
#include "ame_dtypes.h"
#include "decode.h"

#include <cstdio>
#include <cstdlib>

static void expect(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "ame-dtype-descriptors: %s\n", msg);
    std::exit(1);
  }
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt extension not enabled");

  expect(ame_group_size(8, 32) == 1, "group_size Int8");
  expect(ame_group_size(32, 32) == 1, "group_size Int32");
  expect(ame_group_size(64, 32) == 2, "group_size Int64");
  expect(ame_pack_factor(4, 32) == 8, "pack_factor Uint4");
  expect(ame_pack_factor(8, 32) == 4, "pack_factor Int8");
  expect(ame_pack_factor(16, 32) == 2, "pack_factor Int16");
  expect(ame_pack_factor(32, 32) == 1, "pack_factor Int32");
  expect(ame_pack_factor(8, 8) == 1, "UINT8 unit at U=8");
  expect(ame_pack_factor(8, 16) == 2, "UINT8 packed at U=16");

  expect(ame_dtype_supported_m(0x00000004, 32), "Uint4 supported in M");
  expect(ame_dtype_supported_m(0x40000004, 32), "Int4 supported in M");
  expect(ame_dtype_supported_m(0x40000008, 32), "Int8 supported in M");
  expect(ame_dtype_supported_m(0x40000020, 32), "Int32 supported in M");
  expect(ame_dtype_supported_m(0x40000040, 32), "Int64 supported in M");
  expect(ame_dtype_supported_m(0x00000040, 32), "Uint64 supported in M");
  expect(ame_dtype_supported_m(0x2c300140, 32), "FP64 supported in M");
  expect(ame_dtype_supported_m(0x10100108, 32), "OCP E4M3 supported in M");
  expect(ame_dtype_supported_m(0x40000080, 32), "Int128 supported in M");
  expect(ame_dtype_supported_acc(0x40000008, 32), "Int8 supported in Acc");
  expect(ame_dtype_supported_acc(0x00000004, 32), "Uint4 supported in Acc");
  expect(ame_dtype_supported_acc(0x40000020, 32), "Int32 supported in Acc");
  expect(ame_dtype_supported_acc(0x40000040, 32), "Int64 supported in Acc");
  expect(ame_dtype_supported_acc(0x2c300140, 32), "FP64 supported in Acc");
  expect(ame_dtype_supported_acc(0x40000080, 32), "Int128 supported in Acc");
  expect(!ame_dtype_supported_m(0x00000018, 32), "Int24 unsupported in M");

  // Rounding-mode don't-care for integers
  expect(ame_dtype_supported_m(0x40000008 | (1u << 27), 32), "Int8 RNE supported");

  insn_t mset(insn_bits_t(MATCH_MSETTYP | (1 << 15) | (0 << 7))); // msettyp m0, x1
  expect((mset.bits() & MASK_MSETTYP) == MATCH_MSETTYP, "msettyp encoding");

  insn_t mget(insn_bits_t(MATCH_MGETTYP | (2 << 15) | (3 << 7))); // mgettyp x3, m2
  expect((mget.bits() & MASK_MGETTYP) == MATCH_MGETTYP, "mgettyp encoding");

  insn_t aset(insn_bits_t(MATCH_ASETTYP | (1 << 15) | (0 << 7)));
  expect((aset.bits() & MASK_ASETTYP) == MATCH_ASETTYP, "asettyp encoding");

  insn_t aget(insn_bits_t(MATCH_AGETTYP | (0 << 15) | (4 << 7)));
  expect((aget.bits() & MASK_AGETTYP) == MATCH_AGETTYP, "agettyp encoding");

  // Fixed acquire encoding: xfunct5=0x09, rs1=a1, rd=a0 (spec 8.1)
  insn_t acq(insn_bits_t(0xa495852b));
  expect((acq.bits() & MASK_AME_ACQUIRE) == MATCH_AME_ACQUIRE, "ame.acquire encoding");
  expect(acq.rs1() == 11, "ame.acquire rs1");
  expect(acq.rd() == 10, "ame.acquire rd");

  fprintf(stderr, "ame-dtype-descriptors: ok\n");
  return 0;
}
