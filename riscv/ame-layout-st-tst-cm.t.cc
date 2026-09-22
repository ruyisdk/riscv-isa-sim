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
    fprintf(stderr, "ame-layout-st-tst-cm: %s\n", msg);
    std::exit(1);
  }
}

static void fill_reg(std::vector<uint8_t>& buf, uint32_t n, uint32_t p, uint32_t e_bits) {
  buf.assign(ame_packed_bytes(p * n * n, e_bits), 0);
  for (uint32_t sq = 0; sq < p; sq++) {
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < n; j++) {
        const uint32_t v = sq * n * n + i * n + j;
        uint8_t tmp[8] = {};
        std::memcpy(tmp, &v, sizeof(v));
        ame_copy_elem(buf.data(), sq * n * n + i * n + j, tmp, 0, e_bits);
      }
    }
  }
}

static uint32_t load_elem_u32(const uint8_t* buf, uint32_t index, uint32_t e_bits) {
  uint8_t tmp[8] = {};
  ame_copy_elem(tmp, 0, buf, index, e_bits);
  uint32_t v = 0;
  std::memcpy(&v, tmp, 4);
  return v & (e_bits == 32 ? 0xffffffffu : ((1u << e_bits) - 1));
}

static void test_encodings() {
  expect((MATCH_MLS_CM & MASK_MLS_CM) == MATCH_MLS_CM, "mls.cm match");
  expect((MATCH_MSS_CM & MASK_MSS_CM) == MATCH_MSS_CM, "mss.cm match");
  expect((MATCH_MLS_ST & MASK_MLS_ST) == MATCH_MLS_ST, "mls.st match");
  expect((MATCH_MSS_ST & MASK_MSS_ST) == MATCH_MSS_ST, "mss.st match");
  expect((MATCH_MLS_TST & MASK_MLS_TST) == MATCH_MLS_TST, "mls.tst match");
  expect((MATCH_MSS_TST & MASK_MSS_TST) == MATCH_MSS_TST, "mss.tst match");

  expect((MATCH_MLS_1R & MASK_MLS_1R) != (MATCH_MLS_CM & MASK_MLS_CM), "mls.1r != mls.cm");
  expect((MATCH_MLS_1R & MASK_MLS_1R) != (MATCH_MSS_CM & MASK_MSS_CM), "mls.1r != mss.cm");
  expect((MATCH_MLS_ST & MASK_MLS_ST) != (MATCH_MLS_TST & MASK_MLS_TST), "st != tst");

  // mls.st m0, (a1), a2 — rs1=base, rs2=stride, rd=md (spec 6.51)
  insn_t mls_st_a1_a2(insn_bits_t(0xa8c5802b));
  expect((mls_st_a1_a2.bits() & MASK_MLS_ST) == MATCH_MLS_ST, "mls.st a1,a2 match");
  expect(mls_st_a1_a2.rd() == 0, "mls.st rd=m0");
  expect(mls_st_a1_a2.rs1() == 11, "mls.st rs1=a1 (base)");
  expect(mls_st_a1_a2.rs2() == 12, "mls.st rs2=a2 (stride)");
}

static void test_cm_roundtrip() {
  const uint32_t n = 8, p = 1, e_bits = 32;
  std::vector<uint8_t> reg, cm, back;
  fill_reg(reg, n, p, e_bits);
  cm.assign(reg.size(), 0);
  back.assign(reg.size(), 0);
  ame_transpose_squares(n, p, e_bits, reg.data(), cm.data());
  ame_transpose_squares(n, p, e_bits, cm.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "cm transpose involution");

  // (0,1) in rm is at (1,0) in cm for a single square.
  expect(load_elem_u32(reg.data(), 0 * n + 1, e_bits) ==
             load_elem_u32(cm.data(), 1 * n + 0, e_bits),
         "cm swaps (i,j)");
}

static void test_st_roundtrip_packed() {
  const uint32_t n = 8, p = 4, e_bits = 8;
  std::vector<uint8_t> reg, st, back;
  fill_reg(reg, n, p, e_bits);
  st.assign(reg.size(), 0);
  back.assign(reg.size(), 0);
  ame_reg_to_st(n, p, e_bits, reg.data(), st.data());
  ame_st_to_reg(n, p, e_bits, st.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "st packed roundtrip");
  expect(load_elem_u32(st.data(), 0, e_bits) == 0, "st row0 sq0 col0");
  expect(load_elem_u32(st.data(), n, e_bits) == uint8_t(n * n), "st row0 sq1 col0");
}

// Sail memory case 0 (spec 2.6.2): N=4, UINT8, P=1, stride=16.
// .st has 4 segments of 4 bytes; padding between segments is not transferred.
//
//   +0:  01 02 03 04 | a5 x 12
//   +16: 05 06 07 08 | a5 x 12
//   +32: 09 0a 0b 0c | a5 x 12
//   +48: 0d 0e 0f 10 | a5 x 12
//
// mls.st then the .rm image must be the 16 bytes 01..10.
static void test_sail_case0_uint8_n4_stride16() {
  const uint32_t n = 4, p = 1, e_bits = 8, stride = 16;
  const uint32_t seg_bytes = ame_packed_bytes(p * n, e_bits);
  expect(seg_bytes == 4, "uint8 p=1 n=4 segment is 4 bytes");

  std::vector<uint8_t> mem(stride * n, 0xa5);
  uint8_t seq = 1;
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < n; j++)
      mem[i * stride + j] = seq++;

  expect(mem[0] == 0x01 && mem[4] == 0xa5 && mem[16] == 0x05, "sail case0 image");

  std::vector<uint8_t> st(size_t(n) * seg_bytes, 0), rm(size_t(n) * n, 0);
  for (uint32_t i = 0; i < n; i++)
    std::memcpy(st.data() + i * seg_bytes, mem.data() + i * stride, seg_bytes);
  ame_st_to_reg(n, p, e_bits, st.data(), rm.data());
  for (uint32_t k = 0; k < n * n; k++) {
    if (rm[k] != uint8_t(k + 1)) {
      fprintf(stderr, "ame-layout-st-tst-cm: sail case0 rm[%u]=0x%02x want 0x%02x\n",
              k, rm[k], k + 1);
      std::exit(1);
    }
  }

  // Inverse: mss.st must write 4-byte rows and leave the a5 padding alone.
  std::vector<uint8_t> out(stride * n, 0xa5), st2(size_t(n) * seg_bytes, 0);
  ame_reg_to_st(n, p, e_bits, rm.data(), st2.data());
  for (uint32_t i = 0; i < n; i++)
    std::memcpy(out.data() + i * stride, st2.data() + i * seg_bytes, seg_bytes);
  expect(std::memcmp(out.data(), mem.data(), mem.size()) == 0,
         "sail case0 mss.st leaves a5 padding");

  // .tst: P*N=4 segments of N*E/8=4 bytes; column j at segment j.
  const uint32_t tst_seg = ame_packed_bytes(n, e_bits);
  std::vector<uint8_t> tmem(stride * n, 0xa5);
  seq = 1;
  // Place row-major 01..10 transposed into columns.
  std::vector<uint8_t> ident(n * n);
  for (uint32_t k = 0; k < n * n; k++)
    ident[k] = uint8_t(k + 1);
  std::vector<uint8_t> tst(size_t(n) * tst_seg, 0);
  ame_reg_to_tst(n, p, e_bits, ident.data(), tst.data());
  for (uint32_t s = 0; s < n; s++)
    std::memcpy(tmem.data() + s * stride, tst.data() + s * tst_seg, tst_seg);
  expect(tmem[0] == 0x01 && tmem[1] == 0x05 && tmem[2] == 0x09 && tmem[3] == 0x0d,
         "sail case0 tst col0");
  expect(tmem[16] == 0x02 && tmem[4] == 0xa5, "sail case0 tst stride padding");

  std::vector<uint8_t> back(n * n, 0);
  ame_tst_to_reg(n, p, e_bits, tst.data(), back.data());
  expect(std::memcmp(back.data(), ident.data(), ident.size()) == 0,
         "sail case0 tst roundtrip");
}

// Spec 2.6 UINT8 packed, N=4 (AME_NELEM=16, udsz=32): .st has 4 segments of 16 bytes;
// element (sq,i,j) at offset (sq*N+j) within row i. Stride 16 is tight packing.
static void test_st_tst_uint8_n4() {
  const uint32_t n = 4, p = 4, e_bits = 8;
  std::vector<uint8_t> reg, st, tst, back;
  fill_reg(reg, n, p, e_bits);
  st.assign(reg.size(), 0);
  tst.assign(reg.size(), 0);
  back.assign(reg.size(), 0);
  ame_reg_to_st(n, p, e_bits, reg.data(), st.data());
  // Row 0 of .st: sq0[0,*], sq1[0,*], sq2[0,*], sq3[0,*]
  expect(load_elem_u32(st.data(), 0, e_bits) == 0 &&
             load_elem_u32(st.data(), 1, e_bits) == 1 &&
             load_elem_u32(st.data(), 2, e_bits) == 2 &&
             load_elem_u32(st.data(), 3, e_bits) == 3,
         "st row0 sq0");
  expect(load_elem_u32(st.data(), 4, e_bits) == uint8_t(n * n) &&
             load_elem_u32(st.data(), 5, e_bits) == uint8_t(n * n + 1),
         "st row0 sq1");
  ame_st_to_reg(n, p, e_bits, st.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "uint8 st roundtrip");

  ame_reg_to_tst(n, p, e_bits, reg.data(), tst.data());
  // Segment sq*N+j is column j of square sq; offset i is row i.
  expect(load_elem_u32(tst.data(), 0, e_bits) == 0 &&
             load_elem_u32(tst.data(), 1, e_bits) == 4 &&
             load_elem_u32(tst.data(), 2, e_bits) == 8 &&
             load_elem_u32(tst.data(), 3, e_bits) == 12,
         "tst sq0 col0");
  expect(load_elem_u32(tst.data(), 16, e_bits) == uint8_t(n * n) &&
             load_elem_u32(tst.data(), 17, e_bits) == uint8_t(n * n + n),
         "tst sq1 col0");
  back.assign(reg.size(), 0);
  ame_tst_to_reg(n, p, e_bits, tst.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "uint8 tst roundtrip");
}

static void test_tst_matches_cm() {
  const uint32_t n = 8, p = 1, e_bits = 32;
  std::vector<uint8_t> reg, cm, tst;
  fill_reg(reg, n, p, e_bits);
  cm.assign(reg.size(), 0);
  tst.assign(reg.size(), 0);
  ame_transpose_squares(n, p, e_bits, reg.data(), cm.data());
  ame_reg_to_tst(n, p, e_bits, reg.data(), tst.data());
  expect(std::memcmp(cm.data(), tst.data(), cm.size()) == 0,
         "unpacked .tst assembled equals .cm");
}

// Spec 2.6: Uint4 elements are packed back-to-back; alignment is ceil(4/8)=1 byte.
// N=4, pack_factor=8 (udsz=32) fills one M register (64 bytes).
static void test_st_tst_uint4_n4() {
  const uint32_t n = 4, p = 8, e_bits = 4;
  std::vector<uint8_t> reg, st, tst, back, cm;
  fill_reg(reg, n, p, e_bits);
  expect(reg.size() == 64, "uint4 packed register bytes");
  st.assign(reg.size(), 0);
  tst.assign(reg.size(), 0);
  back.assign(reg.size(), 0);
  cm.assign(reg.size(), 0);

  ame_reg_to_st(n, p, e_bits, reg.data(), st.data());
  // Row 0: sq0 cols then sq1 cols ...; values wrap to 4 bits.
  expect(load_elem_u32(st.data(), 0, e_bits) == 0 &&
             load_elem_u32(st.data(), 1, e_bits) == 1 &&
             load_elem_u32(st.data(), 4, e_bits) == 0,
         "uint4 st row0");
  ame_st_to_reg(n, p, e_bits, st.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "uint4 st roundtrip");

  ame_reg_to_tst(n, p, e_bits, reg.data(), tst.data());
  expect(load_elem_u32(tst.data(), 0, e_bits) == 0 &&
             load_elem_u32(tst.data(), 1, e_bits) == 4 &&
             load_elem_u32(tst.data(), 2, e_bits) == 8 &&
             load_elem_u32(tst.data(), 3, e_bits) == 12,
         "uint4 tst sq0 col0");
  back.assign(reg.size(), 0);
  ame_tst_to_reg(n, p, e_bits, tst.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "uint4 tst roundtrip");

  ame_transpose_squares(n, p, e_bits, reg.data(), cm.data());
  back.assign(reg.size(), 0);
  ame_transpose_squares(n, p, e_bits, cm.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "uint4 cm involution");
}

// UINT64 unit square at N=4: E=64, P=1, each .st segment is 32 bytes.
static void test_st_uint64_n4() {
  const uint32_t n = 4, p = 1, e_bits = 64;
  std::vector<uint8_t> reg, st, back;
  fill_reg(reg, n, p, e_bits);
  expect(reg.size() == 128, "uint64 square bytes");
  const uint32_t seg_bytes = ame_packed_bytes(p * n, e_bits);
  expect(seg_bytes == 32, "uint64 st segment bytes");
  st.assign(size_t(n) * seg_bytes, 0);
  ame_reg_to_st(n, p, e_bits, reg.data(), st.data());
  // Row 0 is elements 0..3 stored as 8-byte little-endian values.
  expect(st[0] == 0 && st[8] == 1 && st[16] == 2 && st[24] == 3, "uint64 st row0 low");
  expect(st[4] == 0 && st[12] == 0, "uint64 st row0 high bytes stay 0");
  back.assign(reg.size(), 0);
  ame_st_to_reg(n, p, e_bits, st.data(), back.data());
  expect(std::memcmp(reg.data(), back.data(), reg.size()) == 0, "uint64 st roundtrip");
}

int main() {
  isa_parser_t isa("rv64i_ztt", "MSU");
  expect(isa.extension_enabled(EXT_ZTT), "ztt enabled");

  test_encodings();
  test_cm_roundtrip();
  test_st_roundtrip_packed();
  test_sail_case0_uint8_n4_stride16();
  test_st_tst_uint8_n4();
  test_st_tst_uint4_n4();
  test_st_uint64_n4();
  test_tst_matches_cm();

  fprintf(stderr, "ame-layout-st-tst-cm: ok\n");
  return 0;
}
