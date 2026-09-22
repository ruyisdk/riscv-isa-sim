#include <stdint.h>
#include <stdio.h>

#define CSR_AMENLEN 0xcc0
#define CSR_AMEUDSZ 0xcc1
#define CSR_AMEOWN  0xcc3
#define CSR_AMESTATUS 0x800

static inline uint64_t read_csr(unsigned csr) {
  uint64_t val;
  asm volatile("csrr %0, %1" : "=r"(val) : "i"(csr));
  return val;
}

static inline void write_csr(unsigned csr, uint64_t val) {
  asm volatile("csrw %0, %1" :: "i"(csr), "r"(val));
}

static inline void enable_ame_ms(void) {
  uint64_t ms = 1ULL << 25; // MS = Initial
  asm volatile("csrs mstatus, %0" :: "r"(ms));
}

// ame.acquire a0, a1 with TRY descriptor in a1
static inline uint64_t ame_acquire_try(void) {
  register uint64_t desc asm("a1") = 0;
  register uint64_t result asm("a0");
  asm volatile(
      ".word 0xa495852b\n"
      : "=r"(result)
      : "r"(desc)
      : "memory");
  return result;
}

static inline void ame_release_insn(void) {
  asm volatile(".word 0xa4a0002b");
}

// msettyp md=a0_idx encoded in rd, dtype in xs1=a1
// MATCH_MSETTYP | (rs1<<15) | (rd<<7)
static inline void msettyp_m0(uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  asm volatile(
      ".word 0xa235802b\n" // msettyp m0, a1 : rd=0, rs1=11
      :
      : "r"(xs1)
      : "memory");
}

static inline uint64_t mgettyp_m0(void) {
  register uint64_t xd asm("a0");
  asm volatile(
      ".word 0xa220052b\n" // mgettyp a0, m0 : rd=10, rs1=0
      : "=r"(xd)
      :
      : "memory");
  return xd;
}

static inline void asettyp_acc0(uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  asm volatile(
      ".word 0xa215802b\n" // asettyp acc0, a1
      :
      : "r"(xs1)
      : "memory");
}

static inline uint64_t agettyp_acc0(void) {
  register uint64_t xd asm("a0");
  asm volatile(
      ".word 0xa200052b\n" // agettyp a0, acc0
      : "=r"(xd)
      :
      : "memory");
  return xd;
}

int main(void) {
  enable_ame_ms();

  if (read_csr(CSR_AMENLEN) != 8) {
    printf("FAIL: amenlen\n");
    return 1;
  }
  if (read_csr(CSR_AMEUDSZ) != 32) {
    printf("FAIL: ameudsz\n");
    return 1;
  }

  uint64_t result = ame_acquire_try();
  if ((result & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }
  if (read_csr(CSR_AMEOWN) != 1) {
    printf("FAIL: ameown\n");
    return 1;
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m0(0x40000008); // Int8
  if (read_csr(CSR_AMESTATUS) & 1) {
    printf("FAIL: Int8 should be supported\n");
    return 1;
  }
  if (mgettyp_m0() != 0x40000008) {
    printf("FAIL: mgettyp Int8\n");
    return 1;
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m0(0x00000004); // Uint4
  if (read_csr(CSR_AMESTATUS) & 1) {
    printf("FAIL: Uint4 should be supported\n");
    return 1;
  }
  if (mgettyp_m0() != 0x00000004) {
    printf("FAIL: mgettyp Uint4\n");
    return 1;
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m0(0x00000018); // 24-bit integer is not a standard AME dtype
  if (!(read_csr(CSR_AMESTATUS) & 1)) {
    printf("FAIL: Int24 should set UN\n");
    return 1;
  }
  if (mgettyp_m0() != 0x00000004) {
    printf("FAIL: Md should be unchanged after UN\n");
    return 1;
  }

  write_csr(CSR_AMESTATUS, 0);
  asettyp_acc0(0x40000020); // Int32
  if (read_csr(CSR_AMESTATUS) & 1) {
    printf("FAIL: Int32 Acc should be supported\n");
    return 1;
  }
  if (agettyp_acc0() != 0x40000020) {
    printf("FAIL: agettyp Int32\n");
    return 1;
  }

  ame_release_insn();
  if (read_csr(CSR_AMEOWN) != 0) {
    printf("FAIL: ameown after release\n");
    return 1;
  }

  printf("AME dtype-csr ok\n");
  return 0;
}
