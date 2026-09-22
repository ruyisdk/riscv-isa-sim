#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CSR_AMENLEN 0xcc0
#define CSR_AMEOWN  0xcc3
#define CSR_AMESTATUS 0x800

#define N 8
#define INT8_DT  0x40000008u
#define INT32_DT 0x40000020u

static inline uint64_t read_csr(unsigned csr) {
  uint64_t val;
  asm volatile("csrr %0, %1" : "=r"(val) : "i"(csr));
  return val;
}

static inline void write_csr(unsigned csr, uint64_t val) {
  asm volatile("csrw %0, %1" :: "i"(csr), "r"(val));
}

static inline void enable_ame_ms(void) {
  uint64_t ms = 1ULL << 25;
  asm volatile("csrs mstatus, %0" :: "r"(ms));
}

static inline uint64_t ame_acquire_try(void) {
  register uint64_t desc asm("a1") = 0;
  register uint64_t result asm("a0");
  asm volatile(".word 0xa495852b\n" : "=r"(result) : "r"(desc) : "memory");
  return result;
}

static inline void msettyp_m0(uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  asm volatile(".word 0xa235802b\n" : : "r"(xs1) : "memory");
}

static inline void msettyp_m1(uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  // msettyp m1, a1 : rd=1, rs1=11
  asm volatile(".word 0xa23580ab\n" : : "r"(xs1) : "memory");
}

static inline void asettyp_acc0(uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  asm volatile(".word 0xa215802b\n" : : "r"(xs1) : "memory");
}

static inline void mls_rm_m0(const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  // mls.rm m0, a1 : MATCH 0xa380002b | rs1=11 | rd=0
  asm volatile(".word 0xa385802b\n" : : "r"(xs1) : "memory");
}

static inline void mls_rm_m1(const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  // mls.rm m1, a1
  asm volatile(".word 0xa38580ab\n" : : "r"(xs1) : "memory");
}

static inline void mzero_acc0(void) {
  asm volatile(".word 0xa600002b\n" ::: "memory");
}

static inline void mmulacc_acc0_m0_m1(void) {
  // mmulacc.2d acc0, m0, m1 : f7=0x4a, rs2=1, rs1=0, rd=0
  asm volatile(".word 0x9410002b\n" ::: "memory");
}

static inline void mmov_m0_acc0(void) {
  // mmov.m.a m0, acc0 : MATCH 0xa3c0002b | rs1=0 | rd=0
  asm volatile(".word 0xa3c0002b\n" ::: "memory");
}

static inline void mss_rm_m0(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  // mss.rm m0, a1 : MATCH 0xa3b0002b | rs1=11 | rd=0
  asm volatile(".word 0xa3b5802b\n" : : "r"(xs1) : "memory");
}

// Int32 path: identity * 2*identity
static int test_int32_path(void) {
  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t B[N * N] __attribute__((aligned(256)));
  static int32_t C[N * N] __attribute__((aligned(256)));
  memset(A, 0, sizeof(A));
  memset(B, 0, sizeof(B));
  memset(C, 0, sizeof(C));
  for (int i = 0; i < N; i++) {
    A[i * N + i] = 1;
    B[i * N + i] = 2;
  }

  msettyp_m0(INT32_DT);
  msettyp_m1(INT32_DT);
  asettyp_acc0(INT32_DT);
  write_csr(CSR_AMESTATUS, 0);

  mls_rm_m0(A);
  mls_rm_m1(B);
  mzero_acc0();
  mmulacc_acc0_m0_m1();
  if (read_csr(CSR_AMESTATUS) & 1) {
    printf("FAIL: int32 mmulacc UN\n");
    return 1;
  }
  mmov_m0_acc0();
  mss_rm_m0(C);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int32_t want = (i == j) ? 2 : 0;
      if (C[i * N + j] != want) {
        printf("FAIL: C[%d,%d]=%d want %d\n", i, j, C[i * N + j], want);
        return 1;
      }
    }
  }
  return 0;
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }
  if (read_csr(CSR_AMEOWN) != 1) {
    printf("FAIL: ameown\n");
    return 1;
  }
  if (read_csr(CSR_AMENLEN) != 8) {
    printf("FAIL: amenlen\n");
    return 1;
  }
  if (test_int32_path())
    return 1;
  printf("AME matmul-int32 ok\n");
  return 0;
}
