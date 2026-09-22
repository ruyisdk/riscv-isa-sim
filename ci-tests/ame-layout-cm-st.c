#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CSR_AMEOWN 0xcc3
#define CSR_AMESTATUS 0x800
#define N 8
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

static inline void mls_rm_m0(const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  asm volatile(".word 0xa385802b\n" : : "r"(xs1) : "memory");
}

static inline void mss_cm_m0(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  // mss.cm m0, a1 : MATCH 0xa3a0002b | rs1=11 | rd=0
  asm volatile(".word 0xa3a5802b\n" : : "r"(xs1) : "memory");
}

static inline void mls_cm_m0(const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  // mls.cm m0, a1 : MATCH 0xa370002b | rs1=11 | rd=0
  asm volatile(".word 0xa375802b\n" : : "r"(xs1) : "memory");
}

static inline void mss_rm_m0(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  asm volatile(".word 0xa3b5802b\n" : : "r"(xs1) : "memory");
}

static inline void mss_st_m0(void* addr, uint64_t stride) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  register uint64_t xs2 asm("a2") = stride;
  // mss.st m0, (a1), a2 : f7=0x56, rs2=12, rs1=11, rd=0
  asm volatile(".word 0xacc5802b\n" : : "r"(xs1), "r"(xs2) : "memory");
}

static inline void mls_st_m0(const void* addr, uint64_t stride) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  register uint64_t xs2 asm("a2") = stride;
  // mls.st m0, (a1), a2 : f7=0x54, rs2=12, rs1=11, rd=0
  asm volatile(".word 0xa8c5802b\n" : : "r"(xs1), "r"(xs2) : "memory");
}

static int test_cm_roundtrip(void) {
  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t CM[N * N] __attribute__((aligned(256)));
  static int32_t BACK[N * N] __attribute__((aligned(256)));
  memset(A, 0, sizeof(A));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i * N + j] = i * 10 + j;

  msettyp_m0(INT32_DT);
  write_csr(CSR_AMESTATUS, 0);
  mls_rm_m0(A);
  mss_cm_m0(CM);
  mls_cm_m0(CM);
  mss_rm_m0(BACK);

  if (read_csr(CSR_AMESTATUS) & 1) {
    printf("FAIL: cm path UN\n");
    return 1;
  }
  for (int i = 0; i < N * N; i++) {
    if (BACK[i] != A[i]) {
      printf("FAIL: cm roundtrip [%d]=%d want %d\n", i, BACK[i], A[i]);
      return 1;
    }
  }
  if (CM[1] != A[N]) {
    printf("FAIL: cm layout CM[0,1]=%d want A[1,0]=%d\n", CM[1], A[N]);
    return 1;
  }
  return 0;
}

static int test_st_roundtrip(void) {
  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t ST[N * 16] __attribute__((aligned(256)));
  static int32_t BACK[N * N] __attribute__((aligned(256)));
  memset(A, 0, sizeof(A));
  memset(ST, 0, sizeof(ST));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i * N + j] = 100 + i * N + j;

  const uint64_t stride = 16 * sizeof(int32_t);
  msettyp_m0(INT32_DT);
  write_csr(CSR_AMESTATUS, 0);
  mls_rm_m0(A);
  mss_st_m0(ST, stride);
  mls_st_m0(ST, stride);
  mss_rm_m0(BACK);

  if (read_csr(CSR_AMESTATUS) & 1) {
    printf("FAIL: st path UN\n");
    return 1;
  }
  for (int i = 0; i < N * N; i++) {
    if (BACK[i] != A[i]) {
      printf("FAIL: st roundtrip [%d]=%d want %d\n", i, BACK[i], A[i]);
      return 1;
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
  if (test_cm_roundtrip())
    return 1;
  if (test_st_roundtrip())
    return 1;
  printf("AME layout-cm-st ok\n");
  return 0;
}
