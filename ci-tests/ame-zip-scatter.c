#include <stdint.h>
#include <stdio.h>

#define CSR_AMESTATUS 0x800
#define N 8
#define INT32_DT 0x40000020u

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

static inline void msettyp_m(unsigned md, uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  if (md == 0)
    asm volatile(".word 0xa235802b\n" : : "r"(xs1) : "memory");
  else if (md == 1)
    asm volatile(".word 0xa23580ab\n" : : "r"(xs1) : "memory");
  else if (md == 2)
    asm volatile(".word 0xa235812b\n" : : "r"(xs1) : "memory");
}

static inline void mls_rm_m(unsigned md, const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  if (md == 0)
    asm volatile(".word 0xa385802b\n" : : "r"(xs1) : "memory");
  else if (md == 1)
    asm volatile(".word 0xa38580ab\n" : : "r"(xs1) : "memory");
}

static inline void mss_rm_m(unsigned ms, void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  if (ms == 0)
    asm volatile(".word 0xa3b5802b\n" : : "r"(xs1) : "memory");
  else if (ms == 1)
    asm volatile(".word 0xa3b580ab\n" : : "r"(xs1) : "memory");
  else if (ms == 2)
    asm volatile(".word 0xa3b5812b\n" : : "r"(xs1) : "memory");
}

static inline void mcolzip_m0_m1(void) {
  // MATCH_MCOLZIP_EW | rs2=1 | rs1=0
  asm volatile(".word 0xc410002b\n" ::: "memory");
}

static inline void mcolunzip_m0_m1(void) {
  asm volatile(".word 0xc210002b\n" ::: "memory");
}

static inline void mcolscatadd_m2_m0_m1(void) {
  // MATCH | rs2=1 | rs1=0 | rd=2
  asm volatile(".word 0x7410012b\n" ::: "memory");
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t B[N * N] __attribute__((aligned(256)));
  static int32_t D0[N * N] __attribute__((aligned(256)));
  static int32_t D1[N * N] __attribute__((aligned(256)));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i * 10 + j;
      B[i * N + j] = 100 + i * 10 + j;
    }
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, INT32_DT);
  msettyp_m(1, INT32_DT);
  msettyp_m(2, INT32_DT);
  mls_rm_m(0, A);
  mls_rm_m(1, B);
  mcolzip_m0_m1();
  mss_rm_m(0, D0);
  mss_rm_m(1, D1);
  for (int j = 0; j < N / 2; j++) {
    if (D0[2 * j] != A[j] || D0[2 * j + 1] != B[j] ||
        D1[2 * j] != A[N / 2 + j] || D1[2 * j + 1] != B[N / 2 + j]) {
      printf("FAIL: colzip\n");
      return 1;
    }
  }

  mcolunzip_m0_m1();
  mss_rm_m(0, D0);
  mss_rm_m(1, D1);
  for (int i = 0; i < N * N; i++) {
    if (D0[i] != A[i] || D1[i] != B[i]) {
      printf("FAIL: colunzip inverse\n");
      return 1;
    }
  }

  // Scatter-add columns of A into D=m2 using indices in m1 (all 1s).
  for (int i = 0; i < N * N; i++)
    B[i] = 1;
  for (int i = 0; i < N * N; i++)
    D0[i] = 0;
  mls_rm_m(0, A);
  mls_rm_m(1, B);
  mls_rm_m(2, D0);
  mcolscatadd_m2_m0_m1();
  mss_rm_m(2, D0);
  for (int i = 0; i < N; i++) {
    int sum = 0;
    for (int j = 0; j < N; j++)
      sum += A[i * N + j];
    if (D0[i * N + 1] != sum) {
      printf("FAIL: colscatadd\n");
      return 1;
    }
    for (int j = 0; j < N; j++) {
      if (j != 1 && D0[i * N + j] != 0) {
        printf("FAIL: colscatadd other\n");
        return 1;
      }
    }
  }

  printf("AME zip-scatter ok\n");
  return 0;
}
