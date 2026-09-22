#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CSR_AMEOWN 0xcc3
#define CSR_AMESTATUS 0x800
#define CSR_AMESTYPE 0x810
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

static inline void msettyp_m(unsigned md, uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  if (md == 0)
    asm volatile(".word 0xa235802b\n" : : "r"(xs1) : "memory");
  else if (md == 1)
    asm volatile(".word 0xa23580ab\n" : : "r"(xs1) : "memory"); // rd=1
  else if (md == 2)
    asm volatile(".word 0xa235812b\n" : : "r"(xs1) : "memory"); // rd=2
}

static inline void mls_rm_m(unsigned md, const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  if (md == 0)
    asm volatile(".word 0xa385802b\n" : : "r"(xs1) : "memory");
  else if (md == 1)
    asm volatile(".word 0xa38580ab\n" : : "r"(xs1) : "memory");
}

static inline void mss_rm_m2(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  // mss.rm m2, a1 : MATCH | rs1=11 | rd=2
  asm volatile(".word 0xa3b5812b\n" : : "r"(xs1) : "memory");
}

static inline void madd_ew_m2_m0_m1(void) {
  // md=2, ms1=0, ms2=1 → 0x0400002b | (1<<20) | (2<<7)
  asm volatile(".word 0x0410012b\n" ::: "memory");
}

static inline void msub_ew_m2_m0_m1(void) {
  asm volatile(".word 0x3010012b\n" ::: "memory");
}

static inline void mmul_ew_m2_m0_m1(void) {
  asm volatile(".word 0x1810012b\n" ::: "memory");
}

static inline void mmin_ew_m2_m0_m1(void) {
  asm volatile(".word 0x1410012b\n" ::: "memory");
}

static inline void mmax_ew_m2_m0_m1(void) {
  asm volatile(".word 0x0c10012b\n" ::: "memory");
}

static inline void madd_ew_x_m2_a1_m1(uint64_t xs1) {
  register uint64_t s asm("a1") = xs1;
  // md=2, xs1=a1, ms2=m1 → 0x0600002b | (1<<20) | (11<<15) | (2<<7)
  asm volatile(".word 0x0615812b\n" : : "r"(s) : "memory");
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t B[N * N] __attribute__((aligned(256)));
  static int32_t D[N * N] __attribute__((aligned(256)));

  for (int i = 0; i < N * N; i++) {
    A[i] = i + 1;
    B[i] = (i + 1) * 2;
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, INT32_DT);
  msettyp_m(1, INT32_DT);
  msettyp_m(2, INT32_DT);
  mls_rm_m(0, A);
  mls_rm_m(1, B);

  madd_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != A[i] + B[i]) {
      printf("FAIL: add[%d]=%d want %d\n", i, D[i], A[i] + B[i]);
      return 1;
    }
  }

  msub_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != B[i] - A[i]) {
      printf("FAIL: sub\n");
      return 1;
    }
  }

  mmul_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != A[i] * B[i]) {
      printf("FAIL: mul\n");
      return 1;
    }
  }

  mmin_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != (A[i] < B[i] ? A[i] : B[i])) {
      printf("FAIL: min\n");
      return 1;
    }
  }

  mmax_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != (A[i] > B[i] ? A[i] : B[i])) {
      printf("FAIL: max\n");
      return 1;
    }
  }

  write_csr(CSR_AMESTYPE, INT32_DT);
  madd_ew_x_m2_a1_m1(5);
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != 5 + B[i]) {
      printf("FAIL: add.x\n");
      return 1;
    }
  }

  printf("AME elementwise-arith ok\n");
  return 0;
}
