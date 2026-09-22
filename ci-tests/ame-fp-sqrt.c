#include <stdint.h>
#include <stdio.h>

#define CSR_AMESTATUS 0x800
#define N 8
#define FP32_DT 0x20300120u

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
}

static inline void msqrt_m1_m0(void) {
  // MATCH_MSQRT_EW | rs1=0 | rd=1
  asm volatile(".word 0xa34000ab\n" ::: "memory");
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static float A[N * N] __attribute__((aligned(256)));
  static float D[N * N] __attribute__((aligned(256)));
  for (int i = 0; i < N * N; i++)
    A[i] = 4.0f;

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, FP32_DT);
  msettyp_m(1, FP32_DT);
  mls_rm_m(0, A);
  msqrt_m1_m0();
  mss_rm_m(1, D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != 2.0f) {
      printf("FAIL: msqrt\n");
      return 1;
    }
  }

  printf("AME fp-sqrt ok\n");
  return 0;
}
