#include <stdint.h>
#include <stdio.h>

#define CSR_AMESTATUS 0x800
#define N 8
#define INT32_DT 0x40000020u
#define INT8_DT  0x40000008u

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
}

static inline void mconv_m1_m0(void) {
  // MATCH_MCONV_EW | rs1=0 | rd=1
  asm volatile(".word 0xa2d000ab\n" ::: "memory");
}

static inline void mbcast_m2_a1_a2(uint64_t xs1, uint64_t xtyp) {
  register uint64_t s asm("a1") = xs1;
  register uint64_t t asm("a2") = xtyp;
  // MATCH_MBCAST_M_X | rs2=a2(12) | rs1=a1(11) | rd=2
  asm volatile(".word 0xb0c5812b\n" : : "r"(s), "r"(t) : "memory");
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t D[N * N] __attribute__((aligned(256)));
  for (int i = 0; i < N * N; i++)
    A[i] = (i & 1) ? -(i + 1) : (i + 1);

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, INT32_DT);
  msettyp_m(1, INT32_DT);
  msettyp_m(2, INT32_DT);
  mls_rm_m(0, A);
  mconv_m1_m0();
  mss_rm_m(1, D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != A[i]) {
      printf("FAIL: mconv same-type\n");
      return 1;
    }
  }

  mbcast_m2_a1_a2(7, INT32_DT);
  mss_rm_m(2, D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != 7) {
      printf("FAIL: mbcast\n");
      return 1;
    }
  }

  printf("AME convert-bcast ok\n");
  return 0;
}
