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
}

static inline void asettyp_acc0(uint32_t dtype) {
  register uint64_t xs1 asm("a1") = dtype;
  asm volatile(".word 0xa215802b\n" : : "r"(xs1) : "memory");
}

static inline void mls_rm_m(unsigned md, const void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  if (md == 0)
    asm volatile(".word 0xa385802b\n" : : "r"(xs1) : "memory");
  else
    asm volatile(".word 0xa38580ab\n" : : "r"(xs1) : "memory");
}

static inline void mss_rm_m0(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  asm volatile(".word 0xa3b5802b\n" : : "r"(xs1) : "memory");
}

static inline void mzero_acc0(void) {
  asm volatile(".word 0xa600002b\n" ::: "memory");
}

static inline void mmulatacc_acc0_m0_m1(void) {
  // MATCH_MMULATACC_2D | rs2=1 | rs1=0 | rd=0
  asm volatile(".word 0x9a10002b\n" ::: "memory");
}

static inline void mmov_m0_acc0(void) {
  asm volatile(".word 0xa3c0002b\n" ::: "memory");
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t I[N * N] __attribute__((aligned(256)));
  static int32_t D[N * N] __attribute__((aligned(256)));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i * 10 + j;
      I[i * N + j] = (i == j) ? 1 : 0;
    }
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, INT32_DT);
  msettyp_m(1, INT32_DT);
  asettyp_acc0(INT32_DT);
  mls_rm_m(0, A);
  mls_rm_m(1, I);
  mzero_acc0();
  mmulatacc_acc0_m0_m1();
  mmov_m0_acc0();
  mss_rm_m0(D);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (D[i * N + j] != A[j * N + i]) {
        printf("FAIL: mmulatacc\n");
        return 1;
      }
    }
  }

  printf("AME matmul-transpose-accumulate ok\n");
  return 0;
}
