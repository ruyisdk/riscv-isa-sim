#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CSR_AMESTATUS 0x800
#define CSR_AMESTYPE 0x810
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
  else if (md == 2)
    asm volatile(".word 0xa385812b\n" : : "r"(xs1) : "memory");
}

static inline void mss_rm_m2(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  asm volatile(".word 0xa3b5812b\n" : : "r"(xs1) : "memory");
}

static inline void mabs_ew_m2_m0(void) {
  // MATCH_MABS_EW | rs1=0 | rd=2
  asm volatile(".word 0xa240012b\n" ::: "memory");
}

static inline void mmean_ew_m2_m0_m1(void) {
  // MATCH_MMEAN_EW | rs2=1 | rd=2
  asm volatile(".word 0x1010012b\n" ::: "memory");
}

static inline void mhdiff_ew_m2_m0_m1(void) {
  asm volatile(".word 0x0810012b\n" ::: "memory");
}

static inline void mmulacc_ew_m2_m0_m1(void) {
  asm volatile(".word 0x1c10012b\n" ::: "memory");
}

static inline void mmean_ew_x_m2_a1_m1(uint64_t xs1) {
  register uint64_t s asm("a1") = xs1;
  // MATCH_MMEAN_EW_X | rs2=1 | rs1=a1 | rd=2
  asm volatile(".word 0x1215812b\n" : : "r"(s) : "memory");
}

static int32_t rnu_div2(int32_t v) {
  return (v >> 1) + (v & 1);
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t B[N * N] __attribute__((aligned(256)));
  static int32_t C[N * N] __attribute__((aligned(256)));
  static int32_t D[N * N] __attribute__((aligned(256)));
  for (int i = 0; i < N * N; i++) {
    A[i] = (i & 1) ? -(i + 1) : (i + 1);
    B[i] = (i + 1) * 2;
    C[i] = i;
  }

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, INT32_DT);
  msettyp_m(1, INT32_DT);
  msettyp_m(2, INT32_DT);
  mls_rm_m(0, A);
  mls_rm_m(1, B);

  mabs_ew_m2_m0();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    int32_t want = A[i] < 0 ? -A[i] : A[i];
    if (D[i] != want) {
      printf("FAIL: abs\n");
      return 1;
    }
  }

  mmean_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != rnu_div2(A[i] + B[i])) {
      printf("FAIL: mean\n");
      return 1;
    }
  }

  mhdiff_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != rnu_div2(B[i] - A[i])) {
      printf("FAIL: hdiff\n");
      return 1;
    }
  }

  mls_rm_m(2, C);
  mmulacc_ew_m2_m0_m1();
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != C[i] + A[i] * B[i]) {
      printf("FAIL: mulacc\n");
      return 1;
    }
  }

  write_csr(CSR_AMESTYPE, INT32_DT);
  mmean_ew_x_m2_a1_m1(5);
  mss_rm_m2(D);
  for (int i = 0; i < N * N; i++) {
    if (D[i] != rnu_div2(5 + B[i])) {
      printf("FAIL: mean.x\n");
      return 1;
    }
  }

  printf("AME elementwise-abs-mean ok\n");
  return 0;
}
