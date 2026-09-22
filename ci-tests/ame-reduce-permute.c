#include <stdint.h>
#include <stdio.h>

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

static inline void mss_rm_m2(void* addr) {
  register uint64_t xs1 asm("a1") = (uint64_t)addr;
  asm volatile(".word 0xa3b5812b\n" : : "r"(xs1) : "memory");
}

static inline void mreduceadd_row_m2_m0(void) {
  // MATCH | rs1=0 | rd=2
  asm volatile(".word 0xa440012b\n" ::: "memory");
}

static inline void mcolid_m1(void) {
  // MATCH_MCOLID_EW | rd=1
  asm volatile(".word 0xa60100ab\n" ::: "memory");
}

static inline void mcolbcast_m2_m0_a1(uint64_t col) {
  register uint64_t xs asm("a1") = col;
  // mcolbcast.ew.x m2, a1, m0 : MATCH | rs2=m0 | rs1=a1 | rd=m2
  asm volatile(".word 0x6405812b\n" : : "r"(xs) : "memory");
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  static int32_t A[N * N] __attribute__((aligned(256)));
  static int32_t D[N * N] __attribute__((aligned(256)));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i * N + j] = i * 10 + j;

  write_csr(CSR_AMESTATUS, 0);
  msettyp_m(0, INT32_DT);
  msettyp_m(1, INT32_DT);
  msettyp_m(2, INT32_DT);
  mls_rm_m(0, A);

  mreduceadd_row_m2_m0();
  mss_rm_m2(D);
  int sum0 = 0;
  for (int j = 0; j < N; j++)
    sum0 += A[j];
  for (int j = 0; j < N; j++) {
    if (D[j] != sum0) {
      printf("FAIL: row reduce\n");
      return 1;
    }
  }

  mcolid_m1();
  // store m1 via mss.rm m1
  {
    register uint64_t xs1 asm("a1") = (uint64_t)D;
    asm volatile(".word 0xa3b580ab\n" : : "r"(xs1) : "memory");
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (D[i * N + j] != j) {
        printf("FAIL: colid\n");
        return 1;
      }
    }
  }

  mcolbcast_m2_m0_a1(3);
  mss_rm_m2(D);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (D[i * N + j] != A[i * N + 3]) {
        printf("FAIL: colbcast\n");
        return 1;
      }
    }
  }

  printf("AME reduce-permute ok\n");
  return 0;
}
