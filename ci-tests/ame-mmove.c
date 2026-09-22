#include <stdint.h>
#include <stdio.h>

#define N 8

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

static inline void mmove32_m_x_m0(uint32_t data, uint32_t pos) {
  register uint64_t xs2 asm("a2") = data;
  register uint64_t xs1 asm("a1") = pos;
  // MATCH_MMOVE32_M_X | rs2=a1 | rs1=a2 | rd=m0
  asm volatile(".word 0xb6b6002b\n" : : "r"(xs1), "r"(xs2) : "memory");
}

static inline uint64_t mmove32_x_m_m0(uint32_t pos) {
  register uint64_t xs1 asm("a1") = pos;
  register uint64_t xd asm("a0");
  // MATCH_MMOVE32_X_M | rs2=a1 | rs1=m0 | rd=a0
  asm volatile(".word 0xbeb0052b\n" : "=r"(xd) : "r"(xs1) : "memory");
  return xd;
}

int main(void) {
  enable_ame_ms();
  if ((ame_acquire_try() & 1) != 1) {
    printf("FAIL: acquire\n");
    return 1;
  }

  mmove32_m_x_m0(0x11223344u, 0);
  mmove32_m_x_m0(0xaabbccddu, 1);
  if (mmove32_x_m_m0(0) != 0x11223344u || mmove32_x_m_m0(1) != 0xaabbccddu) {
    printf("FAIL: mmove32\n");
    return 1;
  }
  if (mmove32_x_m_m0(64) != 0x11223344u) {
    printf("FAIL: mmove32 wrap\n");
    return 1;
  }

  printf("AME mmove ok\n");
  return 0;
}
