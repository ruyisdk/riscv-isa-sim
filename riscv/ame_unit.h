// See LICENSE for license details.
#ifndef _RISCV_AME_UNIT_H
#define _RISCV_AME_UNIT_H

#include "decode.h"
#include "csrs.h"
#include "ame_dtypes.h"
#include "ame_ops.h"

#include <cstdint>
#include <iosfwd>
#include <vector>

class processor_t;
class mmu_t;

// Exclusive AME backend lock shared by all harts of one sim_t.
class ame_backend_t {
public:
  const void* owner() const { return owner_; }
  bool owned_by(const void* who) const { return owner_ == who; }

  // xs result: SUCCESS in bit 0, STATUS in bits 7:1.
  reg_t acquire(const void* who, reg_t descriptor, bool* newly_granted);
  void release(const void* who);

private:
  const void* owner_ = nullptr;
};

bool ame_params_legal(reg_t nelem, reg_t udsz, reg_t mregs, reg_t accregs,
                      reg_t* n_out = nullptr);

class ame_unit_t {
public:
  processor_t* p = nullptr;

  reg_t ame_nelem = 64;
  reg_t ame_n = 8;
  reg_t ame_udsz = 32;
  reg_t num_m_regs = 32;
  reg_t num_acc_regs = 4;

  bool owned = false;
  ame_backend_t local_backend;
  ame_backend_t* backend = &local_backend;

  uint32_t amestype = 0;
  uint8_t amefflags = 0;
  uint8_t amexsat = 0;
  uint8_t amestatus = 0;

  std::vector<uint8_t> m_data;
  std::vector<uint32_t> md_desc;
  std::vector<uint8_t> acc_data;
  std::vector<uint32_t> ad_desc;

  void reset();
  void set_backend(ame_backend_t* shared);

  reg_t acquire(reg_t descriptor);
  void release();
  void bind_initial_state();

  void dump_summary(std::ostream& o) const;
  void dump_m(std::ostream& o, int idx) const;
  void dump_acc(std::ostream& o, int idx) const;

  void set_unsupported();
  bool ms_enabled() const;
  void set_ms(reg_t status);

  reg_t bytes_per_m_reg() const;
  reg_t bytes_per_acc_reg() const;

  uint32_t group_size_for(uint32_t dtype) const;
  uint32_t pack_factor_for(uint32_t dtype) const;

  bool m_index_legal(reg_t idx, uint32_t dtype) const;
  bool acc_index_legal(reg_t idx) const;

  void msettyp(reg_t md, uint32_t dtype);
  uint32_t mgettyp(reg_t ms1) const;
  void asettyp(reg_t ad, uint32_t dtype);
  uint32_t agettyp(reg_t ad) const;

  void clear_m_group(reg_t md, uint32_t group_size);
  void clear_acc(reg_t ad);

  // Phase 2
  uint8_t* m_bytes(reg_t idx);
  const uint8_t* m_bytes(reg_t idx) const;
  uint8_t* acc_bytes(reg_t idx);
  const uint8_t* acc_bytes(reg_t idx) const;

  reg_t rm_transfer_bytes(uint32_t dtype) const;
  reg_t align_requirement(reg_t nbytes) const;

  void mzero_m(reg_t md);
  void mzero_acc(reg_t ad);
  void mmov_m_m(reg_t md, reg_t ms);
  void mmov_m_a(reg_t md, reg_t acc);
  void mmov_a_m(reg_t acc, reg_t ms);

  void mls(reg_t md, reg_t addr, mmu_t* mmu);
  void mss(reg_t ms1, reg_t addr, mmu_t* mmu);
  void mls_rm(reg_t md, reg_t addr, mmu_t* mmu);
  void mss_rm(reg_t ms1, reg_t addr, mmu_t* mmu);
  void mls_cm(reg_t md, reg_t addr, mmu_t* mmu);
  void mss_cm(reg_t ms1, reg_t addr, mmu_t* mmu);
  void mls_st(reg_t md, reg_t addr, reg_t stride, mmu_t* mmu);
  void mss_st(reg_t ms1, reg_t addr, reg_t stride, mmu_t* mmu);
  void mls_tst(reg_t md, reg_t addr, reg_t stride, mmu_t* mmu);
  void mss_tst(reg_t ms1, reg_t addr, reg_t stride, mmu_t* mmu);

  void mmul_2d(reg_t acc, reg_t ms1, reg_t ms2, bool accumulate,
               bool negate = false, bool transpose_a = false, bool transpose_b = false);

  // Phase 4: 1D elementwise. Binary: D=f(A,B). Scalar: D=f(c,B).
  void ew_binop(ame_ew_op_t op, reg_t md, reg_t ms1, reg_t ms2);
  void ew_binop_x(ame_ew_op_t op, reg_t md, reg_t xs1_val, reg_t ms2);
  void ew_unop(ame_ew_op_t op, reg_t md, reg_t ms1);
  void ew_ldexp(reg_t md, reg_t ms1, reg_t ms2, bool accumulate, bool negate_exp);
  void ew_ldexp_x(reg_t md, reg_t xs1_val, reg_t ms2, bool accumulate);
  void ew_log2bias(reg_t md, reg_t ms1, reg_t ms2, bool bias_minus_log);
  void ew_log2bias_x(reg_t md, reg_t xs1_val, reg_t ms2, bool bias_minus_log);
  void ew_cmp(bool lt, reg_t md, reg_t ms1, reg_t ms2);
  void ew_cmp_x(bool lt, reg_t md, reg_t xs1_val, reg_t ms2);
  void ew_sel(bool lt, bool cmov, reg_t md, reg_t ms1, reg_t ms2);
  void ew_shift(ame_shift_op_t op, reg_t md, reg_t ms1, reg_t ms2);
  void ew_shift_x(ame_shift_op_t op, reg_t md, reg_t xs1_val, reg_t ms2);
  void mconv(reg_t md, reg_t ms1);
  void mpack_x(reg_t md, reg_t ms1, reg_t xs1_val);
  void munpack_x(reg_t md, reg_t ms1, reg_t xs1_val);
  void mbcast_m_x(reg_t md, reg_t xs1_val, reg_t xtyp_val);

  // Phase 5: reduce / prefix / permute.
  void fold_reduce(ame_fold_op_t op, bool along_row, reg_t md, reg_t ms1);
  void fold_prefix(ame_fold_op_t op, bool along_row, reg_t md, reg_t ms1);
  void write_ids(bool row_ids, reg_t md);
  void bcast_x(bool along_row, reg_t md, reg_t ms1, reg_t xs1_val);
  void shift_x(bool along_row, reg_t md, reg_t ms1, reg_t xs1_val);
  void gather(bool along_row, reg_t md, reg_t ms1, reg_t ms2);
  void zip2(bool along_row, bool unzip, reg_t ms1, reg_t ms2);
  void scatter(bool along_row, ame_scatter_op_t op, reg_t md, reg_t ms1, reg_t ms2);

  // Phase 7: debug scalar mmove (spec 8.58–8.65). elem_bits in {8,16,32,64}.
  void mmove_m_x(unsigned elem_bits, reg_t md, reg_t data_val, reg_t pos_val);
  reg_t mmove_x_m(unsigned elem_bits, reg_t ms, reg_t pos_val);

  bool acc_span_legal(reg_t base, uint32_t count) const;
  bool m_span_legal(reg_t base, uint32_t n_sq, uint32_t dtype) const;

private:
  void allocate_storage();
  void load_mem(mmu_t* mmu, reg_t addr, uint8_t* dst, reg_t len);
  void store_mem(mmu_t* mmu, reg_t addr, const uint8_t* src, reg_t len);
  void require_load_align(reg_t addr, reg_t align) const;
  void require_store_align(reg_t addr, reg_t align) const;

  bool prepare_typed_m(reg_t idx, uint32_t& dtype);
  reg_t element_bytes(uint32_t dtype) const;
  reg_t wrap_seg_addr(reg_t base, reg_t stride, uint32_t seg) const;
  void load_segments(mmu_t* mmu, reg_t base, reg_t stride, uint32_t nseg,
                     reg_t seg_bytes, uint8_t* dst);
  void store_segments(mmu_t* mmu, reg_t base, reg_t stride, uint32_t nseg,
                      reg_t seg_bytes, const uint8_t* src);
};

#endif
