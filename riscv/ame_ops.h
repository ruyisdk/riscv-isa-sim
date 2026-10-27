// See LICENSE for license details.
// Ztt / AME Phase-2 helpers: memory, move, matmul.

#ifndef _RISCV_AME_OPS_H
#define _RISCV_AME_OPS_H

#include "ame_dtypes.h"

#include <cstdint>

class ame_unit_t;
class mmu_t;

// Returns true if (A_dtype, B_dtype, Acc_dtype) is a supported mmul/mmulacc tuple.
bool ame_matmul_supported(uint32_t a_dtype, uint32_t b_dtype, uint32_t c_dtype,
                          uint32_t udsz);

// Pure GEMM on host buffers. Used by Spike and unit tests.
// A and B are square-major: N_sq squares of N×N (each square row-major).
// Spec 6.2.1 pairs square s of A with square s of B; C is one N×N Acc square.
// Transpose is within each square. Element sizes come from the descriptors.
bool ame_host_matmul(uint32_t n, uint32_t n_sq,
                     uint32_t a_dtype, const uint8_t* a,
                     uint32_t b_dtype, const uint8_t* b,
                     uint32_t c_dtype, uint8_t* c,
                     bool accumulate, bool negate = false,
                     bool transpose_a = false, bool transpose_b = false,
                     uint8_t* fflags = nullptr, uint8_t* xsat = nullptr);

// Layout transforms on packed squares (element size `e_bits`).
// Register / .rm: square-major, each square row-major.
// .cm: square-major, each square column-major.
// Assembled .st: N segments of ceil(P*N*E/8) bytes; element (sq,i,j)
//   is at bit (sq*N+j)*E in segment i (spec 2.6.2).
// Assembled .tst: P*N segments of ceil(N*E/8) bytes; segment sq*N+j
//   holds column j of square sq at bit i*E.
void ame_transpose_squares(uint32_t n, uint32_t p, uint32_t e_bits,
                           const uint8_t* src, uint8_t* dst);
void ame_reg_to_st(uint32_t n, uint32_t p, uint32_t e_bits,
                   const uint8_t* src, uint8_t* dst);
void ame_st_to_reg(uint32_t n, uint32_t p, uint32_t e_bits,
                   const uint8_t* src, uint8_t* dst);
void ame_reg_to_tst(uint32_t n, uint32_t p, uint32_t e_bits,
                    const uint8_t* src, uint8_t* dst);
void ame_tst_to_reg(uint32_t n, uint32_t p, uint32_t e_bits,
                    const uint8_t* src, uint8_t* dst);

// Copy one packed element (used for sub-byte dtypes such as Uint4).
void ame_copy_elem(uint8_t* dst, uint32_t dst_index,
                   const uint8_t* src, uint32_t src_index, uint32_t size_bits);

enum class ame_ew_op_t {
  ADD,
  SUB,   // D = B - A  (or c - B for scalar)
  MUL,
  MIN,
  MAX,
  AND,
  ANDNOT, // D = A & ~B
  OR,
  ORNOT,  // D = A | ~B
  XOR,
  ABS,        // D = |A|
  ABSDIFF,    // D = |A - B|  (.ew.x: |c - B|)
  HDIFF,      // D = (B - A) / 2  (.ew.x: (c - B) / 2)
  MEAN,       // D = (A + B) / 2
  MULNEG,     // D = -(A * B)
  MULACC,     // D = D + A * B
  MULACCNEG,  // D = D - A * B
  MULADD,     // D = A + B * D
  MULSUB,     // D = A - B * D
  COS, EXP2, LOG2, REC, RSQRT, SIN, SQRT, TANH,
  FRINTM, FRINTN, FRINTP, FRINTZ,
};

inline bool ame_ew_is_unary(ame_ew_op_t op) {
  switch (op) {
    case ame_ew_op_t::ABS:
    case ame_ew_op_t::COS:
    case ame_ew_op_t::EXP2:
    case ame_ew_op_t::LOG2:
    case ame_ew_op_t::REC:
    case ame_ew_op_t::RSQRT:
    case ame_ew_op_t::SIN:
    case ame_ew_op_t::SQRT:
    case ame_ew_op_t::TANH:
    case ame_ew_op_t::FRINTM:
    case ame_ew_op_t::FRINTN:
    case ame_ew_op_t::FRINTP:
    case ame_ew_op_t::FRINTZ:
      return true;
    default:
      return false;
  }
}

inline bool ame_ew_reads_dest(ame_ew_op_t op) {
  return op == ame_ew_op_t::MULACC || op == ame_ew_op_t::MULACCNEG ||
         op == ame_ew_op_t::MULADD || op == ame_ew_op_t::MULSUB;
}

//	ex×-form elementwise: the scalar (rs1) is the lhs operand and the ms2
// element is the rhs.  ame_host_ew computes SUB/HDIFF as rhs - lhs, which
// matches the manual (D[i] = B[i] - c and D[i] = (B[i] - c) / 2), so the
// operands are passed without any swap.
enum class ame_shift_op_t { SLL, SRL, SRA };

// Identical-dtype elementwise over `nelem` elements.
// Returns false if dtype unsupported. Stages amefflags/amexsat bits into *fflags/*xsat.
bool ame_host_ew(ame_ew_op_t op, uint32_t dtype, uint32_t nelem,
                 const uint8_t* a, const uint8_t* b, uint8_t* d,
                 uint8_t* fflags, uint8_t* xsat);

// Compare: dest is integer all-ones / zeros. lt=false → A>=B; lt=true → A<B.
bool ame_host_ew_cmp(bool lt, uint32_t src_dtype, uint32_t dst_dtype, uint32_t nelem,
                     const uint8_t* a, const uint8_t* b, uint8_t* d, uint8_t* fflags);

// Select / conditional move. MSB of pred encoding is the sign bit.
// cmov=true preserves dest when unselected; else writes semantic zero.
bool ame_host_ew_sel(bool lt, bool cmov, uint32_t pred_dtype, uint32_t data_dtype,
                     uint32_t nelem, const uint8_t* pred, const uint8_t* src,
                     uint8_t* dest);

// Shift: D = data << amt. amt_dtype is integer; .ew.x uses a raw unsigned control.
bool ame_host_ew_shift(ame_shift_op_t op, uint32_t data_dtype, uint32_t amt_dtype,
                       uint32_t nelem, const uint8_t* data, const uint8_t* amt,
                       uint8_t* d);
bool ame_host_ew_shift_x(ame_shift_op_t op, uint32_t data_dtype, uint64_t shamt,
                         uint32_t nelem, const uint8_t* data, uint8_t* d);

// Convert X-register scalar (amestype) into tensor dtype bits in `out_elem`.
// Returns false if conversion unsupported.
bool ame_convert_scalar(uint32_t scalar_dtype, uint64_t xval, uint32_t xlen,
                        uint32_t tensor_dtype, uint8_t* out_elem);

// Element conversion (spec 2.5 / 8.26). Unsupported tuples return false.
bool ame_host_convert_elem(uint32_t src_dtype, const uint8_t* src,
                           uint32_t dst_dtype, uint8_t* dst,
                           uint8_t* fflags, uint8_t* xsat);
bool ame_host_convert(uint32_t src_dtype, uint32_t dst_dtype, uint32_t nelem,
                      const uint8_t* src, uint8_t* dst,
                      uint8_t* fflags, uint8_t* xsat);

// D = data × 2^(±exp). data is numeric; exp is integer. accumulate → D += ...
bool ame_host_ldexp(uint32_t data_dtype, uint32_t exp_dtype, uint32_t nelem,
                    const uint8_t* data, const uint8_t* expn, uint8_t* dest,
                    bool accumulate, bool negate_exp, uint8_t* fflags, uint8_t* xsat);
bool ame_host_ldexp_x(uint32_t data_dtype, uint32_t nelem, const uint8_t* data,
                      int64_t exp, uint8_t* dest, bool accumulate,
                      uint8_t* fflags, uint8_t* xsat);

// log2sub: D = log2(|A|) - B; sublog2: D = B - log2(|A|). FP only.
bool ame_host_log2bias(uint32_t dtype, uint32_t nelem,
                       const uint8_t* a, const uint8_t* bias, uint8_t* dest,
                       bool bias_minus_log, uint8_t* fflags);

enum class ame_fold_op_t { ADD, MIN, MAX };

// Reduce/prefix over one N×N square (row-major). along_row=true → operate across columns.
bool ame_host_reduce(ame_fold_op_t op, bool along_row, uint32_t dtype, uint32_t n,
                     const uint8_t* src, uint8_t* dst, uint8_t* fflags, uint8_t* xsat);
bool ame_host_prefix(ame_fold_op_t op, bool along_row, uint32_t dtype, uint32_t n,
                     const uint8_t* src, uint8_t* dst, uint8_t* fflags, uint8_t* xsat);

// Permute one square. Indices use low log2(n) bits for gather.
bool ame_host_bcast(bool along_row, uint32_t dtype, uint32_t n,
                    const uint8_t* src, uint8_t* dst, uint32_t idx);
bool ame_host_shift(bool along_row, uint32_t dtype, uint32_t n,
                    const uint8_t* src, uint8_t* dst, int32_t offset);
bool ame_host_gather(bool along_row, uint32_t dtype, uint32_t idx_dtype, uint32_t n,
                     const uint8_t* src, const uint8_t* idx, uint8_t* dst);
bool ame_host_write_ids(bool row_ids, uint32_t dtype, uint32_t n, uint8_t* dst);

// Zip/unzip two N×N squares. unzip=false → zip. along_row=true → rows.
bool ame_host_zip2(bool along_row, bool unzip, uint32_t dtype, uint32_t n,
                   const uint8_t* a, const uint8_t* b,
                   uint8_t* out1, uint8_t* out2);

enum class ame_scatter_op_t { ADD, MAX };

// dest is in/out (scatter-accumulate). Indices use low log2(n) bits.
bool ame_host_scatter(bool along_row, ame_scatter_op_t op,
                      uint32_t dtype, uint32_t idx_dtype, uint32_t n,
                      const uint8_t* src, const uint8_t* idx, uint8_t* dest,
                      uint8_t* fflags, uint8_t* xsat);

// Debug mmove: one M register as N rows of (N*udsz/elem_bits) elements.
bool ame_mmove_legal(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits);
uint32_t ame_mmove_count(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits);
void ame_host_mmove_store(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits,
                          uint8_t* mreg, uint32_t pos, uint64_t data);
uint64_t ame_host_mmove_load(uint32_t n, uint32_t udsz_bits, uint32_t elem_bits,
                             const uint8_t* mreg, uint32_t pos);

#endif
