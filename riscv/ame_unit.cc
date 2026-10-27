// See LICENSE for license details.

#include "config.h"
#include "ame_unit.h"
#include "processor.h"
#include "isa_parser.h"
#include "ame_encoding.h"
#include "mmu.h"
#include "trap.h"
#include "cfg.h"

#include <cstring>
#include <algorithm>
#include <vector>
#include <ostream>
#include <iomanip>
#include <cstdio>
#include <cinttypes>

namespace {

class ame_csr_t : public csr_t {
 public:
  ame_csr_t(processor_t* const proc, const reg_t addr, ame_unit_t* amu, bool context_state)
    : csr_t(proc, addr), amu(amu), context_state(context_state) {}

  void verify_permissions(insn_t insn, bool write) const override {
    if (!proc->extension_enabled(EXT_ZTT))
      throw trap_illegal_instruction(insn.bits());
    if (!amu->ms_enabled())
      throw trap_illegal_instruction(insn.bits());
    if (context_state && !amu->owned)
      throw trap_illegal_instruction(insn.bits());
    // Spec §3.2 puts amestype/amefflags/amexsat at 0x810/0x811/0x812. Privileged
    // encoding bits [11:10]=11 would make csr_t treat them as read-only;
    // software may write any 32-bit value to amestype.
    csr_t::verify_permissions(insn, false);
    if (write && !context_state)
      throw trap_illegal_instruction(insn.bits());
  }

 protected:
  ame_unit_t* amu;
  bool context_state;
};

class ameown_csr_t final : public ame_csr_t {
 public:
  ameown_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMEOWN, amu, false) {}

  reg_t read() const noexcept override {
    return amu->owned ? 1 : 0;
  }

 protected:
  bool unlogged_write(const reg_t) noexcept override {
    return false;
  }
};

class amenlen_csr_t final : public ame_csr_t {
 public:
  amenlen_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMENLEN, amu, false) {}

  reg_t read() const noexcept override {
    return amu->ame_n;
  }

 protected:
  bool unlogged_write(const reg_t) noexcept override {
    return false;
  }
};

class ameudsz_csr_t final : public ame_csr_t {
 public:
  ameudsz_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMEUDSZ, amu, false) {}

  reg_t read() const noexcept override {
    return amu->ame_udsz;
  }

 protected:
  bool unlogged_write(const reg_t) noexcept override {
    return false;
  }
};

class amestype_csr_t final : public ame_csr_t {
 public:
  amestype_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMESTYPE, amu, true) {}

  reg_t read() const noexcept override {
    return amu->amestype;
  }

 protected:
  bool unlogged_write(const reg_t val) noexcept override {
    amu->amestype = uint32_t(val);
    return true;
  }
};

class amefflags_csr_t final : public ame_csr_t {
 public:
  amefflags_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMEFFLAGS, amu, true) {}

  reg_t read() const noexcept override {
    return amu->amefflags;
  }

 protected:
  bool unlogged_write(const reg_t val) noexcept override {
    amu->amefflags = val & 0x1f;
    return true;
  }
};

class amexsat_csr_t final : public ame_csr_t {
 public:
  amexsat_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMEXSAT, amu, true) {}

  reg_t read() const noexcept override {
    return amu->amexsat;
  }

 protected:
  bool unlogged_write(const reg_t val) noexcept override {
    amu->amexsat = val & 1;
    return true;
  }
};

class amestatus_csr_t final : public ame_csr_t {
 public:
  amestatus_csr_t(processor_t* const proc, ame_unit_t* amu)
    : ame_csr_t(proc, CSR_AMESTATUS, amu, true) {}

  reg_t read() const noexcept override {
    return amu->amestatus;
  }

 protected:
  bool unlogged_write(const reg_t val) noexcept override {
    // Local diagnostic fix: UN is write-zero-to-clear, not writable-set.
    amu->amestatus &= val & AMESTATUS_UN;
    return true;
  }
};

} // namespace

static uint32_t packed_nbytes(uint32_t dtype, uint32_t nelem) {
  return ame_packed_bytes(nelem, ame_parse_dtype(dtype).size_bits);
}

static uint32_t n_sq_max(const ame_unit_t* u, uint32_t d0, uint32_t d1 = 0,
                         uint32_t d2 = 0) {
  uint32_t n = 1;
  if (d0)
    n = std::max(n, u->pack_factor_for(d0));
  if (d1)
    n = std::max(n, u->pack_factor_for(d1));
  if (d2)
    n = std::max(n, u->pack_factor_for(d2));
  return n;
}

static void read_squares(const ame_unit_t* u, reg_t ms, uint32_t dtype, uint32_t n_sq,
                         std::vector<uint8_t>& out) {
  const uint32_t sqb = packed_nbytes(dtype, uint32_t(u->ame_nelem));
  const uint32_t pf = u->pack_factor_for(dtype);
  const uint32_t gs = u->group_size_for(dtype);
  out.assign(size_t(n_sq) * sqb, 0);
  const uint32_t nreg = (pf > 1) ? (n_sq + pf - 1) / pf : n_sq;
  if (pf > 1) {
    for (uint32_t r = 0; r < nreg; r++) {
      const uint32_t nlane = std::min(pf, n_sq - r * pf);
      std::memcpy(out.data() + size_t(r) * pf * sqb, u->m_bytes(ms + r * gs),
                  size_t(nlane) * sqb);
    }
  } else {
    for (uint32_t s = 0; s < n_sq; s++)
      std::memcpy(out.data() + size_t(s) * sqb, u->m_bytes(ms + s * gs), sqb);
  }
}

static void write_squares(ame_unit_t* u, reg_t md, uint32_t dtype, uint32_t n_sq,
                          const uint8_t* src) {
  const uint32_t sqb = packed_nbytes(dtype, uint32_t(u->ame_nelem));
  const uint32_t pf = u->pack_factor_for(dtype);
  const uint32_t gs = u->group_size_for(dtype);
  const uint32_t nreg = (pf > 1) ? (n_sq + pf - 1) / pf : n_sq;
  if (pf > 1) {
    for (uint32_t r = 0; r < nreg; r++) {
      const uint32_t nlane = std::min(pf, n_sq - r * pf);
      std::memcpy(u->m_bytes(md + r * gs), src + size_t(r) * pf * sqb,
                  size_t(nlane) * sqb);
    }
  } else {
    for (uint32_t s = 0; s < n_sq; s++)
      std::memcpy(u->m_bytes(md + s * gs), src + size_t(s) * sqb, sqb);
  }
}

static bool conv_buf(uint32_t src_dt, uint32_t dst_dt, uint32_t nelem,
                     const uint8_t* src, std::vector<uint8_t>& dst,
                     uint8_t* ff, uint8_t* xs) {
  dst.assign(packed_nbytes(dst_dt, nelem), 0);
  if (ame_dtype_ident_key(src_dt) == ame_dtype_ident_key(dst_dt)) {
    std::memcpy(dst.data(), src, dst.size());
    return true;
  }
  return ame_host_convert(src_dt, dst_dt, nelem, src, dst.data(), ff, xs);
}

static void broadcast_packed(uint8_t* dst, uint32_t nelem, uint32_t dtype,
                             const uint8_t* elem) {
  const uint32_t bits = ame_parse_dtype(dtype).size_bits;
  for (uint32_t i = 0; i < nelem; i++)
    ame_copy_elem(dst, i, elem, 0, bits);
}

bool ame_params_legal(reg_t nelem, reg_t udsz, reg_t mregs, reg_t accregs, reg_t* n_out) {
  if (nelem == 0 || (nelem & (nelem - 1)) != 0)
    return false;
  reg_t n = 1;
  while (n < 0x10000 && n * n < nelem)
    n <<= 1;
  if (n * n != nelem || n < 2)
    return false;
  if (udsz != 8 && udsz != 16 && udsz != 32 && udsz != 64)
    return false;
  if (mregs != 16 && mregs != 32)
    return false;
  if (accregs != 1 && accregs != 2 && accregs != 4 &&
      accregs != 8 && accregs != 16)
    return false;
  if (n_out)
    *n_out = n;
  return true;
}

reg_t ame_backend_t::acquire(const void* who, reg_t descriptor, bool* newly_granted) {
  if (newly_granted)
    *newly_granted = false;

  const reg_t wait_mode = descriptor & 0x3;
  const reg_t timeout_class = (descriptor >> 2) & 0xf;
  const reg_t reserved = descriptor >> 6;
  if (reserved != 0 || wait_mode > 1)
    return AME_ACQUIRE_RESULT(AME_ACQUIRE_BAD_DESC, false);
  if (wait_mode == 1 && timeout_class > 3 && timeout_class != 0xf)
    return AME_ACQUIRE_RESULT(AME_ACQUIRE_BAD_DESC, false);

  if (owner_ == who)
    return AME_ACQUIRE_RESULT(AME_ACQUIRE_GRANTED, true);
  if (owner_ != nullptr) {
    // Spike's hart scheduler cannot run a peer hart inside this instruction,
    // so BOUNDED_WAIT's implementation-defined interval is zero extra
    // interleaves: busy → TIMEOUT (TRY → BUSY).
    if (wait_mode == 0)
      return AME_ACQUIRE_RESULT(AME_ACQUIRE_BUSY, false);
    return AME_ACQUIRE_RESULT(AME_ACQUIRE_TIMEOUT, false);
  }

  owner_ = who;
  if (newly_granted)
    *newly_granted = true;
  return AME_ACQUIRE_RESULT(AME_ACQUIRE_GRANTED, true);
}

void ame_backend_t::release(const void* who) {
  if (owner_ == who)
    owner_ = nullptr;
}

reg_t ame_unit_t::bytes_per_m_reg() const {
  return ame_nelem * (ame_udsz / 8);
}

reg_t ame_unit_t::bytes_per_acc_reg() const {
  // Widest supported Acc element is Int/Uint128.
  return ame_nelem * 16;
}

void ame_unit_t::allocate_storage() {
  const reg_t m_bytes = num_m_regs * bytes_per_m_reg();
  const reg_t acc_bytes = num_acc_regs * bytes_per_acc_reg();
  m_data.assign(m_bytes, 0);
  md_desc.assign(num_m_regs, 0);
  acc_data.assign(acc_bytes, 0);
  ad_desc.assign(num_acc_regs, 0);
}

bool ame_unit_t::ms_enabled() const {
  return p->get_state()->sstatus->enabled(SSTATUS_AME_MS);
}

void ame_unit_t::set_ms(reg_t status) {
  auto sstatus = p->get_state()->sstatus;
  const reg_t cur = sstatus->read();
  const reg_t next = (cur & ~SSTATUS_AME_MS) | ((status & 0x3) << MSTATUS_AME_MS_SHIFT);
  sstatus->write(next);
}

void ame_unit_t::bind_initial_state() {
  owned = true;
  amestype = 0;
  amefflags = 0;
  amexsat = 0;
  amestatus = 0;
  std::memset(m_data.data(), 0, m_data.size());
  std::memset(acc_data.data(), 0, acc_data.size());
  for (auto& d : md_desc)
    d = 0;
  for (auto& d : ad_desc)
    d = 0;
}

void ame_unit_t::set_unsupported() {
  amestatus |= AMESTATUS_UN;
}

uint32_t ame_unit_t::group_size_for(uint32_t dtype) const {
  return ame_group_size(ame_parse_dtype(dtype).size_bits, ame_udsz);
}

uint32_t ame_unit_t::pack_factor_for(uint32_t dtype) const {
  return ame_pack_factor(ame_parse_dtype(dtype).size_bits, ame_udsz);
}

bool ame_unit_t::m_index_legal(reg_t idx, uint32_t dtype) const {
  const uint32_t gs = group_size_for(dtype);
  if (idx % gs != 0)
    return false;
  if (idx + gs > num_m_regs)
    return false;
  return true;
}

bool ame_unit_t::acc_index_legal(reg_t idx) const {
  return idx < num_acc_regs;
}

bool ame_unit_t::acc_span_legal(reg_t base, uint32_t count) const {
  if (count == 0)
    return false;
  return base + count <= num_acc_regs;
}

void ame_unit_t::clear_m_group(reg_t md, uint32_t group_size) {
  const reg_t bytes = bytes_per_m_reg();
  if (md >= num_m_regs)
    return;
  const reg_t nreg = std::min(reg_t(group_size), num_m_regs - md);
  std::memset(m_data.data() + md * bytes, 0, bytes * nreg);
}

void ame_unit_t::clear_acc(reg_t ad) {
  const reg_t bytes = bytes_per_acc_reg();
  std::memset(acc_data.data() + ad * bytes, 0, bytes);
}

void ame_unit_t::msettyp(reg_t md, uint32_t dtype) {
  if (!ame_dtype_supported_m(dtype, ame_udsz)) {
    set_unsupported();
    return;
  }
  if (!m_index_legal(md, dtype))
    throw trap_illegal_instruction(0);
  const uint32_t gs = group_size_for(dtype);
  md_desc[md] = dtype;
  // Non-base registers in a wide group are not independently typed; drop any
  // leftover descriptor so a later op does not read a stale Md.
  for (uint32_t i = 1; i < gs && md + i < num_m_regs; i++)
    md_desc[md + i] = 0;
  clear_m_group(md, gs);
}

uint32_t ame_unit_t::mgettyp(reg_t ms1) const {
  return md_desc[ms1];
}

void ame_unit_t::asettyp(reg_t ad, uint32_t dtype) {
  if (!ame_dtype_supported_acc(dtype, ame_udsz)) {
    set_unsupported();
    return;
  }
  ad_desc[ad] = dtype;
  clear_acc(ad);
}

uint32_t ame_unit_t::agettyp(reg_t ad) const {
  return ad_desc[ad];
}

reg_t ame_unit_t::acquire(reg_t descriptor) {
  bool newly = false;
  const reg_t result = backend->acquire(this, descriptor, &newly);
  if ((result & 1) == 0)
    return result;
  if (newly) {
    bind_initial_state();
    set_ms(AME_MS_INITIAL);
  } else {
    owned = true;
  }
  return result;
}

void ame_unit_t::release() {
  if (!owned)
    return;

  backend->release(this);
  owned = false;
  amestype = 0;
  amefflags = 0;
  amexsat = 0;
  amestatus = 0;
  std::memset(m_data.data(), 0, m_data.size());
  std::memset(acc_data.data(), 0, acc_data.size());
  for (auto& d : md_desc)
    d = 0;
  for (auto& d : ad_desc)
    d = 0;
  set_ms(AME_MS_INITIAL);
}

void ame_unit_t::set_backend(ame_backend_t* shared) {
  if (owned && backend)
    backend->release(this);
  owned = false;
  backend = shared ? shared : &local_backend;
}

void ame_unit_t::reset() {
  if (!p->extension_enabled(EXT_ZTT))
    return;

  if (owned && backend)
    backend->release(this);
  owned = false;

  const cfg_t& cfg = p->get_cfg();
  ame_nelem = cfg.ame_nelem;
  ame_udsz = cfg.ame_udsz;
  num_m_regs = cfg.ame_mregs;
  num_acc_regs = cfg.ame_accregs;
  if (!ame_params_legal(ame_nelem, ame_udsz, num_m_regs, num_acc_regs, &ame_n)) {
    fprintf(stderr,
            "error: invalid AME parameters nelem=%" PRIu64 " udsz=%" PRIu64
            " mregs=%" PRIu64 " accregs=%" PRIu64 "\n",
            (uint64_t)ame_nelem, (uint64_t)ame_udsz,
            (uint64_t)num_m_regs, (uint64_t)num_acc_regs);
    abort();
  }

  amestype = 0;
  amefflags = 0;
  amexsat = 0;
  amestatus = 0;

  // E5M2's NaN interpretation is a model-wide static choice rather than
  // per-instruction state, so it is installed here from the configuration.
  ame_set_e5m2_nan_ieee(cfg.ame_e5m2_nan_ieee);

  allocate_storage();

  auto state = p->get_state();
  state->add_csr(CSR_AMENLEN, std::make_shared<amenlen_csr_t>(p, this));
  state->add_csr(CSR_AMEUDSZ, std::make_shared<ameudsz_csr_t>(p, this));
  state->add_csr(CSR_AMESTYPE, std::make_shared<amestype_csr_t>(p, this));
  state->add_csr(CSR_AMEOWN, std::make_shared<ameown_csr_t>(p, this));
  state->add_csr(CSR_AMEFFLAGS, std::make_shared<amefflags_csr_t>(p, this));
  state->add_csr(CSR_AMEXSAT, std::make_shared<amexsat_csr_t>(p, this));
  state->add_csr(CSR_AMESTATUS, std::make_shared<amestatus_csr_t>(p, this));
}

uint8_t* ame_unit_t::m_bytes(reg_t idx) {
  return m_data.data() + idx * bytes_per_m_reg();
}

const uint8_t* ame_unit_t::m_bytes(reg_t idx) const {
  return m_data.data() + idx * bytes_per_m_reg();
}

uint8_t* ame_unit_t::acc_bytes(reg_t idx) {
  return acc_data.data() + idx * bytes_per_acc_reg();
}

const uint8_t* ame_unit_t::acc_bytes(reg_t idx) const {
  return acc_data.data() + idx * bytes_per_acc_reg();
}

reg_t ame_unit_t::rm_transfer_bytes(uint32_t dtype) const {
  const ame_dtype_t d = ame_parse_dtype(dtype);
  if (d.size_bits == 0)
    return 0;
  const uint32_t pf = pack_factor_for(dtype);
  return (reg_t(pf) * ame_nelem * d.size_bits + 7) / 8;
}

reg_t ame_unit_t::align_requirement(reg_t elem_bytes) const {
  return elem_bytes == 0 ? 1 : elem_bytes;
}

void ame_unit_t::require_load_align(reg_t addr, reg_t align) const {
  if (align == 0)
    return;
  if (addr % align != 0) {
    const bool gva = p->get_state()->v;
    throw trap_load_address_misaligned(gva, addr, 0, 0);
  }
}

void ame_unit_t::require_store_align(reg_t addr, reg_t align) const {
  if (align == 0)
    return;
  if (addr % align != 0) {
    const bool gva = p->get_state()->v;
    throw trap_store_address_misaligned(gva, addr, 0, 0);
  }
}

void ame_unit_t::load_mem(mmu_t* mmu, reg_t addr, uint8_t* dst, reg_t len) {
  std::vector<uint8_t> tmp(len);
  for (reg_t i = 0; i < len; i++)
    tmp[i] = mmu->load<uint8_t>(addr + i);
  std::memcpy(dst, tmp.data(), len);
}

void ame_unit_t::store_mem(mmu_t* mmu, reg_t addr, const uint8_t* src, reg_t len) {
  mmu->store_ram_spans({{addr, src, size_t(len)}});
}

void ame_unit_t::mzero_m(reg_t md) {
  const uint32_t dtype = md_desc[md];
  if (!ame_dtype_supported_m(dtype, ame_udsz) || dtype == 0) {
    set_unsupported();
    return;
  }
  if (!m_index_legal(md, dtype))
    throw trap_illegal_instruction(0);
  clear_m_group(md, group_size_for(dtype));
}

void ame_unit_t::mzero_acc(reg_t ad) {
  const uint32_t dtype = ad_desc[ad];
  if (!ame_dtype_supported_acc(dtype, ame_udsz) || dtype == 0) {
    set_unsupported();
    return;
  }
  clear_acc(ad);
}

void ame_unit_t::mmov_m_m(reg_t md, reg_t ms) {
  const uint32_t sd = md_desc[ms];
  const uint32_t dd = md_desc[md];
  if (ame_dtype_ident_key(sd) != ame_dtype_ident_key(dd) || sd == 0 || dd == 0) {
    set_unsupported();
    return;
  }
  const uint32_t n_sq = n_sq_max(this, sd, dd);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms, n_sq, sd))
    throw trap_illegal_instruction(0);
  std::vector<uint8_t> tmp;
  read_squares(this, ms, sd, n_sq, tmp);
  write_squares(this, md, dd, n_sq, tmp.data());
}

void ame_unit_t::mmov_m_a(reg_t md, reg_t acc) {
  const uint32_t ad = ad_desc[acc];
  const uint32_t md_t = md_desc[md];
  if (ame_dtype_family_key(ad) != ame_dtype_family_key(md_t) || ad == 0 || md_t == 0) {
    set_unsupported();
    return;
  }
  const uint32_t pf = pack_factor_for(md_t);
  if (pf > num_acc_regs) {
    set_unsupported();
    return;
  }
  if (!m_index_legal(md, md_t) || !acc_span_legal(acc, pf))
    throw trap_illegal_instruction(0);

  const ame_dtype_t d = ame_parse_dtype(md_t);
  const reg_t square_bytes = ame_packed_bytes(uint32_t(ame_nelem), d.size_bits);
  if (pf > 1) {
    for (uint32_t s = 0; s < pf; s++)
      std::memcpy(m_bytes(md) + s * square_bytes, acc_bytes(acc + s), square_bytes);
  } else {
    std::memcpy(m_bytes(md), acc_bytes(acc), square_bytes);
  }
}

void ame_unit_t::mmov_a_m(reg_t acc, reg_t ms) {
  const uint32_t md_t = md_desc[ms];
  const uint32_t ad = ad_desc[acc];
  if (ame_dtype_family_key(ad) != ame_dtype_family_key(md_t) || ad == 0 || md_t == 0) {
    set_unsupported();
    return;
  }
  const uint32_t pf = pack_factor_for(md_t);
  if (pf > num_acc_regs) {
    set_unsupported();
    return;
  }
  if (!m_index_legal(ms, md_t) || !acc_span_legal(acc, pf))
    throw trap_illegal_instruction(0);

  const ame_dtype_t d = ame_parse_dtype(md_t);
  const reg_t square_bytes = ame_packed_bytes(uint32_t(ame_nelem), d.size_bits);
  if (pf > 1) {
    for (uint32_t s = 0; s < pf; s++)
      std::memcpy(acc_bytes(acc + s), m_bytes(ms) + s * square_bytes, square_bytes);
  } else {
    std::memcpy(acc_bytes(acc), m_bytes(ms), square_bytes);
  }
}

void ame_unit_t::mls(reg_t md, reg_t addr, mmu_t* mmu) {
  // mls.1r: unit-dtype row-major image of one physical M register; Md is not read.
  const reg_t eb = (ame_udsz + 7) / 8;
  const reg_t len = bytes_per_m_reg();
  require_load_align(addr, align_requirement(eb));
  load_mem(mmu, addr, m_bytes(md), len);
}

void ame_unit_t::mss(reg_t ms1, reg_t addr, mmu_t* mmu) {
  // mss.1r: unit-dtype row-major image of one physical M register; Md is not read.
  const reg_t eb = (ame_udsz + 7) / 8;
  const reg_t len = bytes_per_m_reg();
  require_store_align(addr, align_requirement(eb));
  store_mem(mmu, addr, m_bytes(ms1), len);
}

bool ame_unit_t::prepare_typed_m(reg_t idx, uint32_t& dtype) {
  dtype = md_desc[idx];
  if (!ame_dtype_supported_m(dtype, ame_udsz) || dtype == 0) {
    set_unsupported();
    return false;
  }
  if (!m_index_legal(idx, dtype))
    throw trap_illegal_instruction(0);
  return true;
}

reg_t ame_unit_t::element_bytes(uint32_t dtype) const {
  return ame_elem_align_bytes(ame_parse_dtype(dtype).size_bits);
}

reg_t ame_unit_t::wrap_seg_addr(reg_t base, reg_t stride, uint32_t seg) const {
  if (p->get_xlen() == 32)
    return uint32_t(uint32_t(base) + uint32_t(seg) * uint32_t(stride));
  return base + reg_t(seg) * stride;
}

void ame_unit_t::load_segments(mmu_t* mmu, reg_t base, reg_t stride, uint32_t nseg,
                               reg_t seg_bytes, uint8_t* dst) {
  // Spec 2.6.3: only the base address is alignment-checked (caller). A later
  // segment fault must leave the destination register unchanged, so stage first.
  std::vector<uint8_t> tmp(size_t(nseg) * size_t(seg_bytes));
  for (uint32_t s = 0; s < nseg; s++)
    load_mem(mmu, wrap_seg_addr(base, stride, s), tmp.data() + s * seg_bytes, seg_bytes);
  std::memcpy(dst, tmp.data(), tmp.size());
}

void ame_unit_t::store_segments(mmu_t* mmu, reg_t base, reg_t stride, uint32_t nseg,
                                reg_t seg_bytes, const uint8_t* src) {
  // Validate the whole instruction, not one independently committed segment.
  std::vector<mmu_t::ram_store_span> spans;
  spans.reserve(nseg);
  for (uint32_t s = 0; s < nseg; s++)
    spans.push_back({wrap_seg_addr(base, stride, s), src + s * seg_bytes,
                     size_t(seg_bytes)});
  mmu->store_ram_spans(spans);
}

void ame_unit_t::mls_rm(reg_t md, reg_t addr, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(md, dtype))
    return;
  const reg_t eb = element_bytes(dtype);
  const reg_t len = rm_transfer_bytes(dtype);
  require_load_align(addr, align_requirement(eb));
  load_mem(mmu, addr, m_bytes(md), len);
}

void ame_unit_t::mss_rm(reg_t ms1, reg_t addr, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(ms1, dtype))
    return;
  const reg_t eb = element_bytes(dtype);
  const reg_t len = rm_transfer_bytes(dtype);
  require_store_align(addr, align_requirement(eb));
  store_mem(mmu, addr, m_bytes(ms1), len);
}

void ame_unit_t::mls_cm(reg_t md, reg_t addr, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(md, dtype))
    return;
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t e_bits = ame_parse_dtype(dtype).size_bits;
  const reg_t eb = element_bytes(dtype);
  const reg_t len = rm_transfer_bytes(dtype);
  require_load_align(addr, align_requirement(eb));
  std::vector<uint8_t> cm(len), rm(len);
  load_mem(mmu, addr, cm.data(), len);
  ame_transpose_squares(ame_n, pf, e_bits, cm.data(), rm.data());
  std::memcpy(m_bytes(md), rm.data(), len);
}

void ame_unit_t::mss_cm(reg_t ms1, reg_t addr, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(ms1, dtype))
    return;
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t e_bits = ame_parse_dtype(dtype).size_bits;
  const reg_t eb = element_bytes(dtype);
  const reg_t len = rm_transfer_bytes(dtype);
  require_store_align(addr, align_requirement(eb));
  std::vector<uint8_t> cm(len);
  ame_transpose_squares(ame_n, pf, e_bits, m_bytes(ms1), cm.data());
  store_mem(mmu, addr, cm.data(), len);
}

void ame_unit_t::mls_st(reg_t md, reg_t addr, reg_t stride, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(md, dtype))
    return;
  const uint32_t n = uint32_t(ame_n);
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t e_bits = ame_parse_dtype(dtype).size_bits;
  require_load_align(addr, align_requirement(element_bytes(dtype)));
  // Spec 2.6.2: N segments, each ceil(P*N*E/8) bytes, at base + i*stride.
  const uint32_t seg_bytes = ame_packed_bytes(pf * n, e_bits);
  const uint32_t rbytes = uint32_t(rm_transfer_bytes(dtype));
  std::vector<uint8_t> st(size_t(n) * seg_bytes), rm(rbytes, 0);
  load_segments(mmu, addr, stride, n, seg_bytes, st.data());
  ame_st_to_reg(n, pf, e_bits, st.data(), rm.data());
  // Same writeback as mls.rm: the .rm image occupies the architectural group
  // starting at md (packed squares share one register; wide types span gs).
  std::memcpy(m_bytes(md), rm.data(), rbytes);
}

void ame_unit_t::mss_st(reg_t ms1, reg_t addr, reg_t stride, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(ms1, dtype))
    return;
  const uint32_t n = uint32_t(ame_n);
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t e_bits = ame_parse_dtype(dtype).size_bits;
  require_store_align(addr, align_requirement(element_bytes(dtype)));
  const uint32_t seg_bytes = ame_packed_bytes(pf * n, e_bits);
  const uint32_t rbytes = uint32_t(rm_transfer_bytes(dtype));
  std::vector<uint8_t> rm(rbytes), st(size_t(n) * seg_bytes, 0);
  std::memcpy(rm.data(), m_bytes(ms1), rbytes);
  ame_reg_to_st(n, pf, e_bits, rm.data(), st.data());
  store_segments(mmu, addr, stride, n, seg_bytes, st.data());
}

void ame_unit_t::mls_tst(reg_t md, reg_t addr, reg_t stride, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(md, dtype))
    return;
  const uint32_t n = uint32_t(ame_n);
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t e_bits = ame_parse_dtype(dtype).size_bits;
  require_load_align(addr, align_requirement(element_bytes(dtype)));
  // Spec 2.6.2: P*N segments, each ceil(N*E/8) bytes.
  const uint32_t nseg = pf * n;
  const uint32_t seg_bytes = ame_packed_bytes(n, e_bits);
  const uint32_t rbytes = uint32_t(rm_transfer_bytes(dtype));
  std::vector<uint8_t> tst(size_t(nseg) * seg_bytes), rm(rbytes, 0);
  load_segments(mmu, addr, stride, nseg, seg_bytes, tst.data());
  ame_tst_to_reg(n, pf, e_bits, tst.data(), rm.data());
  std::memcpy(m_bytes(md), rm.data(), rbytes);
}

void ame_unit_t::mss_tst(reg_t ms1, reg_t addr, reg_t stride, mmu_t* mmu) {
  uint32_t dtype;
  if (!prepare_typed_m(ms1, dtype))
    return;
  const uint32_t n = uint32_t(ame_n);
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t e_bits = ame_parse_dtype(dtype).size_bits;
  require_store_align(addr, align_requirement(element_bytes(dtype)));
  const uint32_t nseg = pf * n;
  const uint32_t seg_bytes = ame_packed_bytes(n, e_bits);
  const uint32_t rbytes = uint32_t(rm_transfer_bytes(dtype));
  std::vector<uint8_t> tst(size_t(nseg) * seg_bytes, 0);
  std::vector<uint8_t> rm(rbytes);
  std::memcpy(rm.data(), m_bytes(ms1), rbytes);
  ame_reg_to_tst(n, pf, e_bits, rm.data(), tst.data());
  store_segments(mmu, addr, stride, nseg, seg_bytes, tst.data());
}

void ame_unit_t::mmul_2d(reg_t acc, reg_t ms1, reg_t ms2, bool accumulate,
                         bool negate, bool transpose_a, bool transpose_b) {
  const uint32_t da = md_desc[ms1];
  const uint32_t db = md_desc[ms2];
  const uint32_t dc = ad_desc[acc];

  if (!ame_matmul_supported(da, db, dc, ame_udsz)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, da, db);
  if (!m_span_legal(ms1, n_sq, da) || !m_span_legal(ms2, n_sq, db) ||
      !acc_index_legal(acc))
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> a_src, b_src;
  read_squares(this, ms1, da, n_sq, a_src);
  read_squares(this, ms2, db, n_sq, b_src);

  uint8_t ff = 0, xs = 0;
  if (!ame_host_matmul(ame_n, n_sq, da, a_src.data(), db, b_src.data(),
                       dc, acc_bytes(acc), accumulate, negate, transpose_a, transpose_b,
                       &ff, &xs)) {
    set_unsupported();
    return;
  }
  amefflags |= ff;
  amexsat |= xs;
}

bool ame_unit_t::m_span_legal(reg_t base, uint32_t n_sq, uint32_t dtype) const {
  const uint32_t pf = pack_factor_for(dtype);
  const uint32_t gs = group_size_for(dtype);
  if (gs == 0 || n_sq == 0)
    return false;
  const uint32_t groups = (pf > 1) ? (n_sq + pf - 1) / pf : n_sq;
  const uint32_t span = groups * gs;
  if (span == 0)
    return false;
  if (base % gs != 0)
    return false;
  if (base >= num_m_regs)
    return false;
  return span <= num_m_regs - base;
}

void ame_unit_t::ew_binop(ame_ew_op_t op, reg_t md, reg_t ms1, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t db = md_desc[ms2];

  if (dd == 0 || da == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz)) {
    set_unsupported();
    return;
  }

  const bool bitwise =
      op == ame_ew_op_t::AND || op == ame_ew_op_t::ANDNOT ||
      op == ame_ew_op_t::OR || op == ame_ew_op_t::ORNOT ||
      op == ame_ew_op_t::XOR;
  const bool minmax = op == ame_ew_op_t::MIN || op == ame_ew_op_t::MAX;
  if ((bitwise || minmax) &&
      (ame_dtype_ident_key(dd) != ame_dtype_ident_key(da) ||
       ame_dtype_ident_key(dd) != ame_dtype_ident_key(db))) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> a_buf, b_buf, a_conv, b_conv, out;
  read_squares(this, ms1, da, n_sq, a_buf);
  read_squares(this, ms2, db, n_sq, b_buf);
  uint8_t ff = 0, xs = 0;
  if (ame_dtype_ident_key(da) != ame_dtype_ident_key(dd)) {
    a_conv.assign(packed_nbytes(dd, nelem), 0);
    if (!ame_host_convert(da, dd, nelem, a_buf.data(), a_conv.data(), &ff, &xs)) {
      set_unsupported();
      return;
    }
  } else {
    a_conv.swap(a_buf);
  }
  if (ame_dtype_ident_key(db) != ame_dtype_ident_key(dd)) {
    b_conv.assign(packed_nbytes(dd, nelem), 0);
    if (!ame_host_convert(db, dd, nelem, b_buf.data(), b_conv.data(), &ff, &xs)) {
      set_unsupported();
      return;
    }
  } else {
    b_conv.swap(b_buf);
  }

  out.assign(packed_nbytes(dd, nelem), 0);
  if (ame_ew_reads_dest(op))
    read_squares(this, md, dd, n_sq, out);
  if (!ame_host_ew(op, dd, nelem, a_conv.data(), b_conv.data(), out.data(), &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::ew_unop(ame_ew_op_t op, reg_t md, reg_t ms1) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];

  if (dd == 0 || da == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz)) {
    set_unsupported();
    return;
  }

  // An integer source of exp2/log2 needs no explicit rejection here: the host
  // float kernels (ame_ops.cc) report the tuple as not applicable, and the
  // failure path below turns that into amestatus.UN as spec 6.1.3 (:56-62)
  // requires.  Spelling the check out for only those two ops would suggest the
  // rest of the family is not covered the same way.

  const uint32_t n_sq = n_sq_max(this, dd, da);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> a_buf, a_conv, out;
  read_squares(this, ms1, da, n_sq, a_buf);
  uint8_t ff = 0, xs = 0;
  if (ame_dtype_ident_key(da) != ame_dtype_ident_key(dd)) {
    a_conv.assign(packed_nbytes(dd, nelem), 0);
    if (!ame_host_convert(da, dd, nelem, a_buf.data(), a_conv.data(), &ff, &xs)) {
      set_unsupported();
      return;
    }
  } else {
    a_conv.swap(a_buf);
  }
  out.assign(packed_nbytes(dd, nelem), 0);
  if (!ame_host_ew(op, dd, nelem, a_conv.data(), nullptr, out.data(), &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::ew_ldexp(reg_t md, reg_t ms1, reg_t ms2, bool accumulate, bool negate_exp) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t de = md_desc[ms2];
  if (dd == 0 || da == 0 || de == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(de, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(da) ||
      ame_parse_dtype(de).is_fp) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da, de);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, de))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> a_buf, e_buf, out;
  read_squares(this, ms1, da, n_sq, a_buf);
  read_squares(this, ms2, de, n_sq, e_buf);
  if (accumulate)
    read_squares(this, md, dd, n_sq, out);
  else
    out.assign(packed_nbytes(dd, nelem), 0);
  uint8_t ff = 0, xs = 0;
  if (!ame_host_ldexp(da, de, nelem, a_buf.data(), e_buf.data(), out.data(),
                      accumulate, negate_exp, &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::ew_ldexp_x(reg_t md, reg_t xs1_val, reg_t ms2, bool accumulate) {
  const uint32_t dd = md_desc[md];
  const uint32_t db = md_desc[ms2];
  if (dd == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(db)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const int64_t exp = int32_t(uint32_t(xs1_val));
  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> b_buf, out;
  read_squares(this, ms2, db, n_sq, b_buf);
  if (accumulate)
    read_squares(this, md, dd, n_sq, out);
  else
    out.assign(packed_nbytes(dd, nelem), 0);
  uint8_t ff = 0, xs = 0;
  if (!ame_host_ldexp_x(db, nelem, b_buf.data(), exp, out.data(), accumulate, &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::ew_log2bias(reg_t md, reg_t ms1, reg_t ms2, bool bias_minus_log) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t db = md_desc[ms2];
  if (dd == 0 || da == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz)) {
    set_unsupported();
    return;
  }

  // Spec 6.2.2 (instruction_semantics.adoc:182-188): "For the mlog2sub.ew and
  // msublog2.ew families, including their .ew.x forms, the operand of log2, and
  // the destination, are standard floating-point datatypes; the bias is
  // governed by the exact tuple's conversion and subtraction capabilities."
  // ms1 is the log2 operand here, so an integer ms1 or md is an unsupported
  // tuple (spec 6.1.3 :56-62: UN set, md unchanged).  The ms2 bias may be a
  // standard integer square: it is converted to the destination format below,
  // which is exactly the conversion-and-subtraction capability the tuple has.
  // A 128-bit integer bias has no standard single conversion to a
  // floating-point datatype, so that tuple stays unsupported.
  const ame_dtype_t db_dt = ame_parse_dtype(db);
  const bool int_bias = ame_is_int_family(db);
  if (!ame_parse_dtype(da).is_fp || !ame_parse_dtype(dd).is_fp ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(da) ||
      (int_bias ? db_dt.size_bits > 64
                : ame_dtype_family_key(dd) != ame_dtype_family_key(db))) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> a_buf, b_buf, out;
  read_squares(this, ms1, da, n_sq, a_buf);
  read_squares(this, ms2, db, n_sq, b_buf);
  uint8_t ff = 0;
  if (int_bias) {
    // Convert the bias with the destination tuple's rounding mode, so an
    // inexact integer-to-float conversion accrues its flags like any other
    // conversion of this tuple.
    std::vector<uint8_t> b_conv(packed_nbytes(dd, nelem), 0);
    if (!ame_host_convert(db, dd, nelem, b_buf.data(), b_conv.data(), &ff, nullptr)) {
      set_unsupported();
      return;
    }
    b_buf.swap(b_conv);
  }
  out.assign(packed_nbytes(dd, nelem), 0);
  if (!ame_host_log2bias(dd, nelem, a_buf.data(), b_buf.data(), out.data(),
                         bias_minus_log, &ff)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
}

void ame_unit_t::ew_log2bias_x(reg_t md, reg_t xs1_val, reg_t ms2, bool bias_minus_log) {
  const uint32_t dd = md_desc[md];
  const uint32_t db = md_desc[ms2];
  if (dd == 0 || db == 0 || amestype == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(db)) {
    set_unsupported();
    return;
  }

  // Spec 6.2.2 (instruction_semantics.adoc:181-187): the .ew.x forms follow
  // mlog2sub/msublog2, so ms2 is the log2 operand and must be a standard
  // floating-point datatype, as must the destination.  The X-register bias may
  // be integer through the amestype conversion; only an integer ms2/md is an
  // unsupported tuple (spec 6.1.3 :56-62: UN set, md unchanged).
  if (!ame_parse_dtype(db).is_fp || !ame_parse_dtype(dd).is_fp) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> scalar(16, 0);
  if (!ame_convert_scalar(amestype, xs1_val, p->get_xlen(), dd, scalar.data())) {
    set_unsupported();
    return;
  }

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> bias(packed_nbytes(dd, nelem));
  broadcast_packed(bias.data(), nelem, dd, scalar.data());

  std::vector<uint8_t> b_buf, out;
  read_squares(this, ms2, db, n_sq, b_buf);
  out.assign(packed_nbytes(dd, nelem), 0);
  uint8_t ff = 0;
  if (!ame_host_log2bias(dd, nelem, b_buf.data(), bias.data(), out.data(),
                         bias_minus_log, &ff)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
}

void ame_unit_t::ew_binop_x(ame_ew_op_t op, reg_t md, reg_t xs1_val, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t db = md_desc[ms2];

  if (dd == 0 || db == 0 || amestype == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz)) {
    set_unsupported();
    return;
  }
  const bool bitwise =
      op == ame_ew_op_t::AND || op == ame_ew_op_t::ANDNOT ||
      op == ame_ew_op_t::OR || op == ame_ew_op_t::ORNOT ||
      op == ame_ew_op_t::XOR;
  const bool minmax = op == ame_ew_op_t::MIN || op == ame_ew_op_t::MAX;
  if ((bitwise || minmax) &&
      (ame_dtype_ident_key(dd) != ame_dtype_ident_key(db))) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> scalar(16, 0);
  // Spec 6.2.4: convert the amestype scalar into Md[ms2], then the operation
  // result into Md[md].
  if (!ame_convert_scalar(amestype, xs1_val, p->get_xlen(), db, scalar.data())) {
    set_unsupported();
    return;
  }

  std::vector<uint8_t> a_broadcast(packed_nbytes(db, nelem));
  broadcast_packed(a_broadcast.data(), nelem, db, scalar.data());

  std::vector<uint8_t> b_buf, b_conv, a_conv;
  read_squares(this, ms2, db, n_sq, b_buf);
  uint8_t ff = 0, xs = 0;
  if (ame_dtype_ident_key(db) != ame_dtype_ident_key(dd)) {
    b_conv.assign(packed_nbytes(dd, nelem), 0);
    a_conv.assign(packed_nbytes(dd, nelem), 0);
    if (!ame_host_convert(db, dd, nelem, b_buf.data(), b_conv.data(), &ff, &xs) ||
        !ame_host_convert(db, dd, nelem, a_broadcast.data(), a_conv.data(), &ff, &xs)) {
      set_unsupported();
      return;
    }
  } else {
    b_conv.swap(b_buf);
    a_conv.swap(a_broadcast);
  }

  std::vector<uint8_t> out(packed_nbytes(dd, nelem));
  if (ame_ew_reads_dest(op))
    read_squares(this, md, dd, n_sq, out);
  bool ok;
  // Scalar (a_conv) is lhs, ms2 elements (b_conv) are rhs; ame_host_ew
  // computes SUB/HDIFF as rhs - lhs per the manual.
  ok = ame_host_ew(op, dd, nelem, a_conv.data(), b_conv.data(), out.data(), &ff, &xs);
  if (!ok) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::ew_cmp(bool lt, reg_t md, reg_t ms1, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t db = md_desc[ms2];
  const ame_dtype_t dtd = ame_parse_dtype(dd);

  if (dd == 0 || da == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      ame_dtype_family_key(da) != ame_dtype_family_key(db) ||
      dtd.is_fp) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> a_buf, b_buf, out;
  read_squares(this, ms1, da, n_sq, a_buf);
  read_squares(this, ms2, db, n_sq, b_buf);
  out.assign(packed_nbytes(dd, nelem), 0);
  uint8_t ff = 0;
  if (!ame_host_ew_cmp(lt, da, dd, nelem, a_buf.data(), b_buf.data(), out.data(), &ff)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
}

void ame_unit_t::ew_cmp_x(bool lt, reg_t md, reg_t xs1_val, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t db = md_desc[ms2];
  const ame_dtype_t dtd = ame_parse_dtype(dd);

  if (dd == 0 || db == 0 || amestype == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      dtd.is_fp) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> scalar(16, 0);
  if (!ame_convert_scalar(amestype, xs1_val, p->get_xlen(), db, scalar.data())) {
    set_unsupported();
    return;
  }

  std::vector<uint8_t> a_broadcast(packed_nbytes(db, nelem));
  broadcast_packed(a_broadcast.data(), nelem, db, scalar.data());

  std::vector<uint8_t> b_buf, out;
  read_squares(this, ms2, db, n_sq, b_buf);
  out.assign(packed_nbytes(dd, nelem), 0);
  uint8_t ff = 0;
  if (!ame_host_ew_cmp(lt, db, dd, nelem, a_broadcast.data(), b_buf.data(), out.data(), &ff)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
}

void ame_unit_t::ew_sel(bool lt, bool cmov, reg_t md, reg_t ms1, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t db = md_desc[ms2];

  if (dd == 0 || da == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      ame_dtype_ident_key(dd) != ame_dtype_ident_key(db)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> pred, src, out;
  read_squares(this, ms1, da, n_sq, pred);
  read_squares(this, ms2, db, n_sq, src);
  if (cmov)
    read_squares(this, md, dd, n_sq, out);
  else
    out.assign(packed_nbytes(dd, nelem), 0);
  if (!ame_host_ew_sel(lt, cmov, da, dd, nelem, pred.data(), src.data(), out.data())) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::ew_shift(ame_shift_op_t op, reg_t md, reg_t ms1, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t db = md_desc[ms2];
  const ame_dtype_t dtd = ame_parse_dtype(dd);

  if (dd == 0 || da == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(da) ||
      dtd.is_fp || (op == ame_shift_op_t::SRA && !dtd.is_signed)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> a_buf, b_buf, out;
  read_squares(this, ms1, da, n_sq, a_buf);
  read_squares(this, ms2, db, n_sq, b_buf);
  out.assign(packed_nbytes(dd, nelem), 0);
  if (!ame_host_ew_shift(op, da, db, nelem, a_buf.data(), b_buf.data(), out.data())) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::ew_shift_x(ame_shift_op_t op, reg_t md, reg_t xs1_val, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t db = md_desc[ms2];
  const ame_dtype_t dtd = ame_parse_dtype(dd);

  if (dd == 0 || db == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(db, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(db) ||
      dtd.is_fp || (op == ame_shift_op_t::SRA && !dtd.is_signed)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, db);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms2, n_sq, db))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> b_buf, out;
  read_squares(this, ms2, db, n_sq, b_buf);
  out.assign(packed_nbytes(dd, nelem), 0);
  if (!ame_host_ew_shift_x(op, db, xs1_val, nelem, b_buf.data(), out.data())) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::fold_reduce(ame_fold_op_t op, bool along_row, reg_t md, reg_t ms1) {
  const uint32_t dd = md_desc[md];
  const uint32_t ds = md_desc[ms1];
  if (dd == 0 || ds == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(ds, ame_udsz)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, ds);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, ds))
    throw trap_illegal_instruction(0);

  const uint32_t n = uint32_t(ame_n);
  const uint32_t sq = n * n;
  std::vector<uint8_t> src, converted, out;
  read_squares(this, ms1, ds, n_sq, src);
  const uint32_t src_sq = packed_nbytes(ds, sq);
  const uint32_t dst_sq = packed_nbytes(dd, sq);
  out.assign(size_t(n_sq) * dst_sq, 0);
  uint8_t ff = 0, xs = 0;
  for (uint32_t s = 0; s < n_sq; s++) {
    if (!conv_buf(ds, dd, sq, src.data() + s * src_sq, converted, &ff, &xs)) {
      set_unsupported();
      return;
    }
    if (!ame_host_reduce(op, along_row, dd, n,
                         converted.data(), out.data() + s * dst_sq, &ff, &xs)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::fold_prefix(ame_fold_op_t op, bool along_row, reg_t md, reg_t ms1) {
  const uint32_t dd = md_desc[md];
  const uint32_t ds = md_desc[ms1];
  if (dd == 0 || ds == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(ds, ame_udsz)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, ds);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, ds))
    throw trap_illegal_instruction(0);

  const uint32_t n = uint32_t(ame_n);
  const uint32_t sq = n * n;
  std::vector<uint8_t> src, converted, out;
  read_squares(this, ms1, ds, n_sq, src);
  const uint32_t src_sq = packed_nbytes(ds, sq);
  const uint32_t dst_sq = packed_nbytes(dd, sq);
  out.assign(size_t(n_sq) * dst_sq, 0);
  uint8_t ff = 0, xs = 0;
  for (uint32_t s = 0; s < n_sq; s++) {
    if (!conv_buf(ds, dd, sq, src.data() + s * src_sq, converted, &ff, &xs)) {
      set_unsupported();
      return;
    }
    if (!ame_host_prefix(op, along_row, dd, n,
                         converted.data(), out.data() + s * dst_sq, &ff, &xs)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::write_ids(bool row_ids, reg_t md) {
  const uint32_t dd = md_desc[md];
  if (dd == 0 || !ame_dtype_supported_m(dd, ame_udsz)) {
    set_unsupported();
    return;
  }
  const uint32_t n_sq = n_sq_max(this, dd);
  if (!m_span_legal(md, n_sq, dd))
    throw trap_illegal_instruction(0);

  const uint32_t sqb = packed_nbytes(dd, uint32_t(ame_nelem));
  std::vector<uint8_t> out(size_t(n_sq) * sqb, 0);
  for (uint32_t s = 0; s < n_sq; s++) {
    if (!ame_host_write_ids(row_ids, dd, uint32_t(ame_n), out.data() + s * sqb)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::bcast_x(bool along_row, reg_t md, reg_t ms1, reg_t xs1_val) {
  const uint32_t dd = md_desc[md];
  const uint32_t ds = md_desc[ms1];
  if (dd == 0 || ds == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(ds, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(ds)) {
    set_unsupported();
    return;
  }
  const uint32_t n_sq = n_sq_max(this, dd, ds);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, ds))
    throw trap_illegal_instruction(0);

  const uint32_t idx = uint32_t(xs1_val & 0xffffffffu);
  if (idx >= ame_n)
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> src, out;
  read_squares(this, ms1, ds, n_sq, src);
  const uint32_t sqb = packed_nbytes(dd, uint32_t(ame_nelem));
  out.assign(size_t(n_sq) * sqb, 0);
  for (uint32_t sq = 0; sq < n_sq; sq++) {
    if (!ame_host_bcast(along_row, dd, uint32_t(ame_n),
                        src.data() + sq * sqb, out.data() + sq * sqb, idx)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::shift_x(bool along_row, reg_t md, reg_t ms1, reg_t xs1_val) {
  const uint32_t dd = md_desc[md];
  const uint32_t ds = md_desc[ms1];
  if (dd == 0 || ds == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(ds, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(ds)) {
    set_unsupported();
    return;
  }
  const uint32_t n_sq = n_sq_max(this, dd, ds);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, ds))
    throw trap_illegal_instruction(0);

  const int32_t offset = int32_t(xs1_val & 0xffffffffu);
  std::vector<uint8_t> src, out;
  read_squares(this, ms1, ds, n_sq, src);
  const uint32_t sqb = packed_nbytes(dd, uint32_t(ame_nelem));
  out.assign(size_t(n_sq) * sqb, 0);
  for (uint32_t sq = 0; sq < n_sq; sq++) {
    if (!ame_host_shift(along_row, dd, uint32_t(ame_n),
                        src.data() + sq * sqb, out.data() + sq * sqb, offset)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::gather(bool along_row, reg_t md, reg_t ms1, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t di = md_desc[ms2];
  if (dd == 0 || da == 0 || di == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(di, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(da) ||
      ame_parse_dtype(di).is_fp) {
    set_unsupported();
    return;
  }
  // Spec 6.25: md, ms1, and ms2 each contain exactly one logical square,
  // so the operand spans are validated per descriptor with N_sq = 1. The
  // packed index form (e.g. U8 index at udsz=32) is one register and stays
  // in-file even at a high base register; folding its pack factor into
  // N_sq inflated the data span and trapped legal setups (m8 + m15 with
  // UINT128 data on a 16-register file).
  const uint32_t n_sq = 1;
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, di))
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> src, idx, out;
  read_squares(this, ms1, da, n_sq, src);
  read_squares(this, ms2, di, n_sq, idx);
  const uint32_t sqb = packed_nbytes(dd, uint32_t(ame_nelem));
  const uint32_t ib = packed_nbytes(di, uint32_t(ame_nelem));
  out.assign(size_t(n_sq) * sqb, 0);
  for (uint32_t s = 0; s < n_sq; s++) {
    if (!ame_host_gather(along_row, dd, di, uint32_t(ame_n),
                         src.data() + s * sqb, idx.data() + s * ib,
                         out.data() + s * sqb)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::zip2(bool along_row, bool unzip, reg_t ms1, reg_t ms2) {
  if (ms1 == ms2)
    throw trap_illegal_instruction(0);

  const uint32_t d1 = md_desc[ms1];
  const uint32_t d2 = md_desc[ms2];
  if (d1 == 0 || d2 == 0 ||
      !ame_dtype_supported_m(d1, ame_udsz) ||
      !ame_dtype_supported_m(d2, ame_udsz) ||
      ame_dtype_family_key(d1) != ame_dtype_family_key(d2) ||
      pack_factor_for(d1) != pack_factor_for(d2)) {
    set_unsupported();
    return;
  }
  const uint32_t n_sq = n_sq_max(this, d1, d2);
  if (!m_span_legal(ms1, n_sq, d1) || !m_span_legal(ms2, n_sq, d2))
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> a, b, out1, out2;
  read_squares(this, ms1, d1, n_sq, a);
  read_squares(this, ms2, d2, n_sq, b);
  const uint32_t sqb = packed_nbytes(d1, uint32_t(ame_nelem));
  out1.assign(size_t(n_sq) * sqb, 0);
  out2.assign(size_t(n_sq) * sqb, 0);
  for (uint32_t s = 0; s < n_sq; s++) {
    if (!ame_host_zip2(along_row, unzip, d1, uint32_t(ame_n),
                       a.data() + s * sqb, b.data() + s * sqb,
                       out1.data() + s * sqb, out2.data() + s * sqb)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, ms1, d1, n_sq, out1.data());
  write_squares(this, ms2, d2, n_sq, out2.data());
}

void ame_unit_t::scatter(bool along_row, ame_scatter_op_t op,
                         reg_t md, reg_t ms1, reg_t ms2) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t di = md_desc[ms2];
  if (dd == 0 || da == 0 || di == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz) ||
      !ame_dtype_supported_m(di, ame_udsz) ||
      ame_dtype_family_key(dd) != ame_dtype_family_key(da) ||
      ame_parse_dtype(di).is_fp) {
    set_unsupported();
    return;
  }
  const uint32_t n_sq = n_sq_max(this, dd, da, di);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da) ||
      !m_span_legal(ms2, n_sq, di))
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> src, idx, dest;
  read_squares(this, ms1, da, n_sq, src);
  read_squares(this, ms2, di, n_sq, idx);
  read_squares(this, md, dd, n_sq, dest);
  const uint32_t sqb = packed_nbytes(dd, uint32_t(ame_nelem));
  const uint32_t ib = packed_nbytes(di, uint32_t(ame_nelem));
  uint8_t ff = 0, xs = 0;
  for (uint32_t s = 0; s < n_sq; s++) {
    if (!ame_host_scatter(along_row, op, dd, di, uint32_t(ame_n),
                          src.data() + s * sqb, idx.data() + s * ib,
                          dest.data() + s * sqb, &ff, &xs)) {
      set_unsupported();
      return;
    }
  }
  write_squares(this, md, dd, n_sq, dest.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::mconv(reg_t md, reg_t ms1) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  if (dd == 0 || da == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd, da);
  if (!m_span_legal(md, n_sq, dd) || !m_span_legal(ms1, n_sq, da))
    throw trap_illegal_instruction(0);

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> src, out;
  read_squares(this, ms1, da, n_sq, src);
  uint8_t ff = 0, xs = 0;
  if (!conv_buf(da, dd, nelem, src.data(), out, &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, n_sq, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::mpack_x(reg_t md, reg_t ms1, reg_t xs1_val) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t pf = pack_factor_for(dd);
  if (dd == 0 || da == 0 || pf <= 1 || pack_factor_for(da) != 1 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz)) {
    set_unsupported();
    return;
  }
  if (!m_span_legal(md, pf, dd) || !m_span_legal(ms1, 1, da))
    throw trap_illegal_instruction(0);

  const uint32_t slot = uint32_t(xs1_val) & (pf - 1);
  const uint32_t sq = uint32_t(ame_nelem);
  const uint32_t slot_bytes = packed_nbytes(dd, sq);
  std::vector<uint8_t> out, src;
  read_squares(this, md, dd, pf, out);
  read_squares(this, ms1, da, 1, src);
  uint8_t ff = 0, xs = 0;
  if (!ame_host_convert(da, dd, sq, src.data(),
                        out.data() + slot * slot_bytes, &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, pf, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::munpack_x(reg_t md, reg_t ms1, reg_t xs1_val) {
  const uint32_t dd = md_desc[md];
  const uint32_t da = md_desc[ms1];
  const uint32_t pf = pack_factor_for(da);
  if (dd == 0 || da == 0 || pf <= 1 || pack_factor_for(dd) != 1 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(da, ame_udsz)) {
    set_unsupported();
    return;
  }
  if (!m_span_legal(md, 1, dd) || !m_span_legal(ms1, pf, da))
    throw trap_illegal_instruction(0);

  const uint32_t slot = uint32_t(xs1_val) & (pf - 1);
  const uint32_t sq = uint32_t(ame_nelem);
  const uint32_t slot_bytes = packed_nbytes(da, sq);
  std::vector<uint8_t> src, out;
  read_squares(this, ms1, da, pf, src);
  uint8_t ff = 0, xs = 0;
  if (!conv_buf(da, dd, sq, src.data() + slot * slot_bytes, out, &ff, &xs)) {
    set_unsupported();
    return;
  }
  write_squares(this, md, dd, 1, out.data());
  amefflags |= ff;
  amexsat |= xs;
}

void ame_unit_t::mbcast_m_x(reg_t md, reg_t xs1_val, reg_t xtyp_val) {
  const uint32_t dd = md_desc[md];
  const uint32_t src_dt = uint32_t(xtyp_val);
  if (dd == 0 || src_dt == 0 ||
      !ame_dtype_supported_m(dd, ame_udsz) ||
      !ame_dtype_supported_m(src_dt, ame_udsz)) {
    set_unsupported();
    return;
  }
  // Spec 6.31 / Sail ame_broadcast_typed_x: only the identity tuple is
  // advertised for typed X ingress — the source descriptor must equal the
  // destination descriptor. Cross-width/narrowing conversions are
  // unsupported (UN), and the destination is left untouched.
  if (ame_dtype_ident_key(src_dt) != ame_dtype_ident_key(dd)) {
    set_unsupported();
    return;
  }

  const uint32_t n_sq = n_sq_max(this, dd);
  if (!m_span_legal(md, n_sq, dd))
    throw trap_illegal_instruction(0);

  std::vector<uint8_t> elem(16, 0);
  if (!ame_convert_scalar(src_dt, xs1_val, p->get_xlen(), dd, elem.data())) {
    set_unsupported();
    return;
  }

  const uint32_t nelem = n_sq * uint32_t(ame_nelem);
  std::vector<uint8_t> out(packed_nbytes(dd, nelem), 0);
  broadcast_packed(out.data(), nelem, dd, elem.data());
  write_squares(this, md, dd, n_sq, out.data());
}

void ame_unit_t::mmove_m_x(unsigned elem_bits, reg_t md, reg_t data_val, reg_t pos_val) {
  if (elem_bits == 64 && p->get_xlen() < 64)
    throw trap_illegal_instruction(0);
  if (!ame_mmove_legal(uint32_t(ame_n), uint32_t(ame_udsz), elem_bits))
    throw trap_illegal_instruction(0);
  ame_host_mmove_store(uint32_t(ame_n), uint32_t(ame_udsz), elem_bits,
                       m_bytes(md), uint32_t(pos_val), uint64_t(data_val));
}

reg_t ame_unit_t::mmove_x_m(unsigned elem_bits, reg_t ms, reg_t pos_val) {
  if (elem_bits == 64 && p->get_xlen() < 64)
    throw trap_illegal_instruction(0);
  if (!ame_mmove_legal(uint32_t(ame_n), uint32_t(ame_udsz), elem_bits))
    throw trap_illegal_instruction(0);
  return ame_host_mmove_load(uint32_t(ame_n), uint32_t(ame_udsz), elem_bits,
                             m_bytes(ms), uint32_t(pos_val));
}

namespace {

void dump_matrix(std::ostream& o, const uint8_t* data, uint32_t n, uint32_t elem_bits) {
  const uint32_t eb = elem_bits / 8;
  const uint32_t hexw = eb * 2;
  for (uint32_t r = 0; r < n; r++) {
    o << "  ";
    for (uint32_t c = 0; c < n; c++) {
      uint64_t u = 0;
      const uint8_t* src = data + (size_t(r) * n + c) * eb;
      for (uint32_t i = 0; i < eb; i++)
        u |= uint64_t(src[i]) << (8 * i);
      o << std::hex << std::setfill('0') << std::setw(hexw) << u;
      if (c + 1 != n)
        o << ' ';
    }
    o << std::dec << std::endl;
  }
}

} // namespace

void ame_unit_t::dump_summary(std::ostream& o) const {
  o << "AME N=" << ame_n << " NELEM=" << ame_nelem
    << " UDSZ=" << ame_udsz
    << " MREGS=" << num_m_regs
    << " ACCREGS=" << num_acc_regs
    << " owned=" << (owned ? 1 : 0)
    << " fflags=0x" << std::hex << unsigned(amefflags)
    << " xsat=" << std::dec << unsigned(amexsat)
    << " status=0x" << std::hex << unsigned(amestatus)
    << std::dec << std::endl;
}

void ame_unit_t::dump_m(std::ostream& o, int idx) const {
  if (idx < 0 || idx >= int(num_m_regs))
    return;
  o << "M" << idx << " md=0x" << std::hex << std::setfill('0') << std::setw(8)
    << md_desc[idx] << std::dec << std::endl;
  dump_matrix(o, m_bytes(idx), uint32_t(ame_n), uint32_t(ame_udsz));
}

void ame_unit_t::dump_acc(std::ostream& o, int idx) const {
  if (idx < 0 || idx >= int(num_acc_regs))
    return;
  o << "A" << idx << " ad=0x" << std::hex << std::setfill('0') << std::setw(8)
    << ad_desc[idx] << std::dec << std::endl;
  dump_matrix(o, acc_bytes(idx), uint32_t(ame_n), 64);
}
