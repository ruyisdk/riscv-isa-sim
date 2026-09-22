require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs1() < P.AMU.num_m_regs);
P.AMU.fold_prefix(ame_fold_op_t::MAX, true, insn.rd(), insn.rs1());
dirty_ame_state;
