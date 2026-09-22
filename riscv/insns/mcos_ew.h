require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs1() < P.AMU.num_m_regs);
P.AMU.ew_unop(ame_ew_op_t::COS, insn.rd(), insn.rs1());
dirty_ame_state;
