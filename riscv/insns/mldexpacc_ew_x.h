require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
P.AMU.ew_ldexp_x(insn.rd(), RS1, insn.rs2(), true);
dirty_ame_state;
