require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs1() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
P.AMU.ew_ldexp(insn.rd(), insn.rs1(), insn.rs2(), true, false);
dirty_ame_state;
