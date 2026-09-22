require_ame_owned_insn;
require(insn.rs1() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
require(insn.rs1() != insn.rs2());
P.AMU.zip2(false, true, insn.rs1(), insn.rs2());
dirty_ame_state;
