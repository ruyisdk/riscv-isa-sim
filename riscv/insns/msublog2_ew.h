require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs1() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
P.AMU.ew_log2bias(insn.rd(), insn.rs1(), insn.rs2(), true);
dirty_ame_state;
