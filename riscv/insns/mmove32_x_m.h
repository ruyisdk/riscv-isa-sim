require_ame_owned_insn;
require(insn.rs1() < P.AMU.num_m_regs);
WRITE_RD(P.AMU.mmove_x_m(32, insn.rs1(), RS2));
