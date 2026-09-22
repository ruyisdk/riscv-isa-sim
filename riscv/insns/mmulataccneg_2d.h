require_ame_owned_insn;
require(insn.rd() < P.AMU.num_acc_regs);
require(insn.rs1() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
P.AMU.mmul_2d(insn.rd(), insn.rs1(), insn.rs2(), true, true, true, false);
dirty_ame_state;
