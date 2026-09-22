require_ame_owned_insn;
require(insn.rd() < P.AMU.num_acc_regs);
require(insn.rs1() < P.AMU.num_m_regs);
P.AMU.mmov_a_m(insn.rd(), insn.rs1());
dirty_ame_state;
