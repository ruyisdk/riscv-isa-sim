require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
P.AMU.mls_rm(insn.rd(), RS1, &MMU);
dirty_ame_state;
