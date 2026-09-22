require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
P.AMU.mls_st(insn.rd(), RS1, RS2, &MMU);
dirty_ame_state;
