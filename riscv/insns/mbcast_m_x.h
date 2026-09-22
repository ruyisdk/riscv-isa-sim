require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
P.AMU.mbcast_m_x(insn.rd(), RS1, RS2);
dirty_ame_state;
