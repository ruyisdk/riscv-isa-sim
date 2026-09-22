require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
P.AMU.write_ids(false, insn.rd());
dirty_ame_state;
