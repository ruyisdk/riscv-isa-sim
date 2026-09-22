require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
P.AMU.shift_x(true, insn.rd(), insn.rs2(), RS1);
dirty_ame_state;
