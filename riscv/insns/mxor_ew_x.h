require_ame_owned_insn;
require(insn.rd() < P.AMU.num_m_regs);
require(insn.rs2() < P.AMU.num_m_regs);
P.AMU.ew_binop_x(ame_ew_op_t::XOR, insn.rd(), RS1, insn.rs2());
dirty_ame_state;
