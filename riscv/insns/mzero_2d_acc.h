require_ame_owned_insn;
require(P.AMU.acc_index_legal(insn.rd()));
P.AMU.mzero_acc(insn.rd());
dirty_ame_state;
