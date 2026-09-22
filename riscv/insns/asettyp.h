require_ame_owned_insn;
require(P.AMU.acc_index_legal(insn.rd()));
P.AMU.asettyp(insn.rd(), RS1);
dirty_ame_state;
