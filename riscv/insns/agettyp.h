require_ame_owned_insn;
require(P.AMU.acc_index_legal(insn.rs1()));
WRITE_RD(P.AMU.agettyp(insn.rs1()));
