require_ame_owned_insn;
// Spec 2.4.4 / 6.119: unsupported encodings set UN and do not interpret
// group geometry. Alignment is checked only after the datatype is accepted.
if (ame_dtype_supported_m(uint32_t(RS1), P.AMU.ame_udsz) && uint32_t(RS1) != 0)
  require(P.AMU.m_index_legal(insn.rd(), uint32_t(RS1)));
P.AMU.msettyp(insn.rd(), uint32_t(RS1));
dirty_ame_state;
