// See LICENSE for license details.

#ifndef _RISCV_AME_EXT_MACROS_H
#define _RISCV_AME_EXT_MACROS_H

#include "ame_unit.h"
#include "ame_encoding.h"

#define require_ame require_extension(EXT_ZTT)
#define require_ame_ms require(STATE.sstatus->enabled(SSTATUS_AME_MS))
#define require_ame_owned require(P.AMU.owned)

// Every AME instruction that can perform floating-point arithmetic ends with
// this macro.  AME accrues its IEEE flags in `amefflags` (spec 6.2.3: "do not
// modify `fflags`"), but the softfloat kernel it drives is the same
// process-global one the scalar unit uses, and it leaves its accrual behind.
// A later scalar floating-point instruction drains that variable into `fflags`
// when it retires, so the residue would be reported as an exception of an
// instruction that raised none.  Clearing here, once the AME accrual has been
// moved to `amefflags`, is what keeps the two flag registers independent.
#define dirty_ame_state \
  do { \
    softfloat_exceptionFlags = 0; \
    STATE.sstatus->dirty(SSTATUS_AME_MS); \
  } while (0)

#define require_ame_insn \
  do { \
    require_ame; \
    require_ame_ms; \
  } while (0)

#define require_ame_owned_insn \
  do { \
    require_ame_insn; \
    require_ame_owned; \
  } while (0)

#endif
