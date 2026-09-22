// See LICENSE for license details.

#ifndef _RISCV_AME_EXT_MACROS_H
#define _RISCV_AME_EXT_MACROS_H

#include "ame_unit.h"
#include "ame_encoding.h"

#define require_ame require_extension(EXT_ZTT)
#define require_ame_ms require(STATE.sstatus->enabled(SSTATUS_AME_MS))
#define require_ame_owned require(P.AMU.owned)
#define dirty_ame_state STATE.sstatus->dirty(SSTATUS_AME_MS)

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
