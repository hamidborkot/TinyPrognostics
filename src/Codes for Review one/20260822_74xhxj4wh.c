/* ai_perf.h — DWT-based latency + RAM watermark measurement */
#ifndef AI_PERF_H
#define AI_PERF_H

#include "stdint.h"
#include "stm32h7xx_hal.h"

/* ---- DWT cycle counter ---- */
static inline void perf_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t perf_start(void) {
    DWT->CYCCNT = 0;
    return DWT->CYCCNT;
}
static inline uint32_t perf_stop(void) {
    return DWT->CYCCNT;
}

/* ---- RAM watermark: fill stack/heap region with a pattern ---- */
extern uint32_t _estack;      /* top of stack (linker) */
extern uint32_t _Min_Stack_Size;
#define WATERMARK 0xDEADBEEFu

void perf_ram_fill(void);
uint32_t perf_ram_used_peak(void);   /* bytes of stack touched */

#endif