/* ai_perf.c */
#include "ai_perf.h"

static uint32_t *g_stack_base;
static uint32_t  g_stack_words;

void perf_ram_fill(void) {
    /* Fill the free stack region with a known pattern.
       After inference we scan to find how far it was overwritten. */
    extern uint32_t _ebss, _estack;
    uint32_t *p   = &_ebss;            /* start after .bss */
    uint32_t *top = &_estack - 16;     /* leave guard for this fill */
    g_stack_base  = p;
    g_stack_words = (uint32_t)(top - p);
    for (uint32_t i = 0; i < g_stack_words; i++) p[i] = WATERMARK;
}

uint32_t perf_ram_used_peak(void) {
    uint32_t i = 0;
    while (i < g_stack_words && g_stack_base[i] == WATERMARK) i++;
    /* bytes actually touched = (words - untouched) * 4 */
    return (g_stack_words - i) * 4u;
}