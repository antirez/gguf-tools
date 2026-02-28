/* See fp16.c. */

#ifndef FP16_h
#define FP16_h

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#endif

float from_half(uint16_t h);
uint16_t to_half(float f);

#ifdef __cplusplus
}
#endif

#endif
