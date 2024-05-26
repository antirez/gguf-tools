#ifndef BF16_h
#define BF16_h
#include <stdint.h>

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float from_brain(uint16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h << 16;
    return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This function is binary identical to AMD Zen4 VCVTNEPS2BF16.
 * Subnormals shall be flushed to zero, and NANs will be quiet.
 * This code should vectorize nicely if using modern compilers.
 */
static inline uint16_t to_brain(float s) {
    uint16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    if (!(u.i & 0x7f800000)) { /* subnormal */
        h = (u.i & 0x80000000) >> 16; /* flush to zero */
        return h;
    }
    return (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
}

#endif
