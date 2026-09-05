/* stdendian.h — compatibility shim for pre-C23 compilers.
 * Provides BYTE_ORDER, LITTLE_ENDIAN, BIG_ENDIAN macros.
 * Required by Fathom (tbprobe.c). */
#ifndef STDENDIAN_H
#define STDENDIAN_H

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__)
  /* GCC/Clang predefined macros */
  #define LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
  #define BIG_ENDIAN    __ORDER_BIG_ENDIAN__
  #define BYTE_ORDER    __BYTE_ORDER__
  /* Single-underscore variants used by Fathom (tbprobe.c) */
  #ifndef _LITTLE_ENDIAN
    #define _LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
  #endif
  #ifndef _BIG_ENDIAN
    #define _BIG_ENDIAN    __ORDER_BIG_ENDIAN__
  #endif
  #ifndef _BYTE_ORDER
    #define _BYTE_ORDER    __BYTE_ORDER__
  #endif
#elif defined(_WIN32) || defined(_WIN64)
  /* Windows is always little-endian (x86/x64/ARM64) */
  #define LITTLE_ENDIAN 1234
  #define BIG_ENDIAN    4321
  #define BYTE_ORDER    LITTLE_ENDIAN
  #ifndef _LITTLE_ENDIAN
    #define _LITTLE_ENDIAN 1234
  #endif
  #ifndef _BIG_ENDIAN
    #define _BIG_ENDIAN    4321
  #endif
  #ifndef _BYTE_ORDER
    #define _BYTE_ORDER    1234
  #endif
#elif defined(__EMSCRIPTEN__)
  /* WASM is little-endian */
  #define LITTLE_ENDIAN 1234
  #define BIG_ENDIAN    4321
  #define BYTE_ORDER    LITTLE_ENDIAN
  #ifndef _LITTLE_ENDIAN
    #define _LITTLE_ENDIAN 1234
  #endif
  #ifndef _BIG_ENDIAN
    #define _BIG_ENDIAN    4321
  #endif
  #ifndef _BYTE_ORDER
    #define _BYTE_ORDER    1234
  #endif
#else
  /* Default: assume little-endian (covers x86, ARM64) */
  #define LITTLE_ENDIAN 1234
  #define BIG_ENDIAN    4321
  #define BYTE_ORDER    LITTLE_ENDIAN
  #ifndef _LITTLE_ENDIAN
    #define _LITTLE_ENDIAN 1234
  #endif
  #ifndef _BIG_ENDIAN
    #define _BIG_ENDIAN    4321
  #endif
  #ifndef _BYTE_ORDER
    #define _BYTE_ORDER    1234
  #endif
#endif

/* Byte swap functions used by Fathom for big-endian support.
 * On little-endian (our case), these are only called in the big-endian path
 * which is dead code, but must compile. */
#ifndef bswap64
  #if defined(__GNUC__) || defined(__clang__)
    #define bswap64(x) __builtin_bswap64(x)
  #elif defined(_MSC_VER)
    #include <stdlib.h>
    #define bswap64(x) _byteswap_uint64(x)
  #else
    static inline uint64_t bswap64(uint64_t x) {
        return ((x >> 56) & 0xFFULL) | ((x >> 40) & 0xFF00ULL) |
               ((x >> 24) & 0xFF0000ULL) | ((x >> 8) & 0xFF000000ULL) |
               ((x << 8) & 0xFF00000000ULL) | ((x << 24) & 0xFF0000000000ULL) |
               ((x << 40) & 0xFF000000000000ULL) | ((x << 56) & 0xFF00000000000000ULL);
    }
  #endif
#endif
#ifndef bswap32
  #if defined(__GNUC__) || defined(__clang__)
    #define bswap32(x) __builtin_bswap32(x)
    #define bswap16(x) __builtin_bswap16(x)
  #elif defined(_MSC_VER)
    #include <stdlib.h>
    #define bswap32(x) _byteswap_ulong(x)
    #define bswap16(x) _byteswap_ushort(x)
  #else
    static inline uint32_t bswap32(uint32_t x) {
        return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
               ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
    }
    static inline uint16_t bswap16(uint16_t x) {
        return (x >> 8) | (x << 8);
    }
  #endif
#endif

#endif /* STDENDIAN_H */
