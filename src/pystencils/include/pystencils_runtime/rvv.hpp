/*
Copyright 2026 pystencils authors

This file is part of pystencils.

pystencils is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

pystencils is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with pystencils.
If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

/**
 * @brief Runtime library for the RISC-V Vector (RVV 1.0) code generation platform.
 *
 * The code generator (backend/platforms/rvv.py) emits `__riscv_*` intrinsics with a constant
 * vector length for a fixed VLEN: 256 bits (`Target.RISCV_RVV256`, the default) or 128 bits
 * (`Target.RISCV_RVV128`, whose kernels include rvv_vlen128.hpp, which sets
 * `PYSTENCILS_RVV_VLEN` before including this file). This header provides what the x86
 * platform gets from its own runtime headers:
 *
 *  - `set_<sfx>(...)`              -- build a vector from per-lane values (cf. `_mm256_set_*`)
 *  - `horizontal_<op>_<sfx>(d, v)` -- fold a vector into a scalar (cf. `_mm256_horizontal_*`)
 *  - `random::philox_float{32,64}` -- vectorized Philox RNG (cf. `x86::random`)
 *
 * `<sfx>` is the RVV type suffix, e.g. `f64m1` (one register of doubles: 4 at VLEN 256,
 * 2 at VLEN 128) or `f32mf2` (half a register of floats).
 */

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "./bits/philox_rand.h"

//  The VLEN the kernels were generated for (rvv_vlen128.hpp sets 128 before including this file).
#ifndef PYSTENCILS_RVV_VLEN
#define PYSTENCILS_RVV_VLEN 256
#endif

//  The generated code is only correct on hardware with VLEN >= PYSTENCILS_RVV_VLEN. We cannot
//  check the hardware at compile time, but we can make sure the compiler was told so
//  (`-march=..._zvl<VLEN>b` sets `__riscv_v_min_vlen`), which is what the JIT passes.
#if !defined(__riscv_v_min_vlen) || __riscv_v_min_vlen < PYSTENCILS_RVV_VLEN
#if PYSTENCILS_RVV_VLEN == 128
#error "pystencils RVV kernels generated for VLEN 128 need -march=rv64gcv_zvl128b (or a larger zvl)"
#else
#error "pystencils RVV kernels generated for VLEN 256 need -march=rv64gcv_zvl256b"
#endif
#endif

namespace pystencils::runtime::rvv
{

    /**
     * CONSTANT VECTORS
     *
     * `set_<sfx>(v0, v1, ...)` returns the vector [v0, v1, ...] (lane 0 first). RVV has no
     * intrinsic for this, so the values go through a small stack array; the code
     * generator hoists constant vectors out of loops, so this runs once per kernel call.
     */

#define PYSTENCILS_RVV_DEFINE_SET(SCALAR, VTYPE, SEW, SFX)                     \
    template <typename... Ts>                                                \
    inline VTYPE set_##SFX(Ts... values)                                     \
    {                                                                        \
        const SCALAR lanes[] = {static_cast<SCALAR>(values)...};             \
        return __riscv_vle##SEW##_v_##SFX(lanes, sizeof...(Ts));             \
    }

    //  All vector types the code generator may produce (see `_lmul` in rvv.py): register groups
    //  m1 to m8 and, for narrow enough elements, fractional registers (mf2, mf4, mf8).
    PYSTENCILS_RVV_DEFINE_SET(double, vfloat64m1_t, 64, f64m1)
    PYSTENCILS_RVV_DEFINE_SET(double, vfloat64m2_t, 64, f64m2)
    PYSTENCILS_RVV_DEFINE_SET(double, vfloat64m4_t, 64, f64m4)
    PYSTENCILS_RVV_DEFINE_SET(double, vfloat64m8_t, 64, f64m8)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32mf2_t, 32, f32mf2)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32m1_t, 32, f32m1)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32m2_t, 32, f32m2)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32m4_t, 32, f32m4)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32m8_t, 32, f32m8)
    PYSTENCILS_RVV_DEFINE_SET(int64_t, vint64m1_t, 64, i64m1)
    PYSTENCILS_RVV_DEFINE_SET(int64_t, vint64m2_t, 64, i64m2)
    PYSTENCILS_RVV_DEFINE_SET(int64_t, vint64m4_t, 64, i64m4)
    PYSTENCILS_RVV_DEFINE_SET(int64_t, vint64m8_t, 64, i64m8)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32mf2_t, 32, i32mf2)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32m1_t, 32, i32m1)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32m2_t, 32, i32m2)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32m4_t, 32, i32m4)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32m8_t, 32, i32m8)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16mf4_t, 16, i16mf4)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16mf2_t, 16, i16mf2)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16m1_t, 16, i16m1)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16m2_t, 16, i16m2)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16m4_t, 16, i16m4)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16m8_t, 16, i16m8)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8mf8_t, 8, i8mf8)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8mf4_t, 8, i8mf4)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8mf2_t, 8, i8mf2)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8m1_t, 8, i8m1)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8m2_t, 8, i8m2)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8m4_t, 8, i8m4)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8m8_t, 8, i8m8)

#undef PYSTENCILS_RVV_DEFINE_SET

    /**
     * HORIZONTAL REDUCTIONS
     *
     * `horizontal_<op>_<sfx>(dst, v)` returns `dst <op> v[0] <op> v[1] <op> ...`.
     *
     * Implemented the simple way: store the lanes and fold them with a scalar loop. This
     * gives exactly the semantics (and summation order) of scalar code for all four
     * operations -- RVV has no multiply reduction anyway -- and reductions only run once
     * per kernel call, after the vectorized loop.
     */

#define PYSTENCILS_RVV_DEFINE_HORIZONTAL(SCALAR, VTYPE, SEW, SFX, LANES)       \
    inline SCALAR horizontal_add_##SFX(SCALAR dst, VTYPE src)                \
    {                                                                        \
        SCALAR lanes[LANES];                                                 \
        __riscv_vse##SEW##_v_##SFX(lanes, src, LANES);                       \
        for (size_t i = 0; i < LANES; ++i)                                   \
            dst += lanes[i];                                                 \
        return dst;                                                          \
    }                                                                        \
    inline SCALAR horizontal_mul_##SFX(SCALAR dst, VTYPE src)                \
    {                                                                        \
        SCALAR lanes[LANES];                                                 \
        __riscv_vse##SEW##_v_##SFX(lanes, src, LANES);                       \
        for (size_t i = 0; i < LANES; ++i)                                   \
            dst *= lanes[i];                                                 \
        return dst;                                                          \
    }                                                                        \
    inline SCALAR horizontal_min_##SFX(SCALAR dst, VTYPE src)                \
    {                                                                        \
        SCALAR lanes[LANES];                                                 \
        __riscv_vse##SEW##_v_##SFX(lanes, src, LANES);                       \
        for (size_t i = 0; i < LANES; ++i)                                   \
            dst = lanes[i] < dst ? lanes[i] : dst;                           \
        return dst;                                                          \
    }                                                                        \
    inline SCALAR horizontal_max_##SFX(SCALAR dst, VTYPE src)                \
    {                                                                        \
        SCALAR lanes[LANES];                                                 \
        __riscv_vse##SEW##_v_##SFX(lanes, src, LANES);                       \
        for (size_t i = 0; i < LANES; ++i)                                   \
            dst = lanes[i] > dst ? lanes[i] : dst;                           \
        return dst;                                                          \
    }

    //  Lane counts follow from the VLEN the kernels were generated for (not from
    //  `__riscv_v_min_vlen`: code for VLEN 128 may be compiled for, and run on, a larger VLEN,
    //  and then still fills only its own lanes). E.g. m1 holds VLEN bits, mf2 VLEN / 2.
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(double, vfloat64m1_t, 64, f64m1, PYSTENCILS_RVV_VLEN / 64)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(double, vfloat64m2_t, 64, f64m2, PYSTENCILS_RVV_VLEN * 2 / 64)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(double, vfloat64m4_t, 64, f64m4, PYSTENCILS_RVV_VLEN * 4 / 64)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(double, vfloat64m8_t, 64, f64m8, PYSTENCILS_RVV_VLEN * 8 / 64)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32mf2_t, 32, f32mf2, PYSTENCILS_RVV_VLEN / 2 / 32)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32m1_t, 32, f32m1, PYSTENCILS_RVV_VLEN / 32)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32m2_t, 32, f32m2, PYSTENCILS_RVV_VLEN * 2 / 32)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32m4_t, 32, f32m4, PYSTENCILS_RVV_VLEN * 4 / 32)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32m8_t, 32, f32m8, PYSTENCILS_RVV_VLEN * 8 / 32)

#undef PYSTENCILS_RVV_DEFINE_HORIZONTAL

    /**
     * SOFTWARE PREFETCH
     *
     * Emitted by the code generator when `cpu.vectorize.prefetch_distance` is set. These are
     * Zicbop's `prefetch.r` / `prefetch.w`, which are encoded as RISC-V HINTs
     * (`ori x0, rs1, imm`): cores without Zicbop execute them as no-ops, so no `-march`
     * flag is needed. The SpacemiT X60 executes them although its ISA string does not list
     * Zicbop. With `-march=..._zicbop` the compiler builtin is used, which can fold the offset.
     */
    template <typename T>
    inline void prefetch_read(const T *p)
    {
#if defined(__riscv_zicbop)
        __builtin_prefetch(p, 0, 3);
#else
        __asm__ volatile("ori x0, %0, 1" : : "r"(p));
#endif
    }

    template <typename T>
    inline void prefetch_write(const T *p)
    {
#if defined(__riscv_zicbop)
        __builtin_prefetch(p, 1, 3);
#else
        __asm__ volatile("ori x0, %0, 3" : : "r"(p));
#endif
    }

    /**
     * SCALAR OPERANDS IN REGISTERS
     *
     * `in_register(x)` returns `x`, but the compiler can no longer see that the value is a
     * constant. The code generator passes the scalar operands of `.vf` instructions that are
     * compile-time constants through it. Otherwise GCC rematerializes every such operand
     * inside the loop (`lla` + `flw` per use) instead of keeping it in one of the
     * floating-point registers, which a vector loop leaves idle.
     */
    inline float in_register(float x)
    {
        __asm__("" : "+f"(x));
        return x;
    }

    inline double in_register(double x)
    {
        __asm__("" : "+f"(x));
        return x;
    }

    /**
     * RANDOM NUMBER GENERATION
     *
     * Same contract as `pystencils::runtime::x86::random`: counters come in as vectors of
     * indices with the same bit width as the result type, keys are scalars, and the
     * results are returned as an array of vectors -- here an RVV tuple type, because RVV
     * vector types are sizeless and cannot be members of a struct.
     *
     * The Philox kernels in bits/philox_rand.h work on full `vuint32m1_t` registers
     * (`vsetvlmax_e32m1()` lanes), which matches our float vectors at LMUL 1 (the code
     * generator allows random numbers only there).
     */

    namespace random
    {
        template <typename Idx>
        vfloat32m1x4_t philox_float32(vint32m1_t ctr0, vint32m1_t ctr1, vint32m1_t ctr2, vint32m1_t ctr3,
                                      Idx key0, Idx key1)
        {
            vfloat32m1_t r0, r1, r2, r3;
            ::pystencils::runtime::random::detail::philox_float4(
                __riscv_vreinterpret_v_i32m1_u32m1(ctr0), __riscv_vreinterpret_v_i32m1_u32m1(ctr1),
                __riscv_vreinterpret_v_i32m1_u32m1(ctr2), __riscv_vreinterpret_v_i32m1_u32m1(ctr3),
                (uint32_t)key0, (uint32_t)key1,
                r0, r1, r2, r3);
            return __riscv_vcreate_v_f32m1x4(r0, r1, r2, r3);
        }

        namespace detail
        {
            //  N = VLEN / 64 int64 counters -> the low 32 bits of each, in lanes 0..N-1 of a
            //  full u32 register (x86 does the same with `_mm256_pack_2x64_epi32`).
            //  The upper lanes stay unspecified: Philox treats lanes independently and
            //  `philox_double2` reads only lanes 0..N-1 for its "lo" outputs.
            inline vuint32m1_t ctr64_to_32(vint64m1_t ctr)
            {
                vuint32mf2_t lo = __riscv_vncvt_x_x_w_u32mf2(__riscv_vreinterpret_v_i64m1_u64m1(ctr),
                                                             PYSTENCILS_RVV_VLEN / 64);
                return __riscv_vlmul_ext_v_u32mf2_u32m1(lo);
            }
        } // namespace detail

        template <typename Idx>
        vfloat64m1x2_t philox_float64(vint64m1_t ctr0, vint64m1_t ctr1, vint64m1_t ctr2, vint64m1_t ctr3,
                                      Idx key0, Idx key1)
        {
            vfloat64m1_t r0, r1, ignore;
            ::pystencils::runtime::random::detail::philox_double2(
                detail::ctr64_to_32(ctr0), detail::ctr64_to_32(ctr1),
                detail::ctr64_to_32(ctr2), detail::ctr64_to_32(ctr3),
                (uint32_t)key0, (uint32_t)key1,
                r0, ignore, r1, ignore);
            return __riscv_vcreate_v_f64m1x2(r0, r1);
        }
    } // namespace random

} // namespace pystencils::runtime::rvv
