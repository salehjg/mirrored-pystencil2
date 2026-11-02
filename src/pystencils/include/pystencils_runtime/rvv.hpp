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
 * The code generator (backend/platforms/rvv.py) assumes VLEN = 256 bits and emits
 * `__riscv_*` intrinsics with a constant vector length. This header provides what the
 * x86 platform gets from its own runtime headers:
 *
 *  - `set_<sfx>(...)`              -- build a vector from per-lane values (cf. `_mm256_set_*`)
 *  - `horizontal_<op>_<sfx>(d, v)` -- fold a vector into a scalar (cf. `_mm256_horizontal_*`)
 *  - `random::philox_float{32,64}` -- vectorized Philox RNG (cf. `x86::random`)
 *
 * `<sfx>` is the RVV type suffix, e.g. `f64m1` (4 x double) or `f32mf2` (4 x float).
 */

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "./bits/philox_rand.h"

//  The generated code is only correct on hardware with VLEN >= 256. We cannot check the
//  hardware at compile time, but we can make sure the compiler was told so
//  (`-march=..._zvl256b` sets `__riscv_v_min_vlen`), which is what the JIT passes.
#if !defined(__riscv_v_min_vlen) || __riscv_v_min_vlen < 256
#error "pystencils RVV kernels assume VLEN >= 256 bits: compile with -march=rv64gcv_zvl256b"
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

    //  All vector types the code generator may produce (see `_lmul` in rvv.py):
    //  full registers (m1) and, for elements of at most 32 bits, half registers (mf2).
    PYSTENCILS_RVV_DEFINE_SET(double, vfloat64m1_t, 64, f64m1)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32m1_t, 32, f32m1)
    PYSTENCILS_RVV_DEFINE_SET(float, vfloat32mf2_t, 32, f32mf2)
    PYSTENCILS_RVV_DEFINE_SET(int64_t, vint64m1_t, 64, i64m1)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32m1_t, 32, i32m1)
    PYSTENCILS_RVV_DEFINE_SET(int32_t, vint32mf2_t, 32, i32mf2)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16m1_t, 16, i16m1)
    PYSTENCILS_RVV_DEFINE_SET(int16_t, vint16mf2_t, 16, i16mf2)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8m1_t, 8, i8m1)
    PYSTENCILS_RVV_DEFINE_SET(int8_t, vint8mf2_t, 8, i8mf2)

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

    //  Lane counts follow from VLEN = 256: m1 holds 256 bits, mf2 holds 128 bits.
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(double, vfloat64m1_t, 64, f64m1, 4)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32m1_t, 32, f32m1, 8)
    PYSTENCILS_RVV_DEFINE_HORIZONTAL(float, vfloat32mf2_t, 32, f32mf2, 4)

#undef PYSTENCILS_RVV_DEFINE_HORIZONTAL

    /**
     * RANDOM NUMBER GENERATION
     *
     * Same contract as `pystencils::runtime::x86::random`: counters come in as vectors of
     * indices with the same bit width as the result type, keys are scalars, and the
     * results are returned as an array of vectors -- here an RVV tuple type, because RVV
     * vector types are sizeless and cannot be members of a struct.
     *
     * The Philox kernels in bits/philox_rand.h work on full `vuint32m1_t` registers
     * (`vsetvlmax_e32m1()` lanes = 8 at VLEN = 256), which matches our 8-lane float
     * vectors exactly.
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
            //  4 x int64 counters -> the low 32 bits of each, in lanes 0..3 of a full
            //  u32 register (x86 does the same with `_mm256_pack_2x64_epi32`).
            //  Lanes 4..7 stay unspecified: Philox treats lanes independently and
            //  `philox_double2` reads only lanes 0..3 for its "lo" outputs.
            inline vuint32m1_t ctr64_to_32(vint64m1_t ctr)
            {
                vuint32mf2_t lo = __riscv_vncvt_x_x_w_u32mf2(__riscv_vreinterpret_v_i64m1_u64m1(ctr), 4);
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
