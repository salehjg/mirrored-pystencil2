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

//  RISC-V Vector extension (RVV 1.0) runtime support for pystencils.
//
//  This header re-exports <riscv_vector.h> and provides:
//    * the VLS runtime guard `require_vlen` (design decision D8), and
//    * length-agnostic Philox RNG wrappers (D13) -- see below.
//
//  The generated kernel refers to intrinsics as `__riscv_*` free functions (from
//  <riscv_vector.h>) directly; only the helpers that need a stable C++ symbol live here.

#include <riscv_vector.h>

#include <cstdio>
#include <cstdlib>

#include "./bits/philox_rand.h"

namespace pystencils::runtime::rvv
{
    /**
     * VLS runtime guard (design decision D8).
     *
     * A kernel generated for an assumed VLEN of `required_bits` is only correct on
     * hardware whose VLEN is at least that wide. On narrower hardware, `vsetvl(L)`
     * grants fewer than `L` lanes while the outer loop still strides by `L`, silently
     * skipping elements. This guard turns that silent corruption into a loud abort.
     *
     * `assert()` is unusable here because the JIT compiles with -DNDEBUG.
     */
    inline void require_vlen(unsigned long required_bits)
    {
        unsigned long hw_bits = (unsigned long)__riscv_vlenb() * 8ul;
        if (hw_bits < required_bits)
        {
            std::fprintf(stderr,
                         "pystencils RVV kernel: hardware VLEN = %lu bits is narrower than the "
                         "%lu bits this kernel was generated for; aborting to avoid silent data "
                         "corruption.\n",
                         hw_bits, required_bits);
            std::abort();
        }
    }

    /**
     * VLS runtime guard requiring VLEN to be *exactly* `required_bits`.
     *
     * Used for kernels (e.g. the LMUL=1 Philox RNG in VLS mode) whose correctness
     * depends on the lane count matching VLMAX exactly, not merely being large enough.
     */
    inline void require_vlen_exact(unsigned long required_bits)
    {
        unsigned long hw_bits = (unsigned long)__riscv_vlenb() * 8ul;
        if (hw_bits != required_bits)
        {
            std::fprintf(stderr,
                         "pystencils RVV kernel: hardware VLEN = %lu bits does not match the "
                         "%lu bits this kernel requires exactly; aborting.\n",
                         hw_bits, required_bits);
            std::abort();
        }
    }

    namespace random
    {
        //  Philox4x32 wrappers (design decision D13).
        //
        //  The length-fixed (LMUL=1) Philox kernels live in bits/philox_rand.h and run at
        //  `vsetvlmax` internally, so these wrappers are only valid when the kernel's lane
        //  count equals VLMAX(e32m1) (= VLEN/32). The RVV intrinsic selector enforces that
        //  precondition; here we just adapt the counter/key arguments and pack the four
        //  result vectors into an RVV tuple via `vcreate`.

        template <typename Idx>
        vfloat32m1x4_t philox_float32(vuint32m1_t ctr0, vuint32m1_t ctr1, vuint32m1_t ctr2, vuint32m1_t ctr3, Idx key0, Idx key1)
        {
            vfloat32m1_t r0, r1, r2, r3;
            pystencils::runtime::random::detail::philox_float4(
                ctr0, ctr1, ctr2, ctr3,
                (uint32_t)key0, (uint32_t)key1,
                r0, r1, r2, r3);
            return __riscv_vcreate_v_f32m1x4(r0, r1, r2, r3);
        }

        template <typename Idx>
        vfloat32m1x4_t philox_float32(vint32m1_t ctr0, vint32m1_t ctr1, vint32m1_t ctr2, vint32m1_t ctr3, Idx key0, Idx key1)
        {
            return philox_float32(
                __riscv_vreinterpret_v_i32m1_u32m1(ctr0),
                __riscv_vreinterpret_v_i32m1_u32m1(ctr1),
                __riscv_vreinterpret_v_i32m1_u32m1(ctr2),
                __riscv_vreinterpret_v_i32m1_u32m1(ctr3),
                key0, key1);
        }
    } // namespace random
} // namespace pystencils::runtime::rvv
