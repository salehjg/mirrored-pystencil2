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

//  Kernels generated for `Target.RISCV_RVV128` include this header instead of rvv.hpp: it tells
//  rvv.hpp the VLEN they were generated for (its compile-time guard, the reduction lane counts).
#if defined(PYSTENCILS_RVV_VLEN) && PYSTENCILS_RVV_VLEN != 128
#error "pystencils RVV kernels generated for different VLENs in one translation unit"
#endif
#define PYSTENCILS_RVV_VLEN 128

#include "rvv.hpp"
