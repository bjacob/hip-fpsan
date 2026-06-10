// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/fpsan_semantics.hpp
// ----------------------------------------------------------------------------
// The single source of truth for "which FPSan-family semantics every device
// self-consistency test exercises". A self-consistency test asserts that the
// device payload equals the host recomputation in the SAME semantics, so it
// generalizes to every value model; the test bodies are written generic over
// Semantics and driven by the loop below.
//
// To add, remove, or rename a Semantics variant, edit ONLY this list -- every
// test that loops over for_each_fpsan_semantics picks the change up, with no
// per-test, per-intrinsic, or per-gfx-arch edits. The intrinsic wrappers are
// already generic over Semantics, so they are never touched.
//
// Native is excluded on purpose: it is the bit-exact-vs-hardware oracle (the
// reference the fpsan payloads are checked against), driven separately by each
// test, not a self-consistency variant.
// ----------------------------------------------------------------------------
#ifndef FPSAN_TESTS_FPSAN_SEMANTICS_HPP
#define FPSAN_TESTS_FPSAN_SEMANTICS_HPP

#include "fpsan/value.hpp"

#include <type_traits>

namespace fpsan_test
{
// Invoke f(std::integral_constant<Semantics, S>{}) for each FPSan-family
// semantics. Use a generic lambda and read the value as decltype(sem)::value:
//
//   for_each_fpsan_semantics([](auto sem) {
//       run_my_self_consistency_test<Traits, decltype(sem)::value>();
//   });
template <class F>
void for_each_fpsan_semantics(F&& f)
{
    using S = fpsan::Semantics;
    f(std::integral_constant<S, S::FPSanLikeTriton>{});
    f(std::integral_constant<S, S::FPSanAlgebraicField>{});
    f(std::integral_constant<S, S::FPSanAlgebraicField2>{});
    f(std::integral_constant<S, S::FPSanAlgebraicRingSophieGermain>{});
    f(std::integral_constant<S, S::FPSanAlgebraicRingSophieGermain2>{});
    f(std::integral_constant<S, S::FPSanAlgebraicRingPythagorean>{});
    f(std::integral_constant<S, S::FPSanAlgebraicRingPythagorean2>{});
}

} // namespace fpsan_test

#endif // FPSAN_TESTS_FPSAN_SEMANTICS_HPP
