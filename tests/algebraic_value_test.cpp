// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// End-to-end test of the algebraic Semantics wired into Value<>.
//   c++ -std=c++17 -I include tests/algebraic_value_test.cpp -o /tmp/algv && /tmp/algv
// ----------------------------------------------------------------------------
#include "fpsan/cast.hpp"
#include "fpsan/math.hpp"
#include "fpsan/value.hpp"

#include <cstdio>

using namespace fpsan;
template <Semantics S>
using F = Value<float, S, Conversions::Explicit>;

static long pass = 0, fail = 0;
static void check(bool ok, const char* msg)
{
    if(ok) ++pass; else { ++fail; std::printf("  FAIL: %s\n", msg); }
}

// Generic multiply-accumulate over an arbitrary index order -- the exact shape
// the MMA dataflows use (acc = acc + a*b). Written once, instantiated for any
// Semantics: this is the orthogonality the intrinsic layer relies on.
template <class V>
static V mac(const float* a, const float* b, int n, const int* order)
{
    V acc{0.0f};
    for(int i = 0; i < n; ++i)
        acc = acc + V{a[order[i]]} * V{b[order[i]]};
    return acc;
}

int main()
{
    using Alg   = F<Semantics::FPSanAlgebraic1>;
    using Scr   = F<Semantics::FPSanLikeTriton>; // Triton-style free model

    // ---- algebraic = value model: rational identities hold within a width ----
    check((Alg{2.0f} + Alg{2.0f}) == Alg{4.0f}, "alg: 2+2 == 4");
    check((Alg{3.0f} * Alg{3.0f}) == Alg{9.0f}, "alg: 3*3 == 9");
    check((Alg{1.5f} + Alg{1.5f}) == (Alg{2.0f} * Alg{1.5f}), "alg: x+x == 2x");
    {
        Alg a{2.5f}, b{1.25f}, c{0.5f};
        check(a * (b + c) == a * b + a * c, "alg: distributivity");
    }
    // field: x/x == 1 for every nonzero value
    {
        long ok = 0, n = 0;
        for(int i = 1; i <= 300; ++i, ++n)
            ok += ((Alg{(float)i} / Alg{(float)i}) == Alg{1.0f});
        check(ok == n, "alg: x/x == 1 (field)");
    }
    // commutativity / associativity
    check(Alg{1.1f} + Alg{2.2f} == Alg{2.2f} + Alg{1.1f}, "alg: a+b == b+a");
    check((Alg{1.1f} + Alg{2.2f}) + Alg{3.3f} == Alg{1.1f} + (Alg{2.2f} + Alg{3.3f}),
          "alg: associativity");

    // ---- contrast: the FREE model (FPSan) does NOT see value coincidences ----
    check((Scr{2.0f} + Scr{2.0f}) != Scr{4.0f}, "fpsan free model: 2+2 != 4");

    // ---- a second prime variant is independent but obeys the same identities -
    using Alg2 = F<Semantics::FPSanAlgebraic2>;
    check((Alg2{2.0f} + Alg2{2.0f}) == Alg2{4.0f}, "alg2: 2+2 == 4");

    // ---- the Exponentials variant honors exp(a+b) == exp(a)*exp(b) ----
    using Exp = F<Semantics::FPSanAlgebraicExponentials1>;
    check(exp(Exp{0.0f}) == Exp{1.0f}, "exp: exp(0) == 1");
    {
        long ok = 0, n = 0;
        float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 0.25f, 3.0f};
        for(float u : xs)
            for(float v : xs)
            {
                Exp a{u}, b{v};
                ok += (exp(a + b) == exp(a) * exp(b));
                ++n;
            }
        check(ok == n, "exp: exp(a+b) == exp(a)*exp(b) (Exponentials variant)");
    }
    // the Field variant has no exp homomorphism (exp is a tagged token there)
    check(exp(Alg{1.25f} + Alg{2.5f}) != exp(Alg{1.25f}) * exp(Alg{2.5f}),
          "field: exp is NOT a homomorphism (tagged token)");

    // Exp2 is an independent variant: its own exp homomorphism, distinct modulus.
    using Exp2 = F<Semantics::FPSanAlgebraicExponentials2>;
    check(exp(Exp2{1.5f} + Exp2{2.5f}) == exp(Exp2{1.5f}) * exp(Exp2{2.5f}),
          "exp2-variant: exp(a+b)==exp(a)*exp(b)");
    // 0.5 -> (n+1)/2 differs between the two moduli (small integers wouldn't).
    check(Exp{0.5f}.fpsan_payload() != Exp2{0.5f}.fpsan_payload(),
          "Exp1 and Exp2 use distinct moduli (different residue for 0.5)");

    // ---- log: the Exp variant honors log(x*y) == log(x) + log(y) ----
    check(log(Exp{1.0f}) == Exp{0.0f}, "log: log(1) == 0");
    {
        long  ok = 0, n = 0;
        float xs[] = {1.0f, 2.0f, 3.0f, 5.0f, 0.5f, 1.5f, 7.0f};
        for(float u : xs)
            for(float v : xs)
            {
                Exp a{u}, b{v};
                ok += (log(a * b) == log(a) + log(b));
                ++n;
            }
        check(ok == n, "log: log(x*y) == log(x)+log(y) (Exp variant, dlog homomorphism)");
    }
    // log inverts exp on the exp-image: exp(log(exp v)) == exp v
    check(exp(log(exp(Exp{1.5f}))) == exp(Exp{1.5f}), "log: exp(log(exp v)) == exp v");
    // Field variant: log is a tagged token, not a homomorphism
    check(log(Alg{2.0f} * Alg{3.0f}) != log(Alg{2.0f}) + log(Alg{3.0f}),
          "field: log is NOT a homomorphism (tagged token)");

    // ---- Trigonometry variant: genuine sin/cos angle-addition (order-d rotation)
    using Trig = F<Semantics::FPSanAlgebraicTrigonometry1>;
    check(cos(Trig{0.0f}) == Trig{1.0f}, "trig: cos(0) == 1");
    check(sin(Trig{0.0f}) == Trig{0.0f}, "trig: sin(0) == 0");
    {
        long  ok = 0, n = 0;
        float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, -1.0f, 0.25f};
        for(float u : xs)
            for(float v : xs)
            {
                Trig a{u}, b{v};
                bool c1 = (cos(a + b) == cos(a) * cos(b) - sin(a) * sin(b));
                bool c2 = (sin(a + b) == sin(a) * cos(b) + cos(a) * sin(b));
                ok += (c1 && c2);
                ++n;
            }
        check(ok == n, "trig: angle-addition for sin & cos (Trig variant)");
    }
    check(cos(Trig{1.3f}) * cos(Trig{1.3f}) + sin(Trig{1.3f}) * sin(Trig{1.3f}) == Trig{1.0f},
          "trig: cos^2 + sin^2 == 1");
    // a Trig variant ALSO carries exp + log (p=4d+1 keeps the d-channel)
    check(exp(Trig{1.0f} + Trig{2.0f}) == exp(Trig{1.0f}) * exp(Trig{2.0f}),
          "trig: exp homomorphism still holds");
    check(log(Trig{2.0f} * Trig{3.0f}) == log(Trig{2.0f}) + log(Trig{3.0f}),
          "trig: log homomorphism still holds");
    // the Exp variant has NO angle-addition (sin/cos are tagged tokens there)
    check(cos(Exp{0.5f} + Exp{1.0f})
              != cos(Exp{0.5f}) * cos(Exp{1.0f}) - sin(Exp{0.5f}) * sin(Exp{1.0f}),
          "exp-variant: sin/cos are tagged (no angle-addition)");

    // ---- Inf / NaN reach the payload, via 1/0 ----
    check(((Alg{1.0f} / Alg{0.0f}) / (Alg{1.0f} / Alg{0.0f})) == (Alg{1.0f} / Alg{0.0f}) /
              (Alg{1.0f} / Alg{0.0f}),
          "alg: Inf compares equal to itself (deterministic)");

    // ---- orthogonality: the SAME generic dataflow runs for every Semantics, and
    //      is reassociation-invariant in a payload mode (the sanitizer property) --
    {
        float A[4] = {0.5f, 1.25f, -0.75f, 2.0f};
        float B[4] = {2.0f, 0.5f, 4.0f, 1.5f};
        int   o1[4] = {0, 1, 2, 3};
        int   o2[4] = {3, 1, 0, 2}; // a different accumulation order
        // compiles & runs identically for Float, FPSan, and the algebraic variants:
        (void)mac<F<Semantics::Native>>(A, B, 4, o1);
        (void)mac<F<Semantics::FPSanLikeTriton>>(A, B, 4, o1);
        check(mac<Alg>(A, B, 4, o1) == mac<Alg>(A, B, 4, o2),
              "alg: matmul is reassociation-invariant (sanitizer property)");
        check(mac<Exp>(A, B, 4, o1) == mac<Exp>(A, B, 4, o2),
              "exp-variant: matmul is reassociation-invariant");
    }

    // ---- faithful fma: a*b+c exactly (value model) ----
    check(fma(Alg{2.0f}, Alg{3.0f}, Alg{1.0f}) == Alg{7.0f}, "alg: fma(2,3,1) == 7");
    check(fma(Alg{1.5f}, Alg{2.0f}, Alg{0.5f}) == Alg{3.5f}, "alg: fma value-faithful");

    // ---- min/max: deterministic, commutative, reassociation-invariant ----
    check(min(Alg{1.0f}, Alg{2.0f}) == min(Alg{2.0f}, Alg{1.0f}), "alg: min commutes");
    check(max(min(Alg{3.0f}, Alg{1.0f}), Alg{2.0f})
              == max(Alg{2.0f}, min(Alg{1.0f}, Alg{3.0f})),
          "alg: min/max reassoc-invariant");

    // ---- cast: same-width is identity; cross-width is deterministic ----
    check(cast<float>(Alg{1.25f}) == Alg{1.25f}, "alg: same-width cast is identity");
    {
        using H = Value<_Float16, Semantics::FPSanAlgebraic1, Conversions::Explicit>;
        H h1 = cast<_Float16>(Alg{1.5f});
        H h2 = cast<_Float16>(Alg{1.5f});
        check(h1 == h2, "alg: cross-width cast is deterministic");
    }

    std::printf("passed %ld, failed %ld\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
