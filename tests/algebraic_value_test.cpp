// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// End-to-end test of the algebraic Semantics wired into Value<>.
//   c++ -std=c++17 -I include tests/algebraic_value_test.cpp -o /tmp/algv && /tmp/algv
// ----------------------------------------------------------------------------
#include "fpsan/cast.hpp"
#include "fpsan/detail/fp8.hpp"
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
    using Alg   = F<Semantics::FPSanAlgebraicField>;
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
    using Alg2 = F<Semantics::FPSanAlgebraicField2>;
    check((Alg2{2.0f} + Alg2{2.0f}) == Alg2{4.0f}, "alg2: 2+2 == 4");

    // ---- the Exponentials variant honors exp(a+b) == exp(a)*exp(b) ----
    using Exp = F<Semantics::FPSanAlgebraicRingSophieGermain>;
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
    using Exp2 = F<Semantics::FPSanAlgebraicRingSophieGermain2>;
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
    using Trig = F<Semantics::FPSanAlgebraicRingPythagorean>;
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

    // ---- exp2 / log2: a second homomorphism pair on the same order-d channel ----
    check(exp2(Exp{0.0f}) == Exp{1.0f}, "exp2: exp2(0) == 1");
    check(log2(Exp{1.0f}) == Exp{0.0f}, "log2: log2(1) == 0");
    {
        long  ok = 0, n = 0;
        float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 0.25f, 3.0f};
        for(float u : xs)
            for(float v : xs)
            {
                Exp a{u}, b{v};
                ok += (exp2(a + b) == exp2(a) * exp2(b));
                ++n;
            }
        check(ok == n, "exp2: exp2(a+b) == exp2(a)*exp2(b) (Exponentials variant)");
    }
    {
        long  ok = 0, n = 0;
        float xs[] = {1.0f, 2.0f, 3.0f, 5.0f, 0.5f, 1.5f, 7.0f};
        for(float u : xs)
            for(float v : xs)
            {
                Exp a{u}, b{v};
                ok += (log2(a * b) == log2(a) + log2(b));
                ++n;
            }
        check(ok == n, "log2: log2(x*y) == log2(x)+log2(y) (Exp variant, dlog homomorphism)");
    }
    check(exp2(log2(exp2(Exp{1.5f}))) == exp2(Exp{1.5f}), "log2: exp2(log2(exp2 v)) == exp2 v");
    // exp2 uses a distinct base, so it is NOT the same fingerprint as exp
    check(exp2(Exp{2.0f}) != exp(Exp{2.0f}), "exp2 != exp (distinct base change)");
    // exp2/log2 carry over to the Trig variant (also a two-moduli channel)
    check(exp2(Trig{1.0f} + Trig{2.0f}) == exp2(Trig{1.0f}) * exp2(Trig{2.0f}),
          "trig: exp2 homomorphism still holds");
    check(log2(Trig{2.0f} * Trig{3.0f}) == log2(Trig{2.0f}) + log2(Trig{3.0f}),
          "trig: log2 homomorphism still holds");
    // Field variant: exp2/log2 are tagged tokens, not homomorphisms
    check(exp2(Alg{1.25f} + Alg{2.5f}) != exp2(Alg{1.25f}) * exp2(Alg{2.5f}),
          "field: exp2 is NOT a homomorphism (tagged token)");
    check(log2(Alg{2.0f} * Alg{3.0f}) != log2(Alg{2.0f}) + log2(Alg{3.0f}),
          "field: log2 is NOT a homomorphism (tagged token)");

    // ---- exp10 / log10: the base-10 members of the same family ----
    check(exp10(Exp{0.0f}) == Exp{1.0f}, "exp10: exp10(0) == 1");
    check(log10(Exp{1.0f}) == Exp{0.0f}, "log10: log10(1) == 0");
    {
        long  e10 = 0, l10 = 0, inv = 0, n = 0;
        float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 7.0f};
        for(float u : xs)
            for(float v : xs)
            {
                Exp a{u}, b{v};
                e10 += (exp10(a + b) == exp10(a) * exp10(b));
                l10 += (log10(a * b) == log10(a) + log10(b));
                inv += (exp10(log10(exp10(a))) == exp10(a));
                ++n;
            }
        check(e10 == n, "exp10: exp10(a+b) == exp10(a)*exp10(b) (Exponentials variant)");
        check(l10 == n, "log10: log10(x*y) == log10(x)+log10(y)");
        check(inv == n, "log10: exp10(log10(exp10 v)) == exp10 v");
    }
    // the three bases are distinct fingerprints (e, 2, 10)
    check(exp10(Exp{2.0f}) != exp(Exp{2.0f}) && exp10(Exp{2.0f}) != exp2(Exp{2.0f}),
          "exp10 != exp and != exp2 (distinct base)");
    // carries to Trig; tokens in the Field variant
    check(exp10(Trig{1.0f} + Trig{2.0f}) == exp10(Trig{1.0f}) * exp10(Trig{2.0f}),
          "trig: exp10 homomorphism holds");
    check(exp10(Alg{1.25f} + Alg{2.5f}) != exp10(Alg{1.25f}) * exp10(Alg{2.5f}),
          "field: exp10 is NOT a homomorphism (tagged token)");

    // ---- sqrt / cbrt / rsqrt: multiplicative algebraic roots ----
    check(Alg::alg_cfg().has_cbrt, "field: cbrt is a perfect cube root (has_cbrt)");
    {
        long  ms = 0, mc = 0, c3 = 0, inset = 0, half = 0, n = 0;
        float xs[] = {1.f, 2.f, 3.f, 5.f, 7.f, 0.5f, 1.5f, 6.f, 9.f, 0.25f, 11.f, 13.f};
        for(float u : xs)
            for(float v : xs)
            {
                Alg a{u}, b{v};
                ms += (sqrt(a * b) == sqrt(a) * sqrt(b));   // sqrt multiplicative
                mc += (cbrt(a * b) == cbrt(a) * cbrt(b));   // cbrt multiplicative
                c3 += (cbrt(a) * cbrt(a) * cbrt(a) == a);   // cbrt perfect: cbrt(x)^3==x
                Alg s2 = sqrt(a) * sqrt(a);
                inset += ((s2 == a) || (s2 == -a));         // sqrt(x)^2 == +/- x always
                half += (s2 == a);                          // == x on the squares (~half)
                ++n;
            }
        check(ms == n, "field sqrt: sqrt(x*y) == sqrt(x)*sqrt(y) (multiplicative)");
        check(mc == n, "field cbrt: cbrt(x*y) == cbrt(x)*cbrt(y) (multiplicative)");
        check(c3 == n, "field cbrt: cbrt(x)^3 == x (perfect)");
        check(inset == n, "field sqrt: sqrt(x)^2 in {x, -x} always");
        check(half > 0 && half < n, "field sqrt: round-trip sqrt(x)^2==x on ~half (QRs)");
    }
    // rsqrt is exactly 1/sqrt
    {
        long  ok = 0, cons = 0, n = 0;
        float xs[] = {1.f, 2.f, 3.f, 5.f, 0.5f, 4.f, 9.f, 7.f};
        for(float u : xs)
        {
            Alg a{u};
            ok += (rsqrt(a) * sqrt(a) == Alg{1.0f});
            cons += (rsqrt(a) == Alg{1.0f} / sqrt(a));
            ++n;
        }
        check(ok == n, "field rsqrt: rsqrt(x)*sqrt(x) == 1");
        check(cons == n, "field rsqrt == 1/sqrt");
    }
    // Exp variant: sqrt multiplicative, cbrt a perfect cube root (3 coprime to lambda)
    check(Exp::alg_cfg().has_cbrt, "exp: cbrt is a perfect cube root (has_cbrt)");
    {
        long  ms = 0, c3 = 0, n = 0;
        float xs[] = {1.f, 2.f, 3.f, 5.f, 7.f, 0.5f, 1.5f, 6.f};
        for(float u : xs)
            for(float v : xs)
            {
                Exp a{u}, b{v};
                ms += (sqrt(a * b) == sqrt(a) * sqrt(b));
                c3 += (cbrt(a) * cbrt(a) * cbrt(a) == a);
                ++n;
            }
        check(ms == n, "exp sqrt: multiplicative");
        check(c3 == n, "exp cbrt: cbrt(x)^3 == x (perfect)");
    }
    // Trig variant: sqrt multiplicative; cbrt has no cube root here (3 | group order)
    check(!Trig::alg_cfg().has_cbrt, "trig: cbrt is a token (3 divides the group order)");
    {
        long  ms = 0, n = 0;
        float xs[] = {1.f, 2.f, 3.f, 5.f, 0.5f, 1.5f};
        for(float u : xs)
            for(float v : xs)
            {
                Trig a{u}, b{v};
                ms += (sqrt(a * b) == sqrt(a) * sqrt(b));
                ++n;
            }
        check(ms == n, "trig sqrt: multiplicative");
    }
    // contrast: the free model's sqrt is a tagged token (not multiplicative)
    check(sqrt(Scr{2.0f} * Scr{3.0f}) != sqrt(Scr{2.0f}) * sqrt(Scr{3.0f}),
          "fpsan free model: sqrt is a token (not multiplicative)");

    // ---- Field casts form a commutative diagram of homomorphisms ----
    // The fp4|fp8|fp16|fp32 primes are a coprime tower, so every widening and
    // narrowing cast is multiplicative, they compose, and narrow(widen(x)) == x.
    {
        using F8  = Value<fp8_e4m3, Semantics::FPSanAlgebraicField, Conversions::Explicit>;
        using F16 = Value<_Float16, Semantics::FPSanAlgebraicField, Conversions::Explicit>;
        using F32 = Value<float, Semantics::FPSanAlgebraicField, Conversions::Explicit>;
        long  w16 = 0, w32 = 0, nA = 0, rt = 0, n = 0;
        float xs[] = {1.f, 2.f, 3.f, 0.5f, 4.f, 6.f, 1.5f, 0.25f};
        for(float u : xs)
            for(float v : xs)
            {
                F8  a{static_cast<fp8_e4m3>(u)}, b{static_cast<fp8_e4m3>(v)};
                F32 c{u}, d{v};
                // widening is multiplicative (fp8 -> fp16, fp8 -> fp32)
                w16 += (cast<_Float16>(a * b) == cast<_Float16>(a) * cast<_Float16>(b));
                w32 += (cast<float>(a * b) == cast<float>(a) * cast<float>(b));
                // narrowing is multiplicative (fp32 -> fp8)
                nA += (cast<fp8_e4m3>(c * d) == cast<fp8_e4m3>(c) * cast<fp8_e4m3>(d));
                // up-then-down round trip recovers the original (narrow . widen == id)
                rt += (cast<fp8_e4m3>(cast<float>(a)) == a);
                ++n;
            }
        check(w16 == n, "field cast fp8->fp16: multiplicative (cast(x*y)==cast(x)*cast(y))");
        check(w32 == n, "field cast fp8->fp32: multiplicative");
        check(nA == n, "field cast fp32->fp8: narrowing is multiplicative");
        check(rt == n, "field cast: narrow(widen(x)) == x (round-trip identity)");
        // commutative diagram: widen direct == widen via an intermediate width
        {
            long ok = 0, m = 0;
            for(float u : xs)
            {
                F8 a{static_cast<fp8_e4m3>(u)};
                ok += (cast<float>(a) == cast<float>(cast<_Float16>(a))); // fp8->fp32 == fp8->fp16->fp32
                ++m;
            }
            check(ok == m, "field cast: fp8->fp32 == fp8->fp16->fp32 (widening composes)");
        }
        check(cast<_Float16>(F8{static_cast<fp8_e4m3>(1.0f)}) == F16{static_cast<_Float16>(1.0f)},
              "field cast fp8->fp16: cast(1) == 1");
        check(cast<float>(Alg{1.25f}) == Alg{1.25f}, "field cast: same-width is identity");
    }

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
        using H = Value<_Float16, Semantics::FPSanAlgebraicField, Conversions::Explicit>;
        H h1 = cast<_Float16>(Alg{1.5f});
        H h2 = cast<_Float16>(Alg{1.5f});
        check(h1 == h2, "alg: cross-width cast is deterministic");
    }

    // ---- 64-bit element types (double): the full algebra at n ~ 2^64 ----------
    // Exercises the 128-bit modular multiply, the overflow-safe modular add, the
    // overflow-free cbrt exponent, and -- for log/log2/log10 on the d ~ 2^31
    // channel -- the Pollard-rho discrete log.
    {
        using DFld = Value<double, Semantics::FPSanAlgebraicField, Conversions::Explicit>;
        check((DFld{2.0} + DFld{2.0}) == DFld{4.0}, "dbl field: 2+2 == 4");
        check((DFld{3.0} * DFld{3.0}) == DFld{9.0}, "dbl field: 3*3 == 9");
        {
            long ok = 0, n = 0;
            for(int i = 1; i <= 400; ++i, ++n)
                ok += ((DFld{(double)i} / DFld{(double)i}) == DFld{1.0});
            check(ok == n, "dbl field: x/x == 1");
        }
        check(sqrt(DFld{3.0} * DFld{5.0}) == sqrt(DFld{3.0}) * sqrt(DFld{5.0}),
              "dbl field: sqrt(x*y) == sqrt(x)*sqrt(y)");
        {
            DFld x{7.0}, c = cbrt(x);
            check(c * c * c == x, "dbl field: cbrt(x)^3 == x (perfect)");
        }

        using DExp = Value<double, Semantics::FPSanAlgebraicRingSophieGermain, Conversions::Explicit>;
        check(exp(DExp{0.0}) == DExp{1.0}, "dbl SG: exp(0) == 1");
        {
            long  ok = 0, n = 0;
            double xs[] = {0.5, 1.0, 1.5, 2.0, -1.0, 0.25, 3.0};
            for(double u : xs)
                for(double v : xs) { ok += (exp(DExp{u} + DExp{v}) == exp(DExp{u}) * exp(DExp{v})); ++n; }
            check(ok == n, "dbl SG: exp(a+b) == exp(a)*exp(b)");
        }
        {
            long  ok = 0, n = 0;
            double xs[] = {1.0, 2.0, 3.0, 5.0, 0.5, 1.5, 7.0};
            for(double u : xs)
                for(double v : xs) { ok += (log(DExp{u} * DExp{v}) == log(DExp{u}) + log(DExp{v})); ++n; }
            check(ok == n, "dbl SG: log(x*y) == log(x)+log(y) (Pollard-rho dlog)");
        }
        check(exp(log(exp(DExp{1.5}))) == exp(DExp{1.5}), "dbl SG: exp(log(exp v)) == exp v");
        check(exp2(DExp{2.0}) != exp(DExp{2.0}), "dbl SG: exp2 != exp (distinct base)");
        check(exp10(log10(exp10(DExp{1.5}))) == exp10(DExp{1.5}), "dbl SG: exp10(log10(exp10 v))");

        using DTrig = Value<double, Semantics::FPSanAlgebraicRingPythagorean, Conversions::Explicit>;
        check(cos(DTrig{0.0}) == DTrig{1.0}, "dbl Pyth: cos(0) == 1");
        check(sin(DTrig{0.0}) == DTrig{0.0}, "dbl Pyth: sin(0) == 0");
        {
            long  ok = 0, n = 0;
            double xs[] = {0.5, 1.0, 1.5, 2.0, 3.0, -1.0, 0.25};
            for(double u : xs)
                for(double v : xs)
                {
                    bool c1 = (cos(DTrig{u} + DTrig{v})
                               == cos(DTrig{u}) * cos(DTrig{v}) - sin(DTrig{u}) * sin(DTrig{v}));
                    bool c2 = (sin(DTrig{u} + DTrig{v})
                               == sin(DTrig{u}) * cos(DTrig{v}) + cos(DTrig{u}) * sin(DTrig{v}));
                    ok += (c1 && c2);
                    ++n;
                }
            check(ok == n, "dbl Pyth: sin/cos angle-addition");
        }
        check(cos(DTrig{1.3}) * cos(DTrig{1.3}) + sin(DTrig{1.3}) * sin(DTrig{1.3}) == DTrig{1.0},
              "dbl Pyth: cos^2 + sin^2 == 1");
        check(log(DTrig{2.0} * DTrig{3.0}) == log(DTrig{2.0}) + log(DTrig{3.0}),
              "dbl Pyth: log homomorphism still holds");
        // independent-prime twins also work at 64 bits
        using DFld2 = Value<double, Semantics::FPSanAlgebraicField2, Conversions::Explicit>;
        check((DFld2{2.0} + DFld2{2.0}) == DFld2{4.0}, "dbl field2: 2+2 == 4");
        check(DFld{0.5}.fpsan_payload() != DFld2{0.5}.fpsan_payload(),
              "dbl field/field2 use distinct moduli");
    }

    std::printf("passed %ld, failed %ld\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
