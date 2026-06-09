// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/algebraic.hpp
// ----------------------------------------------------------------------------
// The payload algebra for the *algebraic* FPSan semantics (see the study in
// bjacob/fpsan: algebraic-fpsan.md).  Where Semantics::FPSan scrambles the
// float bits into the free ring Z/2^w (mix.hpp's ring_*), the algebraic
// semantics carry the genuine residue
//
//     phi_n(m * 2^e) = (m mod n) * (2^-1)^(-e)   in  Z/nZ
//
// of the value's *exact* dyadic rational, for a per-element-width modulus n.
//
// This header is the ONE place the variants diverge.  Four variants:
//   * Field1 / Field2 : n = a prime just below 2^w.  Z/n is a field
//                       (division total, x/x == 1).  No exp homomorphism.
//   * Exp1   / Exp2   : n = p*d (Sophie-Germain pair, p == 2d+1), p,d prime.
//                       Carries exp(v) = g^(v mod d) so exp(a+b)=exp(a)exp(b);
//                       pays zero-divisors (~1/p+1/d) and an order-d exp image.
//
// Per the design: everything keys on the *scalar element* width.  Exp variants
// only make sense >= 8 bits (sub-byte types are matmul-only), so below 8 bits
// an Exp variant FALLS BACK to its matching Field prime (has_exp = false).
// Vector Values apply all of this LANE-WISE; vector width != element width.
//
// Casts between widths are NOT value-faithful here (a per-width modulus makes
// widening uncomputable from the narrow payload) -- they are a defined
// convention handled in cast.hpp, exactly as Triton-FPSan's resize is.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_ALGEBRAIC_HPP
#define FPSAN_DETAIL_ALGEBRAIC_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/traits.hpp"

#include <cstdint>

namespace fpsan
{
    namespace detail
    {
        using u64 = std::uint64_t;

        // The divergence point: which algebraic variant. (Mapped from the public
        // Semantics enum in value.hpp; kept separate so this header has no
        // dependency on value.hpp.)
        enum class AlgVariant
        {
            Field1,
            Field2,
            Exp1,
            Exp2
        };

        struct AlgModulus
        {
            u64  n       = 0; // modulus; residues in [0, n)
            u64  g       = 0; // exp generator (order d); unused if !has_exp
            u64  d       = 0; // exp exponent modulus; unused if !has_exp
            bool has_exp = false;
        };

        // ---- the constants table (the only per-(variant x width) data) ----------
        // Field primes leave >= 2 codes free (Inf, NaN sentinels). Exp pairs are
        // Sophie-Germain (p = 2d+1); g has order d in (Z/n)^*.  64-bit (double) is
        // not wired yet (needs 128-bit modular multiply); it static_asserts below.
        FPSAN_HOST_DEVICE constexpr u64 alg_field_prime(AlgVariant v, unsigned w)
        {
            const bool a = (v == AlgVariant::Field1 || v == AlgVariant::Exp1);
            switch(w)
            {
            case 4: return a ? 13u : 11u;
            case 6: return a ? 61u : 59u;
            case 8: return a ? 251u : 241u;
            case 16: return a ? 65521u : 65519u;
            case 32: return a ? 4294967291u : 4294967279u;
            default: return 0;
            }
        }
        FPSAN_HOST_DEVICE constexpr AlgModulus alg_exp_pair(AlgVariant /*v*/, unsigned w)
        {
            // (Only Exp1 wired with data; Exp2 reuses the same shapes at other
            // widths in this prototype -- a second independent pair per width is a
            // one-line table addition.)
            switch(w)
            {
            case 8: return {253u, 188u, 11u, true};
            case 16: return {64261u, 63188u, 179u, true};
            case 32: return {4274287111u, 4274009738u, 46229u, true};
            default: return {};
            }
        }

        FPSAN_HOST_DEVICE constexpr AlgModulus alg_modulus(AlgVariant v, unsigned w)
        {
            const bool is_exp = (v == AlgVariant::Exp1 || v == AlgVariant::Exp2);
            if(is_exp && w >= 8)
                return alg_exp_pair(v, w);
            // Field variant, or Exp below 8 bits -> field prime, no exp.
            return {alg_field_prime(v, w), 0u, 0u, false};
        }

        // ---- the per-Value configuration (analogous to MixConfig) ----------------
        struct AlgConfig
        {
            u64 n        = 0;
            u64 inv2     = 0; // 2^{-1} mod n
            u64 g        = 0;
            u64 d        = 0;
            u64 inf_code = 0; // = n
            u64 nan_code = 0; // = n + 1
            bool has_exp = false;
            // decoded float format of the element type:
            unsigned bit_width = 0;
            unsigned mant_bits = 0;
            u64      exp_max   = 0; // all-ones exponent field
            u64      mant_mask = 0;
            int      bias      = 0;
            bool     has_inf_nan = false;
        };

        template <class ElementType>
        FPSAN_HOST_DEVICE constexpr AlgConfig make_alg_config(AlgVariant v)
        {
            using T = fp_traits<ElementType>;
            static_assert(T::bit_width <= 32,
                          "fpsan algebraic: 64-bit element types not wired yet "
                          "(needs 128-bit modular multiply).");
            AlgConfig c;
            const AlgModulus m = alg_modulus(v, T::bit_width);
            c.n           = m.n;
            c.g           = m.g;
            c.d           = m.d;
            c.has_exp     = m.has_exp;
            c.inv2        = (m.n + 1) / 2; // inverse of 2 mod odd n
            c.inf_code    = m.n;
            c.nan_code    = m.n + 1;
            c.bit_width   = T::bit_width;
            c.mant_bits   = T::mantissa_bits;
            c.bias        = T::bias;
            c.mant_mask   = (u64{1} << T::mantissa_bits) - 1;
            c.exp_max     = (u64{1} << T::exponent_bits) - 1;
            c.has_inf_nan = true; // IEEE-style types; sub-byte specifics TBD
            return c;
        }

        // ---- scalar modular arithmetic (n < 2^32, so products fit u64) ----------
        FPSAN_HOST_DEVICE constexpr u64 alg_powmod(u64 b, u64 e, u64 n)
        {
            u64 r = 1 % n;
            b %= n;
            while(e)
            {
                if(e & 1)
                    r = (r * b) % n;
                b = (b * b) % n;
                e >>= 1;
            }
            return r;
        }
        // Returns the inverse, or n (an out-of-range sentinel) if a is a non-unit.
        FPSAN_HOST_DEVICE constexpr u64 alg_inv(u64 a, u64 n)
        {
            std::int64_t t = 0, newt = 1;
            std::int64_t r = (std::int64_t)n, newr = (std::int64_t)(a % n);
            while(newr != 0)
            {
                std::int64_t q = r / newr;
                std::int64_t tmp = 0;
                tmp = t - q * newt; t = newt; newt = tmp;
                tmp = r - q * newr; r = newr; newr = tmp;
            }
            if(r != 1)
                return n; // not invertible (zero-divisor)
            if(t < 0)
                t += (std::int64_t)n;
            return (u64)t;
        }

        // A cheap op-tagged scramble: the "free generator" for transcendentals
        // with no usable identity (and for exp in the Field / sub-byte cases).
        FPSAN_HOST_DEVICE constexpr u64 alg_token(u64 tag, u64 payload, u64 n)
        {
            u64 z = (payload + 1) * 0x9E3779B97F4A7C15ull + tag;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            return z % n;
        }

        // ---- scalar payload ops on residues (incl. the Inf/NaN projective rules) -
        // Codes: [0,n) finite; n = Inf; n+1 = NaN.
        FPSAN_HOST_DEVICE constexpr bool alg_is_inf(const AlgConfig& c, u64 p) { return p == c.inf_code; }
        FPSAN_HOST_DEVICE constexpr bool alg_is_nan(const AlgConfig& c, u64 p) { return p == c.nan_code; }
        FPSAN_HOST_DEVICE constexpr bool alg_is_fin(const AlgConfig& c, u64 p) { return p < c.n; }

        FPSAN_HOST_DEVICE constexpr u64 alg_neg1(const AlgConfig& c, u64 a)
        {
            if(!alg_is_fin(c, a))
                return a; // -Inf == Inf (unsigned), -NaN == NaN
            return a == 0 ? 0 : c.n - a;
        }
        FPSAN_HOST_DEVICE constexpr u64 alg_add1(const AlgConfig& c, u64 a, u64 b)
        {
            if(alg_is_nan(c, a) || alg_is_nan(c, b))
                return c.nan_code;
            if(alg_is_inf(c, a) || alg_is_inf(c, b))
                return (alg_is_inf(c, a) && alg_is_inf(c, b)) ? c.nan_code : c.inf_code;
            u64 s = a + b;
            return s >= c.n ? s - c.n : s;
        }
        FPSAN_HOST_DEVICE constexpr u64 alg_sub1(const AlgConfig& c, u64 a, u64 b)
        {
            return alg_add1(c, a, alg_neg1(c, b));
        }
        FPSAN_HOST_DEVICE constexpr u64 alg_mul1(const AlgConfig& c, u64 a, u64 b)
        {
            if(alg_is_nan(c, a) || alg_is_nan(c, b))
                return c.nan_code;
            const bool ai = alg_is_inf(c, a), bi = alg_is_inf(c, b);
            const bool az = (alg_is_fin(c, a) && a == 0), bz = (alg_is_fin(c, b) && b == 0);
            if(ai || bi)
                return (az || bz) ? c.nan_code : c.inf_code; // 0*Inf -> NaN
            return (a * b) % c.n;
        }
        FPSAN_HOST_DEVICE constexpr u64 alg_div1(const AlgConfig& c, u64 a, u64 b)
        {
            if(alg_is_nan(c, a) || alg_is_nan(c, b))
                return c.nan_code;
            const bool ai = alg_is_inf(c, a), bi = alg_is_inf(c, b);
            if(ai && bi)
                return c.nan_code; // Inf/Inf
            if(ai)
                return c.inf_code; // Inf/finite
            if(bi)
                return 0; // finite/Inf -> 0
            if(b == 0)
                return a == 0 ? c.nan_code : c.inf_code; // 0/0 -> NaN, x/0 -> Inf
            const u64 inv = alg_inv(b, c.n);
            if(inv == c.n)
                return c.nan_code; // zero-divisor (CRT variant) -> poison
            return (a * inv) % c.n;
        }
        FPSAN_HOST_DEVICE constexpr u64 alg_exp1(const AlgConfig& c, u64 a)
        {
            if(!alg_is_fin(c, a))
                return c.nan_code; // exp(Inf) ambiguous (unsigned), exp(NaN)=NaN
            if(c.has_exp)
                return alg_powmod(c.g, a % c.d, c.n); // g^(v mod d): the homomorphism
            return alg_token(/*tag "exp"*/ 0x657870ull, a, c.n);
        }

        // phi_n of a raw float bit-pattern of the element type (one lane).
        FPSAN_HOST_DEVICE constexpr u64 alg_embed1(const AlgConfig& c, u64 raw)
        {
            const u64 sign  = (raw >> (c.bit_width - 1)) & 1;
            const u64 expf  = (raw >> c.mant_bits) & c.exp_max;
            const u64 mantf = raw & c.mant_mask;
            if(c.has_inf_nan && expf == c.exp_max)
                return mantf == 0 ? c.inf_code : c.nan_code;
            u64 mag = 0;
            int e   = 0;
            if(expf == 0)
            {
                if(mantf == 0)
                    return 0; // +/-0 -> residue 0
                mag = mantf;
                e   = 1 - c.bias - (int)c.mant_bits;
            }
            else
            {
                mag = (u64{1} << c.mant_bits) | mantf;
                e   = (int)expf - c.bias - (int)c.mant_bits;
            }
            u64 r = mag % c.n;
            u64 pw = e >= 0 ? alg_powmod(2, (u64)e, c.n) : alg_powmod(c.inv2, (u64)(-e), c.n);
            r = (r * pw) % c.n;
            return sign ? (r == 0 ? 0 : c.n - r) : r;
        }

        // phi_n^{-1} is NOT well-defined (the residue does not determine the value),
        // so this is a best-effort, non-faithful decode used only by to_float() for
        // display: Inf/NaN map to the format's Inf/NaN bit patterns, and a finite
        // residue is returned as-is (a deterministic but meaningless bit pattern).
        // Algebraic Values are meant to be compared by payload, not unembedded.
        FPSAN_HOST_DEVICE constexpr u64 alg_unembed1(const AlgConfig& c, u64 p)
        {
            if(c.has_inf_nan && p == c.inf_code)
                return c.exp_max << c.mant_bits; // +Inf
            if(c.has_inf_nan && p == c.nan_code)
                return (c.exp_max << c.mant_bits) | 1; // NaN
            const u64 width_mask = (c.bit_width >= 64) ? ~u64{0} : ((u64{1} << c.bit_width) - 1);
            return p & width_mask;
        }

        // ---- vector wrappers: apply the scalar core lane-wise (cf. ring_div) -----
        template <class Bits, class Op>
        FPSAN_HOST_DEVICE constexpr Bits alg_lanewise1(Bits a, Op op)
        {
            if constexpr(!is_clang_vector_v<Bits>)
                return static_cast<Bits>(op((u64)a));
            else
            {
                using L = bits_lane_t<Bits>;
                Bits out{};
                for(unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
                    out[i] = static_cast<L>(op((u64)a[i]));
                return out;
            }
        }
        template <class Bits, class Op>
        FPSAN_HOST_DEVICE constexpr Bits alg_lanewise2(Bits a, Bits b, Op op)
        {
            if constexpr(!is_clang_vector_v<Bits>)
                return static_cast<Bits>(op((u64)a, (u64)b));
            else
            {
                using L = bits_lane_t<Bits>;
                Bits out{};
                for(unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
                    out[i] = static_cast<L>(op((u64)a[i], (u64)b[i]));
                return out;
            }
        }

        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_embed(const AlgConfig& c, Bits raw)
        {
            return alg_lanewise1(raw, [&](u64 x) { return alg_embed1(c, x); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_unembed(const AlgConfig& c, Bits p)
        {
            return alg_lanewise1(p, [&](u64 x) { return alg_unembed1(c, x); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_add(const AlgConfig& c, Bits a, Bits b)
        {
            return alg_lanewise2(a, b, [&](u64 x, u64 y) { return alg_add1(c, x, y); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_sub(const AlgConfig& c, Bits a, Bits b)
        {
            return alg_lanewise2(a, b, [&](u64 x, u64 y) { return alg_sub1(c, x, y); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_mul(const AlgConfig& c, Bits a, Bits b)
        {
            return alg_lanewise2(a, b, [&](u64 x, u64 y) { return alg_mul1(c, x, y); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_div(const AlgConfig& c, Bits a, Bits b)
        {
            return alg_lanewise2(a, b, [&](u64 x, u64 y) { return alg_div1(c, x, y); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_neg(const AlgConfig& c, Bits a)
        {
            return alg_lanewise1(a, [&](u64 x) { return alg_neg1(c, x); });
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_exp(const AlgConfig& c, Bits a)
        {
            return alg_lanewise1(a, [&](u64 x) { return alg_exp1(c, x); });
        }

        // Op-tagged free generator for transcendentals with no honored identity
        // (log, sqrt, and -- in this prototype -- exp2/sin/cos): deterministic,
        // op-distinct, faithful to algebraic independence (Schanuel).
        FPSAN_HOST_DEVICE constexpr u64 alg_tagged1(const AlgConfig& c, u64 a, u64 tag)
        {
            return alg_is_fin(c, a) ? alg_token(tag, a, c.n) : c.nan_code;
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_tagged(const AlgConfig& c, Bits a, u64 tag)
        {
            return alg_lanewise1(a, [&](u64 x) { return alg_tagged1(c, x, tag); });
        }
        // Binary tagged token (e.g. fmod): deterministic in both operands.
        FPSAN_HOST_DEVICE constexpr u64 alg_tagged2_1(const AlgConfig& c, u64 a, u64 b, u64 tag)
        {
            if(!alg_is_fin(c, a) || !alg_is_fin(c, b))
                return c.nan_code;
            return alg_token(tag, alg_token(tag, a, c.n) + b, c.n);
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits alg_tagged2(const AlgConfig& c, Bits a, Bits b, u64 tag)
        {
            return alg_lanewise2(a, b, [&](u64 x, u64 y) { return alg_tagged2_1(c, x, y, tag); });
        }

        // Cast convention between widths (NON-faithful by construction -- a
        // per-width modulus makes widening uncomputable; see algebraic-fpsan.md).
        // Deterministic and in-range: Inf/NaN map across, a finite residue maps
        // by reduction mod the destination modulus (identity for same width).
        FPSAN_HOST_DEVICE constexpr u64 alg_cast1(const AlgConfig& from, const AlgConfig& to, u64 p)
        {
            if(from.has_inf_nan && p == from.inf_code)
                return to.inf_code;
            if(from.has_inf_nan && p == from.nan_code)
                return to.nan_code;
            return p % to.n;
        }

    } // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_ALGEBRAIC_HPP
