# An algebraic alternative to FPSan: fingerprint the value, not the bits

This is a companion to [understanding-fpsan.md](understanding-fpsan.md), which
builds the Triton floating-point sanitizer **FPSan** up from scratch. That note
explains the problem and the design; this one explores a different choice at the
heart of it. A complete implementation of everything here ships in this library:
the payload algebra in
[`include/fpsan/detail/algebraic.hpp`](../include/fpsan/detail/algebraic.hpp),
wired into `Value<>` and exercised by
[`tests/algebraic_value_test.cpp`](../tests/algebraic_value_test.cpp).

## The problem, in one paragraph

You refactor a numerical kernel for speed — reassociate a sum, swap a `mul`+`add`
for an `fma`, reorder a reduction — and the output bits change. Did the refactor
change the *math*, or only the rounding error? FPSan answers this by running a
**parallel arithmetic** in which algebraically-equivalent expressions produce
bit-identical fingerprints, so the check is a single `memcmp`. It does that by
replacing each float with a scrambled integer "payload" and doing integer
arithmetic modulo `2^w`.

## The one idea

FPSan makes the leaves **opaque**: it scrambles each float's bits into something
that behaves like a free, unconstrained variable. The only equalities it then
sees are the ones the ring axioms force — associativity, commutativity,
distributivity. It deliberately does *not* see that `2 + 2` and `4` are the same
number, because its scramble of the bits of `2.0` has nothing to do with its
scramble of the bits of `4.0`. Call this the **free model**.

This note goes the other way. It turns out there's an **algebraic** alternative to
the scramble: a way to encode a float by its *actual numeric value* — still a
single fixed-width payload — that nonetheless **respects arithmetic exactly**. The
payload of `a + b` is the sum of the two payloads, and the payload of `a * b` is
their product, on the nose, with no rounding. That a finite, fixed-width encoding
can do this at all — stay perfectly faithful to `+` and `*` while tracking a
*value* rather than scrambling bits — is the one nontrivial piece of mathematics
the whole approach rests on. The construction turns out to be a reduction modulo a
prime; the next section makes it precise, and for now it's enough to know such an
encoding exists. The payoff is immediate: the fingerprint now sees the value
coincidences the scramble is blind to — `2 + 2` really equals `4`, `x / x` really
is `1`. Call this the **value model**. That single change — free model → value
model — is the whole story; every property below follows from it.

Neither is strictly better (that tension is the subject of a later section). The
value model is the faithful one for the *algebraic* core of a kernel; FPSan's
freeness is the faithful one for *transcendentals*. The interesting designs are
hybrids.

## The map: a float is a fraction, so reduce it mod a prime

A finite `float32` is an integer times a power of two: `x = m * 2^e`, with the
signed significand `|m| < 2^24` and exponent `e` in `[-149, 104]`. So every
finite float is a **dyadic rational** — a fraction whose denominator is a power
of two. Write that set `Z[1/2]` ("the integers with 1/2 thrown in").

Pick an odd prime `p`. Because `p` is odd, `2` has a multiplicative inverse mod
`p`, so we can reduce any dyadic rational mod `p`:

```
phi_p(m * 2^e)  =  m * (2^-1)^(-e)   (mod p)
```

The one property that matters: `phi_p` is a **ring homomorphism** — a map that
respects `+`, `-`, and `*`:

```
phi_p(x + y) = phi_p(x) + phi_p(y)        phi_p(x * y) = phi_p(x) * phi_p(y)
```

It is forced by `1 → 1` and `1/2 → (2^-1)`, and it is exact: no rounding. This is
exactly what FPSan's bit-scramble refuses to be.

Division comes along for free. The integers mod a prime form a **field** (written
`F_p`): every nonzero residue has an inverse. So even a quotient that isn't a
dyadic rational — `1/3`, say — has a well-defined residue, namely the inverse of
`3` mod `p`. (For the mathematically inclined: the natural domain is the
localization `Z_(p)`, and `phi_p` is its residue map onto `F_p`; that the target
is a *field* is what makes division total.)

**The model.** For each value, carry its residue `r = phi_p(value)` in `F_p`. Do
`+`, `-`, `*`, `/` as ordinary field arithmetic on residues. The residue of an
expression is then `phi_p` of its *exact, un-rounded* value, so two expressions
with the same exact value always get the same residue — automatically, for
**every** identity true over the rationals, not just the ones we remembered to
build in.

We stay with **compact, per-width encodings** throughout: a 32-bit value carries
a 32-bit residue, a 16-bit value a 16-bit residue, and so on. Two flavors of
modulus appear later — a single prime, and a composite `p*d` that buys an exact
exponential.

## Free model vs. value model: the dichotomy cuts both ways

This is worth stating sharply, because it decides everything.

- **FPSan is free.** Inputs become algebraically-independent generators; two
  expressions match only when they are equal *from the axioms alone*. So `2 + 2`
  does **not** match `4`, `x * (1/x)` does **not** match `1`, `a + a` does **not**
  match `2a`. The model is blind to the actual values.
- **`phi_p` is value.** Leaves are their true residues; two expressions match when
  they have the **same exact value** (mod `p`). So `2 + 2 == 4`, `x / x == 1`,
  `a + a == 2a`, `(a+b)^2 == a^2 + 2ab + b^2`, constant-folding,
  common-subexpression elimination — all match.

Which one is *faithful* flips depending on what you are computing:

- On the **algebraic part** (polynomials and rational functions of the inputs),
  the real values genuinely have all these relations, so the value model is
  faithful and the free model over-reports — it would flag a value-preserving
  rewrite as a difference.
- On **transcendentals** it flips (see the transcendentals section): values like
  `exp(a)`, `log(b)`, `sin(c)` at independent arguments really are algebraically
  independent, so *freeness* is the faithful choice there.

So the honest framing: the value model is right for the algebraic core, FPSan's
freeness is right for the transcendental parts, and the useful designs combine
them.

## Collisions: what we give up with a finite modulus

A prime below `2^32` can't injectively encode all `~2^32` floats, so distinct
values sometimes share a residue. Three separate questions hide here.

**Zero is safe — the collision that would hurt never happens.** `phi_n(m*2^e)`
is `0` only when `n` divides `m`, and since `|m| < 2^24` while the modulus is far
larger (`~2^32`), that forces `m = 0`. So the *only* finite float with residue
`0` is `±0.0`. Nothing nonzero ever silently becomes zero — the design constraint
that keeps the zero/infinity story clean.

**Leaf collisions (two different floats, same residue) are common but mostly
harmless.** Dropping `~2^32` floats into `~2^32` residues, a constant fraction
share a residue with some other float. This only bites a *literal-substitution*
bug — swapping input `x` for a different `y` that happens to have the same
residue. For the main use case, comparing **re-orderings of the same computation
on the same inputs**, both sides use the same leaves, so leaf collisions are
irrelevant.

**A false match between two genuinely different expressions is just
fingerprinting at a random point.** Two different expressions on shared inputs
collide only when `p` divides the numerator of `value_A - value_B`. That
difference is a dyadic rational whose numerator is huge but bounded (below about
`2^278`), so it has at most ~9 prime factors above `2^32`. With tens of millions
of primes to choose from in `[2^31, 2^32]`, a **randomly chosen** prime divides
it with probability under `2^-23`. This is textbook Freivalds / Schwartz–Zippel /
Rabin fingerprinting: test an identity by evaluating at a random point (here,
reducing mod a random prime). Two consequences:

- Per comparison the false-match rate is roughly `2^-24` to `2^-32` at 32 bits.
  The lever for more margin, while staying compact, is **repetition**: run again
  with a different prime and the rates multiply (two runs ≈ `2^-48` to `2^-64`).
  For a test tool, spending *time* (extra runs) beats spending *space* (a wider
  word), and it keeps the instrumented program's footprint unchanged.
- Unlike FPSan's *fixed* scramble — whose colliding pairs are the same on every
  run — choosing `p` at random makes collisions **non-repeatable**: there is no
  bad pair that fools you every time.

Measurement confirms the rate: over 300k random floats, observed leaf
collisions track the birthday estimate to within noise.

## Why we can't scramble on top

FPSan scrambles to make its leaves free. Could we keep `phi_p`'s homomorphism
*and* scramble on top — `value → sigma(phi_p(value))`? Only if `sigma` is a ring
automorphism of `F_p`, or the homomorphism (the whole point) breaks. But `F_p`
has only the trivial automorphism. **So you cannot scramble inside `F_p` at all**:
homomorphism and value-scrambling are mutually exclusive, not a tunable dial.

What plays the role of FPSan's hash seed instead is the **choice of `p`**. The
randomness moves from "scramble the values" to "pick a random evaluation point,"
which is the more principled place for it and is what gives the non-repeatable
collisions above. The one thing genuinely lost is diffusion as a *debugging* aid:
FPSan's payloads look random, so a single flipped bit produces a wildly different
payload that's easy to eyeball. `phi_p` residues of nearby computations can be
related. For a detector (we compare for equality) that's fine; for forensics
FPSan's avalanche is nicer.

## Arithmetic, division, casts — and what's lost

This is where the value model shines.

- **`+`, `-`, `*`** are exact homomorphic images; every ring identity holds for
  free.
- **`/`**: with a **prime** modulus, `F_p` is a field, so division is total
  except by residue `0`, and `x / x == 1` holds **always** (any nonzero float).
  FPSan's `Z/2^w` has zero-divisors and only a parity-preserving "inverse," so
  `x / x == 1` holds there only for odd payloads. The field is a clean win.
  Dividing by residue `0` happens only for a genuine `±0.0` → honest infinity.
- **Casts / mixed precision.** A *value-faithful* cast — one that recovers the
  wide residue of the same value — is impossible across distinct per-width primes:
  it would have to commute with `+` and `×` at once, i.e. be a ring homomorphism
  between fields of different characteristic, which doesn't exist (the 16-bit
  residue simply can't determine the 32-bit one). But a **multiplicative** cast
  *is* possible, and it's the more useful target anyway: pick the Field primes so
  the `p_w − 1` form a *coprime tower* (each width's cofactor coprime to the ones
  below), and then every widening and narrowing cast along the chain is a
  multiplicative homomorphism — `cast(x·y) == cast(x)·cast(y)` — and they form a
  **commutative diagram**: widening composes, narrowing composes, and
  `narrow(widen(x)) == x` exactly (so an `f16 → f32 → f16` round trip is the
  identity, as it should be). The catch is the same as in §collisions: it can't be
  value-faithful, so cast-up values and native-wide values still don't compare
  equal across `+`. Off the chain (`fp6`, the composite Sophie Germain / Pythagorean moduli, or equal
  width) it's the plain reduce-mod convention. (See the casts section of the
  scorecard.)
- **Subnormals** are dyadic rationals like any other and need no special case.

What's **lost**, and it's fundamental: **order** — `min`, `max`, `abs`,
comparisons. `F_p` is not an ordered field; the residue throws away magnitude and
sign-as-order. (FPSan loses order too, but it fakes `min`/`max` by ordering the
*scrambled payload* — associative and reassociation-invariant, but it can pick
the wrong element.) Here, order-dependent ops are simply **out of scope**; they
need a side channel that carries the actual float.

**Cost.** Arithmetic mod `n` is costlier than FPSan's free wraparound mod `2^w` —
one reduction per op (a multiply-high, or a couple of adds for a pseudo-Mersenne
prime). We pay it on purpose: footprint fidelity matters more than per-op speed
for a sanitizer.

## Roots are algebraic, not transcendental

`sqrt` and `cbrt` look like they belong with `exp`/`log`/`sin`, but they don't:
they're **algebraic** functions — `sqrt(x)` satisfies `sqrt(x)² = x`, `cbrt(x)`
satisfies `cbrt(x)³ = x` — so the value model can honor genuine relations among
their values, unlike a true transcendental. And honoring them is cheap: a root is
just a fixed-exponent **power map** `x ↦ x^e`, which is automatically
multiplicative.

- **`sqrt`** uses `e = (p+1)/4` (for `p ≡ 3 mod 4`). It is always multiplicative —
  `sqrt(x·y) == sqrt(x)·sqrt(y)` for every input — and `rsqrt == 1/sqrt` is the
  consistent reciprocal. The defining round trip `sqrt(x)² == x` holds on the
  *square* residues, exactly half of a prime field, and gives `−x` on the other
  half. Half is the best a homomorphic square root can do — squaring is
  two-to-one, so no power map reaches more.
- **`cbrt`** is *perfect* where `3` is coprime to the group order (choose
  `p ≡ 2 mod 3`): then `e = 3^{-1} mod (p−1)` and `cbrt(x)³ == x` for **all** `x`,
  multiplicatively. The asymmetry with `sqrt` is only that `2` always divides
  `p−1` while `3` need not. More generally any rational power `x^(a/b)` with `b`
  coprime to `p−1` is an exact multiplicative map; `sqrt` (`b = 2`) is the one
  perpetually-imperfect case.

Field and Sophie Germain get both; Pythagorean gets `sqrt` but not `cbrt` (its `p = 4d+1` forces
`3 | p−1`). This is a place the value model is strictly richer than FPSan, which
keeps `sqrt`/`cbrt` as opaque tokens. Genuinely transcendental functions stay
tokens — next.

## Transcendentals: `exp` is recoverable; some functions stay free

`exp`, `log`, `sin`, `cos`, … are not algebraic functions (unlike the roots
above), so the homomorphism says nothing about them directly. The question is
whether we can *build* an `exp` with the right law `exp(a+b) == exp(a)*exp(b)`.

**As a function of the mod-`p` residue: no.** Such an `exp` would be a map from
`(F_p, +)` to `(F_p-nonzero, *)`, and because the additive group has size `p`
while the multiplicative group has size `p-1` (coprime), the only such map is
trivial. So a single-prime variant cannot honor the exponential law — its `exp`
is a tagged hash token, and the law duly holds `0/50000` times in measurement.
FPSan escapes this only because its ring `Z/2^w` has additive and
multiplicative structure that share the prime `2`.

**As a function of a second, mod-`d` residue: yes.** Pick an element `g` of odd
order `d` and set `exp(v) = g^(v mod d)`. This satisfies the law exactly — but it
depends on `v mod d`, not `v mod p`, so we have to *carry* `v mod d`.

**The Chinese Remainder Theorem keeps that compact.** Carrying both `v mod p` and
`v mod d` is the same as carrying `v mod n` for `n = p*d` — still one residue, one
word. Choose `p ≡ 1 (mod d)` and `g` of order `d`; then `v mod d` reads straight
off the single residue and `exp(v) = g^(v mod d)`. The composite
variant honors `exp(a+b) == exp(a)*exp(b)` exactly (`50000/50000` in measurement). The cost (next
section): a composite modulus has a few **zero-divisors**, and `exp`'s image is
only the small order-`d` subgroup, so `exp` outputs collide more often.

**`log` is the exact dual of `exp`.** Where `exp` maps into the *multiplicative*
order-`d` subgroup, `log` maps into the *additive* order-`d` subgroup
`{0, n/d, 2n/d, …}` (which is closed under addition and behaves like the integers
mod `d`):

```
log(r) = (n/d) * dlog_g(r's order-d component)
```

so `log(x*y) == log(x) + log(y)` holds exactly and `log` inverts `exp`. The only
extra cost over `exp` is a discrete logarithm — an O(d) scan for the small `d` of
the ≤32-bit widths, and Pollard's rho (O(√d), O(1) memory) once `d ≈ 2^31` at 64
bits, where the scan's ~3 billion steps would be hopeless. Both return the same
unique log; `exp` itself stays a cheap O(log d) power map at every width.

**`exp2`/`exp10` and `log2`/`log10` ride the same channel by a base change.** The
whole family `exp_b(v) = g^(K_b*v mod d)` is, for each fixed unit `K_b`, again a
valid exponential, with `log_b` its inverse (divide the discrete log by `K_b mod
d`). The "right" constants relating the bases (`log2 e`, `ln 10`, …) are
irrational and so have no residue — so each `K_b` is just a fixed pseudo-random
unit mod `d`, exactly the role Triton's `rcpLog2` magic constant plays. Each base
keeps its own law and inverse; no numeric relation *between* the bases is claimed
(and the bases `e`, `2`, `10` are distinct fingerprints). Because the order-`d`
subgroup is unique, necessarily `exp_b(x) = exp(x)^{K_b}` — Triton has the
identical relation between its `exp` and `exp2`. This is strictly more than
Triton, which has genuine `exp`/`exp2` but leaves `log`/`log2` (and `exp10`,
`log10`) opaque tokens. (Aside: ISO C/C++ `<cmath>` has `exp2`, `log2`, `log10` —
but no `exp10`; we still model it, since GPU libdevice provides it.)

**What `exp_b(1)` can be — and the catch that it is *never* `b`.** An exp-style
homomorphism is fixed by `exp(1)`, and well-definedness on `Z/n` forces
`exp(1)ⁿ = exp(n) = 1`. So `exp(1)` must lie in the `n`-torsion of the units,
`U[n] = { u : uⁿ = 1 }` — which is exactly the order-`d` subgroup `⟨g⟩` (it has
`gcd(n, λ(n)) = d` elements for `n = p·d`; for a prime field it collapses to
`{1}`, the very reason a single prime has no exp). Two consequences:

- **Good news — the bases are genuinely distinct.** `⟨g⟩` is cyclic of *prime*
  order `d`, so all `d−1` non-identity elements are generators; `exp`/`exp2`/`exp10`
  pick three distinct ones (`g`, `g^{K₂}`, `g^{K₁₀}`), so the three bases are
  different fingerprints — with one unavoidable exception, the Pythagorean variant at fp8
  (`d = 3`), where only `{g, g²}` are non-trivial, leaving room for two bases, so
  `exp2(1) = exp10(1)` there. (Concretely, Exp1 at fp8 has `d = 11` and
  `U[n] = ⟨188⟩ = {1, 188, 177, 133, 210, 12, 232, 100, 78, 243, 144}` mod 253,
  with `exp(1)=188`, `exp2(1)=243`, `exp10(1)=133` — three of those eleven.)
- **The catch — `exp_b(1) ≠ b`, fundamentally.** The natural numeric law
  `exp_b(1) = b` (e.g. `exp2(1) = 2`) would require `b ∈ U[n]`, i.e. `ord_n(b) | n`.
  But `ord_n(b) | n` has **no solution for `n > 1`**: take the smallest prime `q | n`;
  then `ord_q(b)` divides both `n` (so its prime factors are all `≥ q`) and `q−1`
  (so they're all `< q`), forcing `ord_q(b) = 1`, i.e. `q | b−1` — impossible for
  `b = 2`, and for any small base (our smallest prime factor is `d ≈ √n`, far above
  `b−1`). So `exp_b(1)` is always an *opaque* element of `⟨g⟩`, never the literal
  base — the same reason `exp(1) ≠ e` (and `e` is not even representable). Numeric
  fidelity and the homomorphism law are mutually exclusive; we keep the law.

**`sin` / `cos` need a genuine rotation**, and that needs the "circle group" to
have order divisible by `d`. That happens when `p ≡ 1 (mod 4)` (circle order
`p-1`), not for the Sophie Germain primes `p = 2d+1` used for `exp` (those are
`≡ 3 (mod 4)`). So `sin`/`cos` live in a **dedicated variant** built on `p = 4d+1`
primes: with `i` a square root of `-1` mod `p` and `w` an order-`d` element of
`(Z/n)[i]`,

```
cos(x) = Re(w^(x mod d))        sin(x) = Im(w^(x mod d))
```

The angle-addition formulas then hold exactly and `cos^2 + sin^2 == 1`. That
variant also keeps `exp` and `log` (still `d | p-1`), at the cost of a roughly
`√2`-smaller `d` — i.e. somewhat more collisions.

The genuinely transcendental ones — `erf`, `tanh`, `lgamma`, … — have no usable
identity and stay **hash tokens**. That isn't a fallback, it's *correct*: by
Lindemann–Weierstrass / Schanuel, transcendental values at distinct algebraic
arguments are algebraically independent, so a fresh free generator per call is the
faithful model.

| fragment | true-value structure | model here |
|---|---|---|
| algebraic (`+ - * /`) | richly related (`2+2 == 4`) | **value**: the `phi_n` residue |
| `sqrt`, `cbrt`, `x^(a/b)` | `sqrt(x)² = x`, etc. | **structured**: power map `x^e` (algebraic, not free) |
| `exp` | `exp(a+b) == exp(a)*exp(b)` | **structured**: `g^(v mod d)` (Sophie Germain / Pythagorean) |
| `log` | `log(x*y) == log x + log y` | **structured**: discrete log (Sophie Germain / Pythagorean) |
| `exp2`/`exp10`, `log2`/`log10` | base-2 & base-10 analogues | **structured**: base change on the same channel |
| `sin`, `cos` | angle addition | **structured**: a rotation (Pythagorean variant) |
| `erf`, `tanh`, … | independent | **free**: a hash token |

## Zero, infinity, NaN, signs

Finite floats live in `Z[1/2]`; `±∞` and NaN don't. Extend the target to the
**projective line plus an absorbing symbol** — the residues, plus one point at
infinity, plus NaN:

- finite nonzero → its residue; `±0.0 → 0`; `±∞ → ∞`; NaN → NaN.
- The point at infinity gives `x / 0 == ∞` for nonzero `x` for free, plus the
  IEEE-shaped rules `∞ + finite == ∞` and `∞ * x == ∞` (`x ≠ 0`). NaN is
  absorbing and is produced by the indeterminate forms `∞ + ∞`, `∞ - ∞`,
  `0 * ∞`, `0 / 0`, `∞ / ∞`. Because our `∞` is **unsigned**, even `∞ + ∞` is
  indeterminate (IEEE resolves it using the sign we dropped), so the model emits
  NaN a touch more freely than IEEE — but only at the already-non-finite fringe.

The "zero is hit only by true zero" fact is what makes this principled: `1/x == ∞`
exactly when `x` is a genuine zero, never because some nonzero float collided onto
`0`. So `0` and `∞` are honest — a **better** infinity/NaN story than FPSan, which
models none of it. What's *not* modeled (these compact variants carry no sign
bit): signed zero and signed infinity (`+0.0` and `-0.0` both map to `0`; `±∞` to
one `∞`). Negation is otherwise exact on finite values.

## The three compact variants

Everything above instantiates as one of three designs. All carry a single
per-width residue plus the infinity/NaN extension; they differ only in the
modulus.

- **Field** (`FPSanAlgebraicField`) — the only variant that is a genuine *field*:
  a single prime `p ≡ 11 (mod 12)` just below `2^w`. The residues form a field:
  division total, `x / x == 1` always, no zero-divisors, best collision
  margin. It also gets multiplicative `sqrt`/`cbrt` (the `11 mod 12` choice) and a
  commutative diagram of multiplicative casts across the width tower. No exp/log/
  trig channel — those are hash tokens.
- **Sophie Germain ring** (`FPSanAlgebraicRingSophieGermain`) — a *ring*, not a
  field: composite `n = p*d` with `p = 2d+1` (a Sophie Germain pair). Buys the
  exact `exp`/`exp2`/`log`/`log2` homomorphisms. Costs: a few **zero-divisors**
  (rate ≈ `1/p + 1/d`; dividing by one poisons to NaN), and `exp`/`log` outputs
  collide at ≈ `1/d` because their image is only the order-`d` subgroup.
- **Pythagorean ring** (`FPSanAlgebraicRingPythagorean`) — also a *ring*: composite
  `n = p*d` with `p = 4d+1`. Adds genuine `sin`/`cos` on top
  of `exp`/`log`, all on the same `d`-channel, at a roughly `√2`-smaller `d` (so
  somewhat more collisions). Opt in for trig-heavy code.

(Using two distinct primes `p*d` is better than `p^2`: same trick, but `p^2`
introduces nilpotents, which are algebraically uglier and less value-faithful.)

All variants share the **hybrid scheme**: carry `phi_n(exact value)` for
`+ - * /`, the algebraic roots, and the multiplicative casts (honest `0`/`∞`/NaN);
at a transcendental call, either apply the structured map (`exp` etc., where the
variant supports it) or mint a fresh hash token. The implementation in
[`include/fpsan/detail/algebraic.hpp`](../include/fpsan/detail/algebraic.hpp)
provides all three variants (each with an independent-prime twin), and
[`tests/algebraic_value_test.cpp`](../tests/algebraic_value_test.cpp) checks the
homomorphism, ring-law, infinity/NaN, and exp/log/trig properties across every
variant and width; the collision and zero-divisor rates match those quoted above.

**The one genuinely open limitation is order** — `min`, `max`, `abs`,
comparisons. No finite ring has a compatible total order, and no sign or exponent
trick recovers one; this is shared with FPSan, which only fakes it. A side channel
(the actual float, or a tag) is the only route.

## A note on rounding

Both systems carry *exact* model values and never round, so both answer "are these
equal up to exact algebra?" and are deliberately **invariant to reassociation and
to rounding order**. The sanitizer flags rewrites that change the *exact* value,
not ones that merely re-round — that's the feature that lets a reduction in two
orders compare equal, and it's the same boundary FPSan draws.

A corollary that surprises people: **overflow to infinity is not modeled.** The
model carries no magnitude, so a sum of finite values is always a finite residue,
never `∞`. The only source of `∞` is division by a true zero. So `1/(huge sum)` is
an ordinary finite inverse here, where IEEE would overflow to `∞` and then give
`0` — both are deliberate consequences of abstracting magnitude and rounding away.

## Scorecard: FPSan vs. the three algebraic variants

A property-by-property comparison of the four shipping semantics. Any FPSan-style
sanitizer replaces float ops with a **deterministic, reassociation-invariant
fingerprint**: two runs that should agree get identical fingerprints, and a
mismatch pinpoints where the numerics diverged. Reassociation-invariance is just
associativity plus commutativity (below); the variants differ in which *further*
identities the fingerprint keeps. The Triton column was read from Triton's
`FpSanitizer.cpp`; the rest from the implementation in this repository.

Columns: **Triton** = `FPSanLikeTriton` (the free model) · **Field** =
`FPSanAlgebraicField` · **Sophie Germain** = `FPSanAlgebraicRingSophieGermain` ·
**Pythagorean** = `FPSanAlgebraicRingPythagorean`. Only **Field** is a genuine
field; the other two are *rings* — their modulus is a product `n = p·d` of two
primes, so they carry zero-divisors and `x/x == 1` can fail. The names tag the
class of the prime `p` (Sophie Germain/safe; Pythagorean, i.e. `p ≡ 1 mod 4`) and
the channel it unlocks — not a claim that the whole ring is a field. Markers:
✅ holds exactly ·
❌ does not hold · 🟡 holds with a caveat — often annotated with the pass rate
(`½`, `≈1`, `¼`, …); see notes.

| property | Triton | Field | Sophie Germain | Pythagorean |
|---|---|---|---|---|
| **— model —** | | | | |
| leaf encoding | scramble of the IEEE bits | value residue φₙ | value residue φₙ | value residue φₙ |
| ring | `Z/2^w` (w = bit width) | `F_p`, p prime (a **field**) | `Z/(p·d)`, p = 2d+1 | `Z/(p·d)`, p = 4d+1 |
| deterministic, platform-independent | ✅ | ✅ | ✅ | ✅ |
| **— ring axioms and exact value relations —** | | | | |
| `0 + x == x`, `1 * x == x` | ✅ ① | ✅ | ✅ | ✅ |
| commutativity, associativity, distributivity, `x − x == 0` | ✅ | ✅ | ✅ | ✅ |
| the leaf **encoding** is itself a ring homomorphism (not just the ops) | ❌ | ✅ | ✅ | ✅ |
| …so any identity of the exact, un-rounded values holds of the fingerprints | ❌ | ✅ | ✅ | ✅ |
| …constant folding: `2+2 == 4`, `2*3 == 6`, `0.5+0.5 == 1` | ❌ | ✅ | ✅ | ✅ |
| …symbolic: `x+x == 2x`, `(x+1)(x−1) == x²−1`, `(a+b)² == a²+2ab+b²` | ❌ ② | ✅ | ✅ | ✅ |
| …common-subexpression / value-preserving refactors compare equal | ❌ | ✅ | ✅ | ✅ |
| **— division & field structure —** | | | | |
| `x / x == 1` for every `x ≠ 0` | ❌ ½ ③ | ✅ | 🟡 ≈1 ④ | 🟡 ≈1 ④ |
| `(a / b) * b == a` (b a unit) | ❌ ½ ③ | ✅ | 🟡 ≈1 ④ | 🟡 ≈1 ④ |
| no zero-divisors (`a*b==0` ⟹ `a==0` or `b==0`) | ❌ ~½ ⑤ | ✅ | 🟡 ~2⁻¹⁵ ⑤ | 🟡 ~2⁻¹⁵ ⑤ |
| **— casts across widths —** | | | | |
| casts multiplicative: `cast<T>(x·y) == cast<T>(x)·cast<T>(y)` | ❌ | ✅ ⑪ | ❌ ⑯ | ❌ ⑯ |
| casts compose: `cast<T>(cast<U>(x)) == cast<T>(x)` | ✅ ⑪ | ✅ ⑪ | ❌ ⑯ | ❌ ⑯ |
| **— infinity & NaN —** | | | | |
| `1/0 = ∞` | ❌ ⑧ | ✅ | ✅ | ✅ |
| `1 / ∞ == 0` | ❌ ⑧ | ✅ | ✅ | ✅ |
| `x + ∞ == ∞`, `x * ∞ == ∞` (finite `x ≠ 0`) | ❌ ⑧ | ✅ | ✅ | ✅ |
| indeterminates `∞±∞`, `0*∞`, `∞/∞`, `0/0` → `NaN` | ❌ ⑧ | ✅ | ✅ | ✅ |
| `NaN` is absorbing (`NaN ∘ x == NaN`) | ❌ ⑧ | ✅ | ✅ | ✅ |
| **— algebraic functions: roots (NOT transcendental) —** | | | | |
| `sqrt(x·y) == sqrt(x)·sqrt(y)`, `rsqrt == 1/sqrt` | ❌ | ✅ | ✅ | ✅ |
| `sqrt(x)² == x` (the square residues only) | ❌ | 🟡 ½ ⑭ | 🟡 ¼ ⑭ | 🟡 ⅛ ⑭ |
| `cbrt(x·y)==cbrt(x)·cbrt(y)` and `cbrt(x)³ == x` (perfect) | ❌ | ✅ | ✅ | ❌ ⑮ |
| how `sqrt` realized | 🟡 token | ✅ power map `x^e` | ✅ power map | ✅ power map |
| how `rsqrt` realized | 🟡 token | ✅ power map `x^-e` | ✅ power map | ✅ power map |
| how `cbrt` realized | 🟡 token | ✅ power map `x^e` | ✅ power map | 🟡 token ⑮ |
| **— transcendental functions —** | | | | |
| `exp(a+b) == exp(a)*exp(b)` | ✅ | ❌ | ✅ | ✅ |
| `exp2(a+b) == exp2(a)*exp2(b)` | ✅ | ❌ | ✅ | ✅ |
| `exp10(a+b) == exp10(a)*exp10(b)` | ❌ ⑥ | ❌ | ✅ | ✅ |
| `log(x*y) == log(x)+log(y)` | ❌ ⑥ | ❌ | ✅ | ✅ |
| `log2(x*y) == log2(x)+log2(y)` | ❌ ⑥ | ❌ | ✅ | ✅ |
| `log10(x*y) == log10(x)+log10(y)` | ❌ ⑥ | ❌ | ✅ | ✅ |
| `exp(log(exp v)) == exp v` (log inverts exp, all bases) | ❌ ⑥ | ❌ | ✅ | ✅ |
| `cos(a+b) == cos a·cos b − sin a·sin b` | ✅ ⑦ | ❌ | ❌ | ✅ |
| `cos²x + sin²x == 1` | ✅ ⑦ | ❌ | ❌ | ✅ |
| how `exp`/`exp2`/`exp10` realized | ✅ exp,exp2 modexp; 🟡 exp10 token ⑥ | 🟡 token ⑩ | ✅ `g^(K·x mod d)` | ✅ `g^(K·x mod d)` |
| how `log`/`log2`/`log10` realized | 🟡 token | 🟡 token | ✅ discrete log in ⟨g⟩ | ✅ discrete log in ⟨g⟩ |
| how `cos`/`sin` realized | ✅ (3,4,5) rotor | 🟡 token | 🟡 token | ✅ order-d rotor in `(Z/n)[i]` |
| `erf`, `tanh`, `floor`, `ceil`, … | 🟡 token | 🟡 token | 🟡 token | 🟡 token |
| `exp`/`exp2`/`exp10` image-collision rate | ✅ ~2⁻ʷ (large image) | ✅ ~2⁻ʷ (token) | 🟡 ~1/d ≈ 1/√(2ʷ) | 🟡 ~1/d ≈ 1/√(2ʷ) |
| **— miscellaneous —** | | | | |
| `min`/`max`/comparisons follow the IEEE numeric order | ❌ ⑨ | ❌ ⑨ | ❌ ⑨ | ❌ ⑨ |
| subnormals | no special case ⑫ | no special case ⑫ | no special case ⑫ | no special case ⑫ |
| footprint | = the value (w bits) | = the value (w bits) | = the value (w bits) | = the value (w bits) |
| leaf-collision rate | ~2⁻ʷ | ~2⁻ʷ | ~2⁻ʷ | ~2⁻ʷ |
| collisions re-rollable: a fresh prime gives independent blind spots | ❌ ⑬ | ✅ ⑬ | ✅ ⑬ | ✅ ⑬ |
| per-op cost | mask to w bits | mod-p reduce | mod-n reduce (+ dlog) | mod-n reduce (+ dlog) |

**Notes.**
① Triton's leaf scramble is *tuned* so `embed(0.0)=0` and `embed(1.0)=1`; that is
why the bare `0+x`, `1*x`, and `x−x` identities survive even though the encoding
otherwise ignores values.
② `x+x = 2·embed(x)` but `2*x = embed(2.0)·embed(x)`, and `embed(2.0) ≠ 2` under
the scramble — so the two sides disagree.
③ Triton divides by the ring inverse of `(denominator | 1)`, forcing the divisor
odd: `x/x == 1` for odd payloads, not for the ~50% that are even. Deterministic
either way.
④ The Sophie Germain / Pythagorean moduli are composite (`n = p·d`), so a residue is a zero-divisor at
rate ≈ `1/p + 1/d` ≈ `2⁻¹⁵` (32-bit); `x/x` on such a value poisons to `NaN`, but
on every *unit* — all but that `~2⁻¹⁵` — `x/x == 1` holds. So the `≈1` pass rate.
⑤ Only `FPSanAlgebraicField` is a field — no zero-divisors at all. The composite
Sophie Germain / Pythagorean rings have a few (residues divisible by `p` or `d`, rate `≈ 2⁻¹⁵`), the
price of the exp/log channel, so the property *nearly* holds. The contrast among
the columns is dramatic: that `~2⁻¹⁵` is five orders gentler than Triton's
`Z/2^w`, where `~½` the ring are zero-divisors (every even payload) — which is why
Triton's `x/x == 1` is barely better than a coin flip while the algebraic variants'
is all-but-certain.
⑥ Triton has genuine `exp` and `exp2` (modular exponentiation) but **no `exp10`
op** — it lowers `exp10`, `log`, `log2`, and `log10` to libdevice externs, i.e.
tokens, so none of those compose. The Sophie Germain / Pythagorean variants instead make all of
`log`/`log2`/`log10` the discrete-log inverses of `exp`/`exp2`/`exp10`. (The whole
`exp_b`/`log_b` family rides one order-`d` channel via per-base constants `K_b`,
`exp_b(v)=g^(K_b·v)`; `log_b(e)` is irrational so, as with Triton's `rcpLog2`, no
numeric relation between the bases is claimed — only each base's own law and
inverse. The order-`d` subgroup is cyclic of *prime* order, so all `d−1` non-identity
elements are generators — one per base — and `K_e/K_2/K_10` are chosen as distinct
units; the lone exception is Pythagorean at fp8 (`d = 3`), whose single non-trivial unit
forces `exp2 == exp10` there.)
⑦ Triton's `cos`/`sin` are the real/imaginary parts of `(a+bi)^x` with
`(a,b) = (−3/5, 4/5)`, a norm-1 Pythagorean rotor in `Z/2^w` — so angle addition
and `cos²+sin² == 1` hold. The Pythagorean variant reproduces this on the value residue,
where it coexists with the exact value relations.
⑧ Triton carries the IEEE `∞`/`NaN` *bit patterns* as ordinary scrambled payloads
and does ring arithmetic on them, so none of the `∞`/`NaN` laws hold — they are
not modeled. The algebraic variants use the projective line plus an absorbing
`NaN`.
⑨ No finite ring carries an order compatible with its arithmetic. All four compute
`min`/`max` on a signed residue/payload representative — deterministic,
commutative, reassociation-invariant, but not the IEEE numeric order.
⑩ *token*: a deterministic, op-distinct scramble — equal inputs give equal
outputs and distinct ops give distinct outputs, but it honors no real-math
identity. This is the correct model for a function with no usable algebraic
structure: by Lindemann–Weierstrass / Schanuel, transcendental values at distinct
algebraic arguments are independent, so a fresh free generator per op is faithful.
In the *how realized* rows (roots and transcendentals) the marker shifts meaning
from the identity rows above: ✅ flags a realization that carries a genuine
identity, 🟡 a bare token (supported, but structureless) — so a column's ✅s there
are exactly the structured (non-token) realizations it has.
⑪ A *value-faithful* cast — one recovering the wide value — is impossible for any
model and isn't the goal here (these are fingerprints, not values): it would have
to commute with both `+` and `×`, i.e. be a ring hom between fields of different
characteristic. What the `FPSanAlgebraicField` coprime tower buys instead is that its
in-chain casts (`fp4`/`fp8`/`fp16`/`fp32`/`fp64`) are multiplicative homomorphisms in
both directions and form a commutative diagram: `cast<T>(x·y) == cast<T>(x)·cast<T>(y)`
and `cast<T>(cast<U>(x)) == cast<T>(x)` (so up-then-down round trips,
`narrow(widen(x)) == x`). `fp6` is off the chain. `fp64` (`double`) *is* in the
tower (`p_64 = (p_32−1)·c + 1` with `c` coprime to `p_32−1`), so its casts are
multiplicative too; the only wrinkle is that the `fp32`↔`fp64` edge dlogs over
`p_32−1 ≈ 2^32`, where the linear scan is hopeless — but `p_32−1` is smooth by
construction, so it goes by Pohlig–Hellman (factor the group order, one small dlog
per prime power, CRT-combine) in a few thousand steps. Triton's scrambled-payload
sign-resize *also* composes — but it is not multiplicative, which is the whole
point. The one limit on composition, shared by all three (and by real IEEE), is
that routing through a `U` narrower than both ends loses information the direct
cast keeps — as it must.
⑫ Neither model special-cases subnormals (no flush-to-zero). A subnormal is still
an exact dyadic value, so `phi_n` encodes it exactly like a normal and it obeys
the same value relations; FPSan scrambles its bit pattern like any other. Listed
because many fp tools *do* special-case subnormals — here nothing is needed.
⑬ The collision set — which distinct inputs land on the same fingerprint — is
fixed for FPSan's single tuned scramble: the same blind spots on every run. Each
algebraic prime has its own independent collision set, so the unnumbered and `2`
variants (or any freshly drawn prime) catch a coincidental match the other missed.
⑭ sqrt's round trip `sqrt(x)² == x` holds only on the square residues — a `1/2^a`
fraction of the units, `a` = how many times 2 divides the group order: `½` for the
Field (`p ≡ 3 mod 4`), `¼` for Sophie Germain (two factors), `⅛` for Pythagorean (`p ≡ 1 mod 4`).
Multiplicativity — the headline — is 100% regardless of `a`.
⑮ cbrt is a *perfect* cube root (multiplicative, `cbrt(x)³ == x` for all `x`)
exactly where 3 is coprime to the group order — Field (`p ≡ 2 mod 3`) and Sophie Germain. For
Pythagorean, `p = 4d+1` forces `3 | p−1`, so cbrt has no power-map form and stays a token.
⑯ The composite Sophie Germain / Pythagorean moduli `n = p·d` can't carry a cast homomorphism — the
prime `d` inside `p−1` blocks the coprime-tower divisibility — so their casts stay
the reduce-mod convention, which doesn't even *compose* (so on casts they fall
behind Triton's resize). `FPSanAlgebraicField` is therefore the only algebraic variant
that keeps pace with Triton on casts — and being multiplicative, it then exceeds
Triton there.

**Bottom line.** Triton tracks the *structure* of a computation, not its values:
it nails the 0/1 ring identities by construction and ships genuine `exp`, `exp2`,
and `cos`/`sin`, but no value coincidence (`2+2==4`) survives the scramble, and
`∞`/`NaN` are unmodeled. The algebraic variants make the encoding itself a ring
homomorphism, so every rational-function identity holds within a width and the
`∞`/`NaN` story is principled — at the cost of mod-`n` arithmetic and a few bits
of collision margin (recoverable by repeating with the `2` variant's second
prime). `FPSanAlgebraicField` is the clean field — exact `+ − × /`, multiplicative
`sqrt`/`cbrt`, and a commutative diagram of multiplicative casts across the width
tower; pick it when no exp/log/trig law is needed. `FPSanAlgebraicRingSophieGermain`
adds the exact `exp`/`exp2`/`log`/`log2` homomorphisms (and keeps `sqrt`/`cbrt`),
at the price of zero-divisors; `FPSanAlgebraicRingPythagorean` adds `sin`/`cos` too,
at a roughly `√2`-smaller `d`. The one limitation shared by all four, not fixable
by any sign or exponent trick, is order (`min`/`max`/comparisons): a finite ring
has no compatible order.
