# pdq

The PDQ perceptual image hash, in pure Go, with no dependencies and no cgo.

PDQ answers one question: **are two pictures essentially the same picture?**
It is a *syntactic* hasher — it compares similarity without extracting
meaning — built for media shared with minimum adversariality: recompressed,
resized, lightly blurred, JPEG turned into PNG. It is not built to survive
cropping, border addition or deliberate obfuscation.

```go
h, quality, err := pdq.Sum(img)   // 256-bit hash, 0..100 quality
d := h.Distance(other)            // 0 identical, 256 opposite
```

A distance of **30 or less — better 20 or less** — indicates a confident
match. Unrelated images average **128**.

## What it is

| | |
|---|---|
| hash | 256 bits, compared by Hamming distance |
| quality | 0..100; near 0 means featureless, and such a hash should not be trusted |
| symmetries | all eight rotations and flips for the price of one transform |
| cost | the reference states hashing is about as fast as reading the file off disk |

`SumDihedral` returns the hashes of all eight symmetries. They are cheap
because they come from the *same* 16×16 transform by transposition and sign
flips — no image is rotated, and no second transform is run.

## The algorithm, and where it came from

The algorithm is Facebook's, published as *The TMK+PDQF video-hashing
algorithm and the PDQ image-hashing algorithm* (`hashing/hashing.pdf` in
[facebook/ThreatExchange](https://github.com/facebook/ThreatExchange)). This
package is a port of the reference C++ in `pdq/cpp` of the same repository.

1. RGB → **luminance** (`0.299R + 0.587G + 0.114B`), not greyscale.
2. A **two-pass Jarosz tent filter** — four 1-D box passes — at full
   resolution, then the centre sample of each block gives a 64×64 image.
3. A **quality metric** from quantised neighbour gradients of that 64×64.
4. A 2-D **DCT**, keeping slots 1–16 in each direction. Slot 0 is the DC term
   — overall brightness — and dropping it is what makes the hash indifferent
   to exposure.
5. Each of the 256 bits is 1 where its coefficient **exceeds the median**.

⭐ Because every bit compares against the median, **every non-degenerate hash
has exactly 128 one-bits**. That is an invariant, not a statistic, and the
test suite asserts it across image sizes: it is the cheapest proof that the
median, the transform and the bit-packing still agree.

## ⛔ Bit identity is not the contract

The specification states that the downscale to hashing resolution is
implementation-specific, and that two implementations hashing the same image
are expected to differ by **0, 2, 4 or 6 bits**. So this package does not
promise, and does not assert, identity with any other implementation. What it
asserts is that published bound, against the reference's own regression
vectors.

Measured against `bridge-1-original.jpg` from the reference corpus:

```
distance 6 from the reference vector, quality 100 vs 100
```

⚠ Within the bound, but at its edge, and **the residual is not yet
explained**. One hypothesis has been tested and killed: decoding with the
fleet's `go-images/jpeg` fork instead of `image/jpeg` produces a
*bit-identical* hash, because that fork differs only for four-component
pictures and the reference image has three components.

## Running the reference vectors

The reference images are **not** redistributed here. Put them in `testdata/`
and the oracle test runs; without them it skips and says so out loud, because
a test that skips quietly is a test that reports zero.

## Licence

BSD-3-Clause.
