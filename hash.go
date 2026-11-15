// Package pdq implements the PDQ perceptual image hash.
//
// PDQ answers one question: are two pictures essentially the same picture?
// It is a SYNTACTIC hasher — it compares similarity without extracting
// meaning — and it is built for media shared with minimum adversariality:
// recompressed, resized, lightly blurred, JPEG turned into PNG. It is NOT
// built to survive cropping, border addition or deliberate obfuscation.
//
// The algorithm is Facebook's, published as "The TMK+PDQF video-hashing
// algorithm and the PDQ image-hashing algorithm" (hashing/hashing.pdf in
// github.com/facebook/ThreatExchange). This package is a port of the
// reference C++ in pdq/cpp of the same repository, read at the commit named
// in testdata/REFERENCE. Every step below cites the source it came from.
//
// ⛔ Bit identity with another implementation is NOT the contract. The
// specification says so itself: the resize down to the hashing resolution is
// implementation-specific, and hashes from two implementations of the same
// image are expected to differ by 0, 2, 4 or 6 bits of Hamming distance. What
// this package promises is the published algorithm, run at full resolution
// the way the reference does, with the stated bound asserted against the
// reference's own regression vectors.
package pdq

import (
	"encoding/hex"
	"errors"
	"math/bits"
	"strconv"
)

// Hash is a 256-bit PDQ hash.
//
// The bit at row i, column j of the 16x16 transform output is bit j of word
// i, which is how the reference stores it (pdqBuffer16x16ToBits sets bit
// i*16+j, and Hash256 keeps 16 words of 16 bits).
type Hash [16]uint16

// ErrMalformed is returned for a hash string that is not 64 hex digits.
var ErrMalformed = errors.New("pdq: malformed hash")

// String renders the hash as 64 lowercase hex digits.
//
// ⛔ The reference prints the HIGHEST word first (Hash256's format lists
// w[15] down to w[0]), so the leading hex digits are row 15, not row 0.
// Printing them the other way round produces a plausible string that matches
// nothing.
func (h Hash) String() string {
	var b [32]byte
	for i := 0; i < 16; i++ {
		w := h[15-i]
		b[2*i] = byte(w >> 8)
		b[2*i+1] = byte(w)
	}
	return hex.EncodeToString(b[:])
}

// ParseHash reads the 64-hex-digit form written by String.
func ParseHash(s string) (Hash, error) {
	var h Hash
	if len(s) != 64 {
		return h, ErrMalformed
	}
	for i := 0; i < 16; i++ {
		v, err := strconv.ParseUint(s[4*i:4*i+4], 16, 16)
		if err != nil {
			return Hash{}, ErrMalformed
		}
		h[15-i] = uint16(v)
	}
	return h, nil
}

// Distance is the Hamming distance between two hashes: 0 when identical, 256
// at most.
//
// The specification's guidance on thresholds: distances between unrelated
// images average 128, and a distance of 30 or less — better 20 or less — has
// been found to indicate a confident match.
func (h Hash) Distance(o Hash) int {
	n := 0
	for i := range h {
		n += bits.OnesCount16(h[i] ^ o[i])
	}
	return n
}

// Norm is the number of 1 bits.
//
// ⭐ It is 128 for every non-degenerate image, and that is a real invariant
// rather than a statistical one: each bit says whether its coefficient
// exceeds the MEDIAN of the 256 coefficients, so exactly half are set. It
// only fails when coefficients tie, which happens for degenerate input such
// as a solid colour — and those are what the quality metric exists to flag.
func (h Hash) Norm() int {
	n := 0
	for _, w := range h {
		n += bits.OnesCount16(w)
	}
	return n
}
