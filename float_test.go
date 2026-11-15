package pdq

import (
	"testing"
)

// TestFloatAndSumAgreeBitForBit is the invariant that keeps the two entry
// points from drifting: the hash is the float feature compared against its
// own median, so recomputing the bits from Float must reproduce Sum exactly.
//
// ⛔ This is what makes exporting the float stage safe. A video hash built on
// a SECOND implementation of the transform would be a second thing to keep
// correct, and the day the two disagreed, nothing would say so.
func TestFloatAndSumAgreeBitForBit(t *testing.T) {
	for _, sz := range [][2]int{{64, 64}, {127, 65}, {300, 200}, {33, 97}} {
		img := gradient(sz[0], sz[1])

		want, wq, err := Sum(img)
		if err != nil {
			t.Fatal(err)
		}
		f, fq, err := Float(img)
		if err != nil {
			t.Fatal(err)
		}
		if fq != wq {
			t.Errorf("%dx%d: quality %d from Float, %d from Sum", sz[0], sz[1], fq, wq)
		}

		// Rebuild the hash from the floats, the way bitsOf does.
		flat := f // a copy: torben must not see a reordered slice
		med := torben(flat[:])
		var got Hash
		for i := 0; i < 16; i++ {
			var w uint16
			for j := 0; j < 16; j++ {
				if f[i*16+j] > med {
					w |= 1 << uint(j)
				}
			}
			got[i] = w
		}
		if got != want {
			t.Fatalf("%dx%d: hash rebuilt from Float is %s, Sum says %s", sz[0], sz[1], got, want)
		}
	}
}

// TestFloatIsNotAllZero guards against the degenerate success where the
// feature is exported as an empty vector and every comparison then agrees.
func TestFloatIsNotAllZero(t *testing.T) {
	f, _, err := Float(gradient(200, 150))
	if err != nil {
		t.Fatal(err)
	}
	nonzero := 0
	for _, v := range f {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < FloatDim/2 {
		t.Fatalf("only %d of %d coefficients are non-zero; the feature is degenerate", nonzero, FloatDim)
	}
}

func TestFloatRejectsAnEmptyImage(t *testing.T) {
	if _, _, err := Float(gradient(0, 0)); err == nil {
		t.Fatal("an empty image produced a feature")
	}
}
