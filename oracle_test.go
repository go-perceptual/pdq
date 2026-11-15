package pdq

import (
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"testing"
)

// ⚠ MEASURED, 2026-11-15, so that nobody repeats the experiment: swapping
// this decoder for the fleet's go-images/jpeg fork changes NOTHING here. The
// hash came out bit-identical. The fork upsamples chroma the way libjpeg does
// for a FOUR-component picture, and the reference image is three-component,
// so the difference it exists for cannot arise. The residual distance from
// the reference is therefore NOT explained by the decoder, and the decoder
// hypothesis remains untested for four-component input.

// The reference's own regression vectors, copied from
// pdq/cpp/output-regtest/out in github.com/facebook/ThreatExchange. These are
// facts about the algorithm, not redistributed content: the IMAGES are not
// kept in this repository. Put them in testdata/ to run this test; it skips
// when they are absent, and ⛔ says so loudly rather than passing quietly,
// because a test that silently skips is a test that reports zero.
var referenceVectors = []struct {
	file    string
	hash    string
	quality Quality
}{
	{"bridge-1-original.jpg", "d8f8f0cee0f4a84f0637022a078f67f0b36e2ed596621e1d33e6339c4e9c9b22", 100},
}

// tolerance is the bound the specification itself states: hashes of the same
// image from two implementations are expected to differ by 0, 2, 4 or 6 bits,
// because the downscale to hashing resolution is implementation-specific.
//
// ⛔ Bit identity is therefore NOT the contract and must not be asserted as
// one. What is asserted is this published bound — and the measured distance
// is always logged, so a drift from 0 to 6 is visible even while passing.
const tolerance = 6

func TestAgainstReferenceVectors(t *testing.T) {
	ran := 0
	for _, v := range referenceVectors {
		path := filepath.Join("testdata", v.file)
		f, err := os.Open(path)
		if os.IsNotExist(err) {
			t.Logf("SKIP %s: not in testdata/ (fetch it from the reference repository to run this)", v.file)
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", v.file, err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s: decode: %v", v.file, err)
		}

		want, err := ParseHash(v.hash)
		if err != nil {
			t.Fatalf("%s: malformed expected hash: %v", v.file, err)
		}

		got, q, err := Sum(img)
		if err != nil {
			t.Fatalf("%s: %v", v.file, err)
		}
		d := got.Distance(want)
		t.Logf("%s: distance %d from the reference (got %s, want %s), quality %d vs %d",
			v.file, d, got, want, q, v.quality)
		if d > tolerance {
			t.Errorf("%s: distance %d exceeds the specification's stated bound of %d",
				v.file, d, tolerance)
		}
		if got.Norm() != 128 {
			t.Errorf("%s: norm = %d, want 128", v.file, got.Norm())
		}
		ran++
	}
	if ran == 0 {
		t.Skip("no reference images in testdata/: this test asserted nothing")
	}
}
