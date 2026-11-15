package pdq

import (
	"image"
	"image/color"
	"math/rand"
	"testing"
)

// gradient is a picture with real structure: enough gradient in both
// directions that the quality metric saturates and no two DCT coefficients
// tie.
func gradient(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := uint8((x*255)/w ^ (y * 3))
			g := uint8((y * 255) / h)
			b := uint8((x ^ y) * 7)
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	return img
}

func solid(w, h int, c color.RGBA) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

func noise(w, h int, seed int64) image.Image {
	rng := rand.New(rand.NewSource(seed))
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{
				uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255,
			})
		}
	}
	return img
}

// TestNormIsAlwaysOneTwentyEight asserts the invariant the specification
// states: because each bit says whether its coefficient exceeds the MEDIAN of
// the 256, exactly half are set. It is the cheapest proof that the median and
// the bit-packing agree, and it fails loudly if either drifts.
func TestNormIsAlwaysOneTwentyEight(t *testing.T) {
	for _, sz := range [][2]int{{64, 64}, {127, 65}, {256, 256}, {640, 480}, {33, 97}} {
		img := gradient(sz[0], sz[1])
		h, q, err := Sum(img)
		if err != nil {
			t.Fatalf("%dx%d: %v", sz[0], sz[1], err)
		}
		if got := h.Norm(); got != 128 {
			t.Errorf("%dx%d: norm = %d, want 128 (hash %s, quality %d)", sz[0], sz[1], got, h, q)
		}
	}
}

// TestDistanceToItselfIsZero is the other half of the same contract.
func TestDistanceToItselfIsZero(t *testing.T) {
	h, _, err := Sum(gradient(200, 150))
	if err != nil {
		t.Fatal(err)
	}
	if d := h.Distance(h); d != 0 {
		t.Fatalf("distance to itself = %d, want 0", d)
	}
}

// TestUnrelatedImagesAreFarApart pins the other end of the scale: the
// specification says distances between unrelated images average 128, and a
// confident match is 30 or less. Two different noise fields must land nowhere
// near a match.
func TestUnrelatedImagesAreFarApart(t *testing.T) {
	a, _, err := Sum(noise(128, 128, 1))
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := Sum(noise(128, 128, 2))
	if err != nil {
		t.Fatal(err)
	}
	if d := a.Distance(b); d <= 30 {
		t.Fatalf("unrelated images at distance %d, which reads as a confident match", d)
	}
}

// TestSolidColourHasNoQuality: a featureless image is exactly what the
// quality metric exists to flag, and it is also the one case where the
// norm-128 invariant is allowed to fail, because coefficients tie.
func TestSolidColourHasNoQuality(t *testing.T) {
	_, q, err := Sum(solid(100, 100, color.RGBA{37, 37, 37, 255}))
	if err != nil {
		t.Fatal(err)
	}
	if q != 0 {
		t.Fatalf("quality = %d for a solid colour, want 0", q)
	}
}

// TestQualityIsHighForDetail is the positive control for the test above: a
// metric that answered 0 for everything would pass it.
func TestQualityIsHighForDetail(t *testing.T) {
	_, q, err := Sum(noise(128, 128, 7))
	if err != nil {
		t.Fatal(err)
	}
	if q < 50 {
		t.Fatalf("quality = %d for a noise field, want a high score", q)
	}
}

// TestStringRoundTrips also pins the word order: the reference prints the
// HIGHEST word first, so a parse that reads it the other way round would
// round-trip happily and match nothing.
func TestStringRoundTrips(t *testing.T) {
	h, _, err := Sum(gradient(300, 200))
	if err != nil {
		t.Fatal(err)
	}
	s := h.String()
	if len(s) != 64 {
		t.Fatalf("String() = %q, want 64 hex digits", s)
	}
	back, err := ParseHash(s)
	if err != nil {
		t.Fatal(err)
	}
	if back != h {
		t.Fatalf("round trip: %s != %s", back, h)
	}
	if back[15] != h[15] || back[0] != h[0] {
		t.Fatalf("word order lost in round trip")
	}
}

func TestParseRejectsMalformed(t *testing.T) {
	for _, s := range []string{"", "abc", "zz" + "00000000000000000000000000000000000000000000000000000000000000"} {
		if _, err := ParseHash(s); err == nil {
			t.Errorf("ParseHash(%q) accepted a malformed hash", s)
		}
	}
}

func TestEmptyImage(t *testing.T) {
	if _, _, err := Sum(image.NewRGBA(image.Rect(0, 0, 0, 0))); err == nil {
		t.Fatal("an empty image was hashed")
	}
}

// TestDihedralsAreEightDistinctHashesOfNormOneTwentyEight checks the cheap
// symmetry set: every one of the eight must be a well-formed hash, and a
// picture without symmetry must not produce the same hash twice.
func TestDihedralsAreEightDistinctHashesOfNormOneTwentyEight(t *testing.T) {
	d, _, err := SumDihedral(gradient(256, 192))
	if err != nil {
		t.Fatal(err)
	}
	all := []struct {
		name string
		h    Hash
	}{
		{"original", d.Original}, {"rotate90", d.Rotate90},
		{"rotate180", d.Rotate180}, {"rotate270", d.Rotate270},
		{"flipX", d.FlipX}, {"flipY", d.FlipY},
		{"flipPlus1", d.FlipPlus1}, {"flipMinus1", d.FlipMinus1},
	}
	seen := map[Hash]string{}
	for _, e := range all {
		if got := e.h.Norm(); got != 128 {
			t.Errorf("%s: norm = %d, want 128", e.name, got)
		}
		if prev, dup := seen[e.h]; dup {
			t.Errorf("%s has the same hash as %s", e.name, prev)
		}
		seen[e.h] = e.name
	}
}

// TestSumAgreesWithSumDihedralOriginal guards the seam between the two entry
// points: they must not drift apart.
func TestSumAgreesWithSumDihedralOriginal(t *testing.T) {
	img := gradient(321, 123)
	h, q1, err := Sum(img)
	if err != nil {
		t.Fatal(err)
	}
	d, q2, err := SumDihedral(img)
	if err != nil {
		t.Fatal(err)
	}
	if h != d.Original || q1 != q2 {
		t.Fatalf("Sum and SumDihedral disagree: %s/%d vs %s/%d", h, q1, d.Original, q2)
	}
}

// TestBoxFilterWritesEveryOutput is the unit guard for the filter whose phase
// structure is easy to get wrong: four phases must write exactly n values,
// and a mean of a constant signal is that constant everywhere, edges
// included. A three-phase reading, or dividing by the full window instead of
// the live one, breaks the edges and nothing else would notice.
func TestBoxFilterWritesEveryOutput(t *testing.T) {
	for _, n := range []int{16, 64, 65, 100} {
		for _, w := range []int{1, 2, 3, 7, 8} {
			if w > n {
				continue
			}
			in := make([]float32, n)
			out := make([]float32, n)
			for i := range in {
				in[i] = 5
			}
			for i := range out {
				out[i] = -1
			}
			box(in, out, n, 1, w)
			for i, v := range out {
				if v != 5 {
					t.Fatalf("n=%d window=%d: out[%d] = %v, want 5 (every output written, live-window mean)", n, w, i, v)
				}
			}
		}
	}
}

// TestJaroszWindowMatchesTheReference pins ceil(old/(2*new)).
func TestJaroszWindowMatchesTheReference(t *testing.T) {
	cases := []struct{ old, new, want int }{
		{512, 64, 4}, {1024, 64, 8}, {64, 64, 1}, {4000, 64, 32}, {1, 64, 1},
	}
	for _, c := range cases {
		if got := jaroszWindow(c.old, c.new); got != c.want {
			t.Errorf("jaroszWindow(%d,%d) = %d, want %d", c.old, c.new, got, c.want)
		}
	}
}

// TestTorbenReturnsTheLowerMedian is the guard for the trap that a sort would
// have walked into: for an even count the reference returns the LOWER middle
// value, not the mean of the two middle ones.
func TestTorbenReturnsTheLowerMedian(t *testing.T) {
	// 1..256: lower middle is 128, the mean of the two middle ones is 128.5.
	v := make([]float32, 256)
	for i := range v {
		v[i] = float32(i + 1)
	}
	if got := torben(v); got != 128 {
		t.Fatalf("torben = %v, want 128 (the lower median, not 128.5)", got)
	}
	// All equal: no element is strictly less, so the guess is returned.
	for i := range v {
		v[i] = 9
	}
	if got := torben(v); got != 9 {
		t.Fatalf("torben of a constant = %v, want 9", got)
	}

	// The third return — the smallest element greater than the guess — cannot
	// be reached at n=256: it needs less+equal <= 127 while greater <= 128,
	// and those cannot sum to 256. It IS reachable at n=5, which is why this
	// case exists: the branch is live code in the ported algorithm, and a
	// branch no test can reach is a branch nobody has checked.
	if got := torben([]float32{1, 1, 3, 3, 3}); got != 3 {
		t.Fatalf("torben of [1 1 3 3 3] = %v, want 3 (the lower median)", got)
	}
}

func BenchmarkSum1080p(b *testing.B) {
	img := gradient(1920, 1080)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := Sum(img); err != nil {
			b.Fatal(err)
		}
	}
}
