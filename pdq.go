package pdq

import (
	"errors"
	"image"
)

// Quality is the 0..100 metric PDQ reports beside the hash. It says how much
// detail the downsample retained: a featureless or solid-colour image scores
// near 0, and such a hash should not be trusted for matching.
type Quality int

// ErrEmpty is returned for an image with no pixels.
var ErrEmpty = errors.New("pdq: empty image")

// Dihedral holds the hashes of the eight symmetries of one image: the
// original, three rotations and four flips.
//
// They cost almost nothing: all eight come from the SAME 16x16 DCT matrix by
// transposition and sign flips, so no image is rotated and no second
// transform is run.
type Dihedral struct {
	Original   Hash
	Rotate90   Hash
	Rotate180  Hash
	Rotate270  Hash
	FlipX      Hash
	FlipY      Hash
	FlipPlus1  Hash // transpose, about the main diagonal
	FlipMinus1 Hash // about the anti-diagonal
}

// Sum returns the PDQ hash of img and its quality metric.
func Sum(img image.Image) (Hash, Quality, error) {
	d, q, err := sum(img, false)
	return d.Original, q, err
}

// SumDihedral returns the hashes of all eight symmetries of img, and the
// quality metric, for the price of one transform.
func SumDihedral(img image.Image) (Dihedral, Quality, error) {
	return sum(img, true)
}

func sum(img image.Image, all bool) (Dihedral, Quality, error) {
	var d Dihedral
	luma, rows, cols, err := lumaFrom(img)
	if err != nil {
		return d, 0, err
	}

	// Jarosz two-pass tent filter, then take the centre of each block.
	//
	// ⛔ The window is computed from the FULL resolution, and the filter runs
	// at full resolution, exactly as the reference does. The 512x512 pre-
	// resize the document recommends is advice to the CALLER about speed, not
	// a step of the algorithm, and taking it changes the hash by a few bits.
	wr := jaroszWindow(cols, 64)
	wc := jaroszWindow(rows, 64)
	jarosz(luma, rows, cols, wr, wc, jaroszPasses)

	var block [64][64]float32
	decimate(luma, rows, cols, &block)

	q := qualityOf(&block)

	var dct [16][16]float32
	dct64To16(&block, &dct)

	d.Original = bitsOf(&dct)
	if all {
		var t [16][16]float32
		rotate90(&dct, &t)
		d.Rotate90 = bitsOf(&t)
		rotate180(&dct, &t)
		d.Rotate180 = bitsOf(&t)
		rotate270(&dct, &t)
		d.Rotate270 = bitsOf(&t)
		flipX(&dct, &t)
		d.FlipX = bitsOf(&t)
		flipY(&dct, &t)
		d.FlipY = bitsOf(&t)
		flipPlus1(&dct, &t)
		d.FlipPlus1 = bitsOf(&t)
		flipMinus1(&dct, &t)
		d.FlipMinus1 = bitsOf(&t)
	}
	return d, q, nil
}

// lumaFrom converts img to luminance in 0..255, row-major.
//
// ⛔ Luminance, NOT greyscale: Y = 0.299R + 0.587G + 0.114B, the weighted
// form, because it is built to match how the eye weighs the channels. An
// unweighted average of R, G and B is a different picture and a different
// hash.
func lumaFrom(img image.Image) ([]float32, int, int, error) {
	b := img.Bounds()
	rows, cols := b.Dy(), b.Dx()
	if rows <= 0 || cols <= 0 {
		return nil, 0, 0, ErrEmpty
	}
	out := make([]float32, rows*cols)
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA() // 16-bit, alpha-premultiplied
			out[i] = 0.299*float32(r>>8) + 0.587*float32(g>>8) + 0.114*float32(bl>>8)
			i++
		}
	}
	return out, rows, cols, nil
}

// qualityOf scores how much detail survived into the 64x64 downsample, by
// summing quantised neighbour gradients.
//
// The quantisation is the point, not an optimisation: it counts SIGNIFICANT
// gradients rather than the sum of many tiny ones. Every constant here is
// heuristic and comes from the reference.
func qualityOf(b *[64][64]float32) Quality {
	sum := 0
	for i := 0; i < 63; i++ {
		for j := 0; j < 64; j++ {
			d := int((b[i][j] - b[i+1][j]) * 100 / 255)
			if d < 0 {
				d = -d
			}
			sum += d
		}
	}
	for i := 0; i < 64; i++ {
		for j := 0; j < 63; j++ {
			d := int((b[i][j] - b[i][j+1]) * 100 / 255)
			if d < 0 {
				d = -d
			}
			sum += d
		}
	}
	q := sum / 90
	if q > 100 {
		q = 100
	}
	return Quality(q)
}

// bitsOf turns the 16x16 transform output into 256 bits: 1 where the
// coefficient is STRICTLY greater than the median of all 256.
//
// Strictly: a coefficient equal to the median yields 0, which is exactly why
// a degenerate image can fail the norm-128 invariant.
func bitsOf(d *[16][16]float32) Hash {
	var flat [256]float32
	k := 0
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			flat[k] = d[i][j]
			k++
		}
	}
	med := torben(flat[:])

	var h Hash
	for i := 0; i < 16; i++ {
		var w uint16
		for j := 0; j < 16; j++ {
			if d[i][j] > med {
				w |= 1 << uint(j)
			}
		}
		h[i] = w
	}
	return h
}

// torben is Torben Mogensen's median, ported from the reference.
//
// ⛔ It is ported rather than replaced by a sort because for an EVEN count it
// returns the LOWER middle value, not the mean of the two middle ones. Taking
// the mean — the obvious reading of "median" — moves the threshold and flips
// the bits of every coefficient lying between them.
//
// It also never modifies or copies the input, which is why the reference can
// hand it the transform output directly.
func torben(m []float32) float32 {
	n := len(m)
	min, max := m[0], m[0]
	for _, v := range m[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	var less, greater, equal int
	var guess, maxltguess, mingtguess float32
	for {
		guess = (min + max) / 2
		less, greater, equal = 0, 0, 0
		maxltguess, mingtguess = min, max
		for _, v := range m {
			switch {
			case v < guess:
				less++
				if v > maxltguess {
					maxltguess = v
				}
			case v > guess:
				greater++
				if v < mingtguess {
					mingtguess = v
				}
			default:
				equal++
			}
		}
		if less <= (n+1)/2 && greater <= (n+1)/2 {
			break
		}
		if less > greater {
			max = maxltguess
		} else {
			min = mingtguess
		}
	}

	switch {
	case less >= (n+1)/2:
		return maxltguess
	case less+equal >= (n+1)/2:
		return guess
	default:
		return mingtguess
	}
}
