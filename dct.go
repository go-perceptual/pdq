package pdq

import "math"

// dctMatrix is the 16x64 DCT matrix, row-major.
//
// The specification gives the one-dimensional transform as
//
//	D[i][j] = sqrt(2/n) * cos( (2*pi / 4n) * i * (2j+1) )
//
// with n = 64. ⛔ PDQ keeps slots 1..16, NOT 0..15: row r of this matrix is
// slot r+1. Slot 0 is the DC term — the average brightness — and dropping it
// is what makes the hash indifferent to overall exposure.
var dctMatrix = func() [16 * 64]float64 {
	var d [16 * 64]float64
	const n = 64.0
	scale := math.Sqrt(2.0 / n)
	for i := 0; i < 16; i++ {
		for j := 0; j < 64; j++ {
			d[i*64+j] = scale * math.Cos((2*math.Pi/(4*n))*float64(i+1)*float64(2*j+1))
		}
	}
	return d
}()

// dct64To16 computes B = D A Dt, where A is the 64x64 downsample, D is the
// 16x64 matrix above and B is the 16x16 output.
//
// Only the 16x16 corner is wanted, so this computes only that: the reference
// notes that a clever full 64x64 transform followed by extraction measured
// SLOWER than this direct form.
func dct64To16(a *[64][64]float32, b *[16][16]float32) {
	var t [16][64]float64 // T = D A

	for i := 0; i < 16; i++ {
		row := dctMatrix[i*64 : i*64+64]
		for j := 0; j < 64; j++ {
			var sum float64
			for k := 0; k < 64; k++ {
				sum += row[k] * float64(a[k][j])
			}
			t[i][j] = sum
		}
	}

	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			row := dctMatrix[j*64 : j*64+64]
			var sum float64
			for k := 0; k < 64; k++ {
				sum += t[i][k] * row[k]
			}
			b[i][j] = float32(sum)
		}
	}
}

// The eight dihedral symmetries, each derived from the ORIGINAL 16x16
// transform by transposition and sign flips — no image is rotated and no
// second transform is run.
//
// ⛔ The sign conventions are the reference's and are not symmetric in the
// way one would assume: flipX negates EVEN rows while flipY negates EVEN
// columns, and rotate90 negates even columns while rotate270 negates even
// rows. Writing these from first principles gets half of them backwards, and
// the result still looks like a valid hash.

func rotate90(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if j&1 == 1 {
				b[j][i] = a[i][j]
			} else {
				b[j][i] = -a[i][j]
			}
		}
	}
}

func rotate180(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if (i+j)&1 == 1 {
				b[i][j] = -a[i][j]
			} else {
				b[i][j] = a[i][j]
			}
		}
	}
}

func rotate270(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if i&1 == 1 {
				b[j][i] = a[i][j]
			} else {
				b[j][i] = -a[i][j]
			}
		}
	}
}

func flipX(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if i&1 == 1 {
				b[i][j] = a[i][j]
			} else {
				b[i][j] = -a[i][j]
			}
		}
	}
}

func flipY(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if j&1 == 1 {
				b[i][j] = a[i][j]
			} else {
				b[i][j] = -a[i][j]
			}
		}
	}
}

func flipPlus1(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			b[j][i] = a[i][j]
		}
	}
}

func flipMinus1(a, b *[16][16]float32) {
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if (i+j)&1 == 1 {
				b[j][i] = -a[i][j]
			} else {
				b[j][i] = a[i][j]
			}
		}
	}
}
