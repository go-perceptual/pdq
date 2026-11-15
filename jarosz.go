package pdq

// jaroszPasses is the number of (rows, cols) repetitions. Two of them means
// four 1-D box passes — X, Y, X, Y — and a box filter applied twice in a
// direction is a tent filter, which is the point: the downsample point for a
// block is an average over the whole block, peaked at its centre.
//
// ⛔ This is what makes PDQ survive a light trim. pHash selects single pixels
// by convolving a mean filter over small patches, so only a few pixels of
// each block reach the hash; crop by a few pixels and a different set is
// chosen. Here every pixel of the block contributes.
const jaroszPasses = 2

// jaroszWindow is the box width for one pass, from the reference:
// ceil(old / (2*new)) — half the block size, so that two passes span the
// block.
func jaroszWindow(oldDim, newDim int) int {
	return (oldDim + 2*newDim - 1) / (2 * newDim)
}

// jarosz smooths buf in place with nreps repetitions of (box along rows, box
// along cols).
func jarosz(buf []float32, rows, cols, windowRows, windowCols, nreps int) {
	tmp := make([]float32, len(buf))
	for i := 0; i < nreps; i++ {
		boxAlongRows(buf, tmp, rows, cols, windowRows)
		boxAlongCols(tmp, buf, rows, cols, windowCols)
	}
}

// boxAlongRows runs a 1-D box filter along each row.
func boxAlongRows(in, out []float32, rows, cols, window int) {
	for i := 0; i < rows; i++ {
		box(in[i*cols:(i+1)*cols], out[i*cols:(i+1)*cols], cols, 1, window)
	}
}

// boxAlongCols runs a 1-D box filter down each column.
func boxAlongCols(in, out []float32, rows, cols, window int) {
	for j := 0; j < cols; j++ {
		box(in[j:], out[j:], rows, cols, window)
	}
}

// box is the 1-D sliding-window mean of n samples taken with the given
// stride: a running sum, so each output costs one add and one subtract
// instead of a window's worth of adds.
//
// ⛔ Ported from the reference's box1DFloat phase for phase, because two
// details decide the hash and neither is guessable from the shape of the
// algorithm:
//
//   - There are FOUR phases, not three. The window GROWS at the leading edge
//     (phase 2), slides full (phase 3), then SHRINKS at the trailing edge
//     (phase 4). A three-phase reading loses the shrink and walks off the end.
//   - Each output divides by the LIVE window size, not by the full window.
//     Dividing by the full window would darken both edges of every row and
//     column, plausibly and wrongly.
//
// halfWindow is (window+2)/2 — 7 gives 4, 8 gives 5 — and the four phase
// counts below sum to exactly n writes.
func box(in, out []float32, n, stride, window int) {
	half := (window + 2) / 2

	p1 := half - 1
	p2 := window - half + 1
	p3 := n - window
	p4 := half - 1

	li, ri, oi := 0, 0, 0
	var sum float32
	cur := 0

	// Phase 1: accumulate the first sum, no writes.
	for i := 0; i < p1; i++ {
		sum += in[ri]
		cur++
		ri += stride
	}
	// Phase 2: initial writes, window still growing.
	for i := 0; i < p2; i++ {
		sum += in[ri]
		cur++
		out[oi] = sum / float32(cur)
		ri += stride
		oi += stride
	}
	// Phase 3: writes with a full window.
	for i := 0; i < p3; i++ {
		sum += in[ri]
		sum -= in[li]
		out[oi] = sum / float32(cur)
		li += stride
		ri += stride
		oi += stride
	}
	// Phase 4: final writes, window shrinking.
	for i := 0; i < p4; i++ {
		sum -= in[li]
		cur--
		out[oi] = sum / float32(cur)
		li += stride
		oi += stride
	}
}

// decimate takes the 64x64 output from the smoothed full-resolution buffer.
//
// ⛔ "Target centers not corners", as the reference puts it: the sample for
// output cell i is at floor((i+0.5)*in/out), the MIDDLE of the block, not its
// first pixel. Sampling corners biases the whole hash toward the top-left.
func decimate(in []float32, rows, cols int, out *[64][64]float32) {
	// No clamp is needed and none is written: the largest index this can
	// produce is int(63.5*rows/64), which is strictly below rows for every
	// rows >= 1. A defensive clamp here would be unreachable code that no
	// test could ever cover — and unreachable code in a hashing path is
	// something a reader has to disprove before trusting the rest.
	for i := 0; i < 64; i++ {
		ini := int((float64(i) + 0.5) * float64(rows) / 64)
		for j := 0; j < 64; j++ {
			inj := int((float64(j) + 0.5) * float64(cols) / 64)
			out[i][j] = in[ini*cols+inj]
		}
	}
}
