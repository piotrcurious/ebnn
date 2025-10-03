Nice code — it’s an embedded implementation of binary / hybrid neural-network layers (EBNN ≈ Embedded Binary Neural Network). I’ll explain what each part does, how data are laid out, and point out a couple of real bugs + concrete fixes and improvement suggestions.


---

High-level summary

The file implements binary versions of fully-connected (linear) and convolutional layers, plus a float-input / binary-filter convolution variant.

Inputs and filters for binary layers are bit-packed (8 activations per byte). Outputs for binary layers are also bit-packed (one output bit = one neuron/feature).

Convolutions support stride, padding and a 2-stage conv+pooling tiling scheme (lots of pl_* parameters).

Batch-normalization is applied after convolution/summing, then the result is binarized by sign (≥0 → 1, <0 → 0).

There’s also a blinear_sm_layer that does argmax (soft-max / winner-take-all style): it returns the index of the largest (non-binarized) response.



---

File layout & key macros/constants

#define MAX/MIN simple helpers.

CEIL_POS(X) — computes ceil(X) by looking at fractional part: ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X)).
Used to compute number of bytes when packing bits.
(I note an integer-based ceil would be safer in many places — see improvements below.)

MAX_FILTER_BYTES = 12 — buffer size used for temporary filter/patch slices (supports filters up to ~10×10).

bits[8] is a bit-mask table for checking a certain bit inside a byte (0x80 for MSB down to 0x01 for LSB) — the code uses MSB-first bit ordering.



---

Data layout (important)

Binary input activations A and binary filters F are stored bit-packed, row-major.
E.g. N bits → (N + 7) / 8 bytes (code uses CEIL_POS to compute this).

Float inputs (for fconv_layer) are plain float arrays with layout: channel-major blocks of w*h each: A + i * (w*h) gives slice for channel i.

Outputs C for binary layers are packed 1 bit per output and written into bytes with shifting logic (c_shift, c_idx, c_mask).



---

Layer functions — what they do

blinear_layer(const uint8_t* A, const uint8_t* F, uint8_t* C, ... , int m, int n, int k)

A binary fully-connected layer:

m = number of input rows/examples (or spatial positions).

n = input dimension (number of input bits per example).

k = number of output neurons (filters).


For each example i and output j:

res = bdot(A_row, F_j, n) — compute dot-product between bit-packed input and bit-packed filter, but result is in integer domain representing sum of ±1 matches (see bdot below).

res += Bias[j]; res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]).

Threshold to sign and pack bit into C (bit-packing logic with c_shift, c_idx).



blinear_sm_layer(...)

Same dot + BN but instead of writing a binary bit, it finds the j with the maximum res and writes that index into C[i] (i.e., classification argmax per input). Not bit-packed.


fconv_layer(const float* A, const uint8_t* F, uint8_t* C, ...)

Input A is float volumes; filters F are binary (bitpacked). The output is binary bit-packed C.

It computes a convolution (with pooling tiling) by calling fconv per output feature and packs the max-pooled & BN-thresholded result into C.


bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C, ...)

Pure binary convolution: both A and F are bit-packed. Uses bconv helper to compute each pooling tile’s max response and pack into C.



---

Core numeric helpers

batch_norm(float f, const float Gamma, const float Beta, const float Mean, const float Std)

Implements: ((f - Mean) / Std) * Gamma + Beta — a standard BN transform. (Guard for Std==0 would be prudent.)


bdot(const uint8_t* A, const uint8_t* B, const int N)

Computes binary dot product between two bit-packed vectors of length N bits.

Implementation detail:

For each byte: popcnt8(~(A[i] ^ B[i])) — this is the count of equal bits per byte (XNOR).

Sum over bytes: res = count_equal_bits.

Then res = res * 2 - N converts count of equal bits to sum of ±1 values: if a bit matches, contribute +1, else -1. So final range is -N..+N.

This is a common trick to implement bipolar binary dot products (weights ∈ {−1, +1}).



fdot_3d(const float* A, const uint8_t* B, ...)

Computes convolution between float activations A and a binary filter B interpreted as ±1 weights.

It iterates over the kw × kh patch and for each bit of the filter calls nthbitset_arr(B, b_idx):

If bit=1 it res += a, else res -= a. That is, binary filter bit = 1 → weight +1, 0 → weight −1.



bdot_3d(...)

The binary counterpart that builds small bit-slices for each input channel (bslice_4d / bslice_2d_filter) and calls bdot.

Takes care of padding: when slice is smaller than full filter window it crops and uses a filtered copy of the binary filter bits.



---

Index & conv helpers

idx_2d, idx_3d, idx_4d — standard flattening helpers; note the function argument naming (rows, cols) — be careful: idx_2d(i,j,rows) returns i*rows + j so rows is the row stride (often equals width).

conv_idx(pl_i, x, kx, sx, px) — converts pool-local index into convolution coordinate (returns -INT_MAX for out of conv range).

convpool_size(...) — computes the size of pooled output (taking into account convolution output size and pool tile dimensions).



---

Bit operations & slicing

rotr1(x) — rotate-right 1 bit on a byte.

nthbitset_arr(arr,n) — returns 1/0 whether n-th bit is set in bit-packed array arr (uses bits[n%8]).

popcnt8(uint8_t v) — counts set bits in a byte with a fast bit-hack (no builtin used).

bslice_2d, bslice_2d_filter, bslice_4d — extract a kw×kh patch from a (bit-packed) 2D or 4D source into a small byte buffer dst (bit-packed). They return the number of bits actually copied (useful when patch touches padding).



---

Bugs & risky code patterns (you should fix these)

1. Uninitialized mask variable
In bslice_2d, bslice_2d_filter, and bslice_4d the local variable uint8_t mask; is declared but never initialized before doing mask = rotr1(mask);. Reading mask uninitialized is undefined behavior. The code uses mask only to call rotr1(mask) but never uses mask afterwards; likely these mask = rotr1(mask) lines are vestigial and should be removed — or initialize mask = 1 << 7 before the loop, if you intend to use mask.

Fix (quick): remove the mask variables and the mask = rotr1(mask); lines in those functions; they are unnecessary because shift already tracks bit position.
Or initialize:

uint8_t mask = 0x80;
/* ... inside loop ... */
mask = rotr1(mask); // if you really need it


2. CEIL_POS uses floating casts
CEIL_POS expects X≥0 — it uses (int) casts and fractional tests. For byte/bit math using integer arithmetic is clearer and safer. Example alternative for integer-ceil of a/b:

#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

Use CEIL_DIV(N, 8) instead of CEIL_POS(N/8.0).


3. batch_norm divide by zero
If Std == 0 you get divide-by-zero. Add a guard:

if (Std == 0.0f) Std = 1e-6f;


4. Performance opportunities

Replace popcnt8 loop with compiler builtin (if available) for speed & correctness: __builtin_popcount((unsigned) v) or __builtin_popcount((unsigned)A[i]^B[i]) etc. On ARM/Cortex-M consider using hardware POPCNT intrinsic if present.

Where bdot loops bytes, you could use 32-bit reads and __builtin_popcount on a 32-bit word for faster throughput (take care of end bytes).

CEIL_POS floating-point manipulations can be slower and less safe on embedded targets; prefer integer arithmetic.



5. Naming / index clarity
idx_2d(i,j,rows) uses rows as row stride; make sure callers pass width/rows consistently — there are places where idx_2d(j,k,w) is used, so check that ordering matches expected layout (the code is consistent internally but easy to misread).




---

Example of how to call (conceptual)

Binary fully connected: suppose you have m examples, each n features binary-packed:

ni = (n + 7) / 8; so A size = m * ni bytes.

F filters: k filters each ni bytes → k * ni bytes.

C must have ceil(m * k / 8) bytes (packed).

Call:

blinear_layer(A, F, C, Bias, Gamma, Beta, Mean, Std, m, n, k);


Binary conv: A is a bit-packed tensor with layout [d, h, w] where each plane is w*h bits packed into bytes. F is num_f filters with bit-packed kw*kh per channel, etc. The many pl_* parameters control pool tile size and pool stride.



---

Concrete small fixes (copy/paste)

Initialize masks / remove problematic lines and use integer ceil macro:

/* safer integer ceil for positive ints */
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

Replace uses like CEIL_POS(kw*kh/8.0) with CEIL_DIV(kw*kh, 8).

Fix bslice_2d init (remove mask or initialize it):

static int bslice_2d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int w, const int h, const int kw,
                     const int kh)
{
  int i, j, n, idx, shift, bytes, x_kw, y_kh;
  uint8_t bitset;

  bytes = CEIL_DIV(kw*kh, 8);
  for (i = 0; i < bytes; ++i) dst[i] = 0;

  idx = 0;
  shift = 7;
  n = 0;
  x_kw = x + kw;
  y_kh = y + kh;
  for (i = x; i < x_kw; ++i) {
    for (j = y; j < y_kh; ++j) {
      if (i < 0 || i > h-1 || j < 0 || j > w-1) continue;
      bitset = nthbitset_arr(src, idx_2d(i, j, w));
      dst[idx/8] |= bitset << shift;
      idx++;
      shift--;
      if (shift < 0) shift = 7;
      n++;
    }
  }
  return n;
}

And similarly for the other bslice_* functions.


---

Final notes / mental model

The implementation treats binary weights/activations as ±1 and encodes them as bits (1 => +1, 0 => -1). Dot products and convolutions are implemented using bit operations + popcount/XNOR tricks for speed and compact memory — ideal for small, low-memory embedded NN inference.

Float convolution variant (fconv) uses float activations and binary filters; it computes inner products by adding/subtracting the floats according to filter bits, then does pooling (max inside pooling tile), BN and thresholding into binary outputs.

The pl_* parameters implement an inner pooling tiling mechanism: first compute convolution outputs for positions inside the pooling tile and take max, then move to next pooling tile.



---

If you want, I can:

produce a minimal example (tiny network) showing how to pack A, F, allocate C and call bconv_layer or blinear_layer, or

prepare a small patch that fixes the uninitialized-variable bugs and replaces CEIL_POS uses with safe integer ceil macros.


Which would you like next?

