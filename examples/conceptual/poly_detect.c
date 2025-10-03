/* poly_detect.c
 *
 * Example using ebnn.h to train a binary classifier that recognizes
 * which polynomial (from a small set) generated a sampled sign-pattern.
 *
 * Build:
 *   gcc -O2 poly_detect.c -o poly_detect -lm
 *
 * Run:
 *   ./poly_detect
 *
 * Make sure ebnn.h is in the same directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ebnn.h" /* The header you posted earlier (must be in same folder) */

#define N_POINTS 64           /* number of sample points per example (input dimension) */
#define N_BYTES ((N_POINTS + 7) / 8)
#define NUM_POLYS 4           /* number of pattern polynomials (classes) */
#define EXAMPLES_PER_POLY 300 /* how many training examples per class */
#define TEST_PER_POLY 100
#define LEARNING_RATE 0.05f
#define EPOCHS 30

/* polynomial definitions (coeff order: c0 + c1*x + c2*x^2 + c3*x^3) */
static const int MAX_DEG = 3;
static const float polys[NUM_POLYS][4] = {
  /* p0(x) = x^2 */
  {0.0f, 0.0f, 1.0f, 0.0f},
  /* p1(x) = x^3 - x */
  {0.0f, -1.0f, 0.0f, 1.0f},
  /* p2(x) = 1 + 2x + x^2  */
  {1.0f, 2.0f, 1.0f, 0.0f},
  /* p3(x) = 2x^2 - 1 */
  {-1.0f, 0.0f, 2.0f, 0.0f}
};

static float eval_poly(float x, const float coeffs[4]) {
  float xpow = 1.0f;
  float res = 0.0f;
  for (int i = 0; i <= MAX_DEG; ++i) {
    res += coeffs[i] * xpow;
    xpow *= x;
  }
  return res;
}

/* simple Gaussian noise generator (Box-Muller) */
static float gaussian_noise(float sigma) {
  static int seeded = 0;
  if (!seeded) { seeded = 1; srand((unsigned)time(NULL)); }
  float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
  float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
  float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
  return z0 * sigma;
}

/* pack a boolean array (0/1) of length nbits into bytes (MSB-first within each byte) */
static void pack_bits_from_bools(const unsigned char *bits, int nbits, uint8_t *out) {
  int nbytes = (nbits + 7) / 8;
  for (int i = 0; i < nbytes; ++i) out[i] = 0;
  int shift = 7;
  int out_idx = 0;
  for (int i = 0; i < nbits; ++i) {
    uint8_t b = bits[i] ? 1 : 0;
    out[out_idx] |= b << shift;
    shift--;
    if (shift < 0) { shift = 7; out_idx++; }
  }
}

/* pack float weights (k x n) into bitpacked filters format used by ebnn:
   bit = 1 -> +1 weight; bit = 0 -> -1 weight */
static void pack_filters_from_weights(const float *W, int k, int n, uint8_t *F) {
  int ni = (n + 7) / 8;
  /* zero */
  for (int i = 0; i < k * ni; ++i) F[i] = 0;
  for (int j = 0; j < k; ++j) {
    int out_idx = j * ni;
    int shift = 7;
    for (int f = 0; f < n; ++f) {
      int bit = (W[j * n + f] >= 0.0f) ? 1 : 0;
      F[out_idx] |= (bit << shift);
      shift--;
      if (shift < 0) { shift = 7; out_idx++; }
    }
  }
}

/* compute float dot between float weight vector Wj and bitpacked input Ain:
   interpret bits as bipolar (+1 for 1, -1 for 0) */
static float float_dot_weights_and_bitinput(const float *Wj, const uint8_t *Ain, int n) {
  float res = 0.0f;
  for (int idx = 0; idx < n; ++idx) {
    int bit = nthbitset_arr(Ain, idx);
    float val = bit ? 1.0f : -1.0f;
    res += Wj[idx] * val;
  }
  return res;
}

/* build a packed input for a polynomial instance: sample N_POINTS grid from -1..1, add noise, threshold at 0 */
static void sample_and_pack_example(const float coeffs[4], float noise_sigma, uint8_t *packed_out) {
  unsigned char bits[N_POINTS];
  for (int i = 0; i < N_POINTS; ++i) {
    float x = -1.0f + 2.0f * ((float)i) / (N_POINTS - 1);
    float y = eval_poly(x, coeffs);
    if (noise_sigma > 0.0f) y += gaussian_noise(noise_sigma);
    bits[i] = (y > 0.0f) ? 1 : 0;
  }
  pack_bits_from_bools(bits, N_POINTS, packed_out);
}

int main(void) {
  const int n = N_POINTS;
  const int ni = (n + 7) / 8;
  const int k = NUM_POLYS;
  const int train_total = k * EXAMPLES_PER_POLY;
  const int test_total  = k * TEST_PER_POLY;

  printf("Polynomial detector using ebnn header\n");
  printf("Input dimension (bits): %d  packed bytes: %d\n", n, ni);
  printf("Classes: %d   training examples: %d   test examples: %d\n", k, train_total, test_total);

  /* allocate training dataset (packed) */
  uint8_t *train_A = malloc(train_total * ni);
  if (!train_A) { perror("malloc"); return 1; }
  int *train_labels = malloc(train_total * sizeof(int));

  /* generate dataset: same grid of x for all, add noise to each example */
  float noise_sigma = 0.20f; /* noise amplitude; tune for difficulty */
  int idx = 0;
  for (int p = 0; p < k; ++p) {
    for (int e = 0; e < EXAMPLES_PER_POLY; ++e) {
      uint8_t *dst = train_A + idx * ni;
      sample_and_pack_example(polys[p], noise_sigma, dst);
      train_labels[idx] = p;
      idx++;
    }
  }

  /* allocate test dataset */
  uint8_t *test_A = malloc(test_total * ni);
  int *test_labels = malloc(test_total * sizeof(int));
  idx = 0;
  for (int p = 0; p < k; ++p) {
    for (int e = 0; e < TEST_PER_POLY; ++e) {
      uint8_t *dst = test_A + idx * ni;
      sample_and_pack_example(polys[p], noise_sigma, dst);
      test_labels[idx] = p;
      idx++;
    }
  }

  /* Float weights W (k x n), init small random */
  float *W = malloc(k * n * sizeof(float));
  if (!W) { perror("malloc W"); return 1; }
  for (int i = 0; i < k * n; ++i) W[i] = (float)((rand() % 1000) / 10000.0f - 0.05f);

  /* Training (simple multiclass perceptron style) */
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    int correct = 0;
    for (int ex = 0; ex < train_total; ++ex) {
      uint8_t *Aex = train_A + ex * ni;
      int label = train_labels[ex];
      /* compute scores */
      int pred = 0;
      float best = -INFINITY;
      for (int j = 0; j < k; ++j) {
        float sc = float_dot_weights_and_bitinput(W + j * n, Aex, n);
        if (sc > best) { best = sc; pred = j; }
      }
      if (pred == label) {
        correct++;
        continue;
      }
      /* update: push correct up, penalize predicted */
      for (int f = 0; f < n; ++f) {
        int bit = nthbitset_arr(Aex, f);
        float val = bit ? 1.0f : -1.0f;
        W[label * n + f] += LEARNING_RATE * val;
        W[pred * n + f]  -= LEARNING_RATE * val;
      }
    } /* examples */
    float acc = 100.0f * (float)correct / (float)train_total;
    if ((epoch % 5) == 0 || epoch == EPOCHS-1) {
      printf("Epoch %2d / %d   training acc: %.2f%% (%d/%d)\n", epoch+1, EPOCHS, acc, correct, train_total);
    }
  }

  /* Binarize weights into bit-packed filters expected by ebnn.h */
  uint8_t *F = malloc(k * ni);
  if (!F) { perror("malloc F"); return 1; }
  pack_filters_from_weights(W, k, n, F);

  /* prepare BN parameters as identity (no effect) */
  float *Bias  = calloc(k, sizeof(float));
  float *Gamma = malloc(k * sizeof(float));
  float *Beta  = calloc(k, sizeof(float));
  float *Mean  = calloc(k, sizeof(float));
  float *Std   = malloc(k * sizeof(float));
  for (int i = 0; i < k; ++i) { Gamma[i] = 1.0f; Std[i] = 1.0f; }

  /* Run inference on test set using blinear_sm_layer (it writes argmax per example into C) */
  uint8_t *C = malloc(test_total); /* blinear_sm_layer writes a byte result per input example */
  memset(C, 0, test_total);

  /* ebnn.h's blinear_sm_layer expects A as m * ni bytes (row-major) */
  blinear_sm_layer(test_A, F, C, Bias, Gamma, Beta, Mean, Std, test_total, n, k);

  /* evaluate accuracy */
  int correct = 0;
  for (int i = 0; i < test_total; ++i) {
    int pred = (int)C[i];
    if (pred == test_labels[i]) correct++;
  }
  printf("Test accuracy: %.2f%% (%d / %d)\n", 100.0f * correct / (float)test_total, correct, test_total);

  /* show a few test examples with predicted vs true */
  printf("\nSample results (true -> predicted):\n");
  for (int i = 0; i < 15 && i < test_total; ++i) {
    printf("%d -> %d\n", test_labels[i], (int)C[i]);
  }

  /* cleanup */
  free(train_A); free(train_labels);
  free(test_A); free(test_labels);
  free(W); free(F);
  free(Bias); free(Gamma); free(Beta); free(Mean); free(Std);
  free(C);
  return 0;
}
