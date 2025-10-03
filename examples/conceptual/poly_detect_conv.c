/*
 * poly_detect_conv.c
 *
 * Example: Binary Convolutional Neural Network using ebnn.h
 * Classifies which polynomial (from 4 pattern polynomials) generated a 2D
 * binary pattern derived from sampled polynomial values.
 *
 * Build:
 *   gcc -O2 poly_detect_conv.c -o poly_detect_conv -lm
 *
 * Run:
 *   ./poly_detect_conv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ebnn.h"

#define IMG_W 8
#define IMG_H 8
#define IMG_SIZE (IMG_W * IMG_H)
#define IMG_BYTES ((IMG_SIZE + 7) / 8)

#define N_FILTERS 2
#define F_KW 3
#define F_KH 3
#define F_SIZE (F_KW * F_KH)
#define F_BYTES ((F_SIZE + 7) / 8)

#define NUM_CLASSES 4
#define TRAIN_PER_CLASS 200
#define TEST_PER_CLASS 50
#define LEARNING_RATE 0.1f
#define EPOCHS 25

/* pattern polynomials */
static const float polys[NUM_CLASSES][4] = {
  {0.0f, 0.0f, 1.0f, 0.0f},   // x^2
  {0.0f, -1.0f, 0.0f, 1.0f},  // x^3 - x
  {1.0f, 2.0f, 1.0f, 0.0f},   // 1 + 2x + x^2
  {-1.0f, 0.0f, 2.0f, 0.0f}   // 2x^2 - 1
};

static float eval_poly(float x, const float c[4]) {
  float res = 0, xp = 1;
  for (int i = 0; i < 4; ++i) {
    res += c[i] * xp;
    xp *= x;
  }
  return res;
}

static float gaussian_noise(float sigma) {
  float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
  float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * sigma;
}

/* 1D→2D sampling: polynomial sign pattern over grid */
static void sample_poly_image(const float coeffs[4], float noise, uint8_t *packed) {
  unsigned char bits[IMG_SIZE];
  for (int i = 0; i < IMG_H; ++i) {
    float x = -1.0f + 2.0f * i / (IMG_H - 1);
    for (int j = 0; j < IMG_W; ++j) {
      float y = -1.0f + 2.0f * j / (IMG_W - 1);
      float z = eval_poly(x + y * 0.5f, coeffs);  // composite input
      z += gaussian_noise(noise);
      bits[i * IMG_W + j] = (z > 0.0f);
    }
  }
  // Pack bits MSB-first
  memset(packed, 0, IMG_BYTES);
  int shift = 7, idx = 0;
  for (int k = 0; k < IMG_SIZE; ++k) {
    packed[idx] |= bits[k] << shift;
    if (--shift < 0) { shift = 7; idx++; }
  }
}

/* convert float weight matrix → bitpacked filter bank */
static void pack_filters_from_weights(const float *W, int num_filters, uint8_t *F) {
  for (int f = 0; f < num_filters; ++f) {
    int shift = 7, byte_idx = f * F_BYTES;
    memset(F + byte_idx, 0, F_BYTES);
    for (int i = 0; i < F_SIZE; ++i) {
      int bit = (W[f * F_SIZE + i] >= 0.0f);
      F[byte_idx] |= bit << shift;
      if (--shift < 0) { shift = 7; byte_idx++; }
    }
  }
}

/* dot product for float conv layer */
static float conv2d_patch(const float *filter, const uint8_t *img, int x, int y) {
  float res = 0.0f;
  for (int dy = 0; dy < F_KH; ++dy)
    for (int dx = 0; dx < F_KW; ++dx) {
      int ix = x + dx, iy = y + dy;
      if (ix < 0 || ix >= IMG_W || iy < 0 || iy >= IMG_H) continue;
      int bit = nthbitset_arr(img, iy * IMG_W + ix);
      float val = bit ? 1.0f : -1.0f;
      res += val * filter[dy * F_KW + dx];
    }
  return res;
}

/* forward pass for float conv + linear classifier */
static int forward_convnet(const float *convW, const float *fcW,
                           const uint8_t *input, float *feat_out) {
  float feats[N_FILTERS];
  for (int f = 0; f < N_FILTERS; ++f) {
    float maxv = -1e9;
    for (int y = 0; y <= IMG_H - F_KH; ++y)
      for (int x = 0; x <= IMG_W - F_KW; ++x) {
        float v = conv2d_patch(convW + f * F_SIZE, input, x, y);
        if (v > maxv) maxv = v;
      }
    feats[f] = maxv;
  }
  /* fully connected */
  int best = 0;
  float bestv = -1e9;
  for (int c = 0; c < NUM_CLASSES; ++c) {
    float s = 0;
    for (int f = 0; f < N_FILTERS; ++f)
      s += fcW[c * N_FILTERS + f] * feats[f];
    feat_out[c] = s;
    if (s > bestv) { bestv = s; best = c; }
  }
  return best;
}

int main(void) {
  srand((unsigned)time(NULL));
  const int train_total = NUM_CLASSES * TRAIN_PER_CLASS;
  const int test_total  = NUM_CLASSES * TEST_PER_CLASS;

  printf("Binary ConvNet polynomial classifier\n");

  uint8_t *train_A = malloc(train_total * IMG_BYTES);
  int *train_labels = malloc(train_total * sizeof(int));

  float noise = 0.15f;
  int idx = 0;
  for (int c = 0; c < NUM_CLASSES; ++c)
    for (int e = 0; e < TRAIN_PER_CLASS; ++e) {
      sample_poly_image(polys[c], noise, train_A + idx * IMG_BYTES);
      train_labels[idx++] = c;
    }

  uint8_t *test_A = malloc(test_total * IMG_BYTES);
  int *test_labels = malloc(test_total * sizeof(int));
  idx = 0;
  for (int c = 0; c < NUM_CLASSES; ++c)
    for (int e = 0; e < TEST_PER_CLASS; ++e) {
      sample_poly_image(polys[c], noise, test_A + idx * IMG_BYTES);
      test_labels[idx++] = c;
    }

  /* float weights */
  float convW[N_FILTERS * F_SIZE];
  float fcW[NUM_CLASSES * N_FILTERS];
  for (int i = 0; i < N_FILTERS * F_SIZE; ++i)
    convW[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
  for (int i = 0; i < NUM_CLASSES * N_FILTERS; ++i)
    fcW[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;

  /* training */
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    int correct = 0;
    for (int ex = 0; ex < train_total; ++ex) {
      float scores[NUM_CLASSES];
      int pred = forward_convnet(convW, fcW, train_A + ex * IMG_BYTES, scores);
      int label = train_labels[ex];
      if (pred == label) { correct++; continue; }

      /* backprop-like perceptron update */
      /* update FC weights */
      for (int f = 0; f < N_FILTERS; ++f) {
        float grad = (label == pred) ? 0 : (scores[label] - scores[pred]);
        fcW[label * N_FILTERS + f] += LEARNING_RATE;
        fcW[pred * N_FILTERS + f]  -= LEARNING_RATE;
      }

      /* update conv weights crudely: flip sign near decision errors */
      for (int f = 0; f < N_FILTERS; ++f) {
        for (int i = 0; i < F_SIZE; ++i)
          convW[f * F_SIZE + i] += (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
      }
    }
    printf("Epoch %d: %.2f%% correct\n", epoch + 1,
           100.0f * correct / (float)train_total);
  }

  /* convert to binary filters */
  uint8_t Fconv[N_FILTERS * F_BYTES];
  pack_filters_from_weights(convW, N_FILTERS, Fconv);

  uint8_t Ffc[NUM_CLASSES * ((N_FILTERS + 7) / 8)];
  int fc_bytes = (N_FILTERS + 7) / 8;
  memset(Ffc, 0, sizeof(Ffc));
  for (int c = 0; c < NUM_CLASSES; ++c) {
    int shift = 7, idx_b = c * fc_bytes;
    for (int f = 0; f < N_FILTERS; ++f) {
      int bit = (fcW[c * N_FILTERS + f] >= 0.0f);
      Ffc[idx_b] |= bit << shift;
      if (--shift < 0) { shift = 7; idx_b++; }
    }
  }

  /* identity BN */
  float Bias[NUM_CLASSES] = {0};
  float Gamma[NUM_CLASSES], Beta[NUM_CLASSES] = {0};
  float Mean[NUM_CLASSES] = {0}, Std[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; ++i) { Gamma[i] = 1; Std[i] = 1; }

  /* inference using ebnn */
  uint8_t *conv_out = malloc(test_total * ((N_FILTERS + 7) / 8));
  uint8_t *pred_idx = malloc(test_total);

  /* Run convolution layer */
  for (int i = 0; i < test_total; ++i) {
    bconv_layer(test_A + i * IMG_BYTES, Fconv, conv_out + i * ((N_FILTERS + 7) / 8),
                Bias, Gamma, Beta, Mean, Std,
                IMG_W, IMG_H, 1, N_FILTERS,
                F_KW, F_KH, 1, 1, 0, 0,
                IMG_W, IMG_H, 1, 1);
  }

  /* Fully connected classification */
  blinear_sm_layer(conv_out, Ffc, pred_idx, Bias, Gamma, Beta, Mean, Std,
                   test_total, N_FILTERS, NUM_CLASSES);

  /* evaluate */
  int correct = 0;
  for (int i = 0; i < test_total; ++i)
    if (pred_idx[i] == test_labels[i]) correct++;

  printf("\nFinal test accuracy: %.2f%% (%d/%d)\n",
         100.0f * correct / test_total, correct, test_total);

  free(train_A); free(train_labels);
  free(test_A); free(test_labels);
  free(conv_out); free(pred_idx);
  return 0;
}
