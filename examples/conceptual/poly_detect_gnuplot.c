/*
 * poly_detect_gnuplot.c
 *
 * Train small CNN on polynomial sign-images (float training),
 * write matrices and training accuracy to files and call gnuplot
 * to produce PNG visualizations (accuracy curve, sample image,
 * conv filters, conv feature maps).
 *
 * Build:
 *   gcc -O2 poly_detect_gnuplot.c -o poly_detect_gnuplot -lm
 *
 * Run:
 *   ./poly_detect_gnuplot
 *
 * Requires: gnuplot binary available on PATH.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* image / model sizes */
#define IMG_W 8
#define IMG_H 8
#define IMG_SIZE (IMG_W*IMG_H)

#define CONV_F 4        /* number of conv filters */
#define K 3             /* filter kernel size */
#define K2 (K*K)

#define HIDDEN_N 8
#define NUM_CLASSES 4

/* training set sizes */
#define TRAIN_PER_CLASS 200
#define TEST_PER_CLASS 50
#define EPOCHS 30
#define LR 0.02f

/* polynomial patterns (4 classes) */
static const float polys[NUM_CLASSES][4] = {
  {0.0f, 0.0f, 1.0f, 0.0f},   /* x^2 */
  {0.0f, -1.0f, 0.0f, 1.0f},  /* x^3 - x */
  {1.0f, 2.0f, 1.0f, 0.0f},   /* 1 + 2x + x^2 */
  {-1.0f, 0.0f, 2.0f, 0.0f}   /* 2x^2 - 1 */
};

/* simple RNG helpers */
static float randf_unit() { return rand() / (float)RAND_MAX; }
static float randf_signed() { return randf_unit() * 2.0f - 1.0f; }

/* evaluate polynomial c0 + c1 x + c2 x^2 + c3 x^3 */
static float eval_poly(float x, const float c[4]) {
  float xp = 1.0f, res = 0.0f;
  for (int i = 0; i < 4; ++i) { res += c[i] * xp; xp *= x; }
  return res;
}

/* sample an 8x8 float image in {-1,+1} from polynomial */
static void sample_poly_image(const float coeffs[4], float noise_amp, float *dst) {
  for (int i = 0; i < IMG_H; ++i) {
    float xv = -1.0f + 2.0f * i / (IMG_H - 1);
    for (int j = 0; j < IMG_W; ++j) {
      float yv = -1.0f + 2.0f * j / (IMG_W - 1);
      /* combine coordinates to add texture */
      float v = eval_poly(xv + 0.5f * yv, coeffs) + noise_amp * randf_signed();
      dst[i * IMG_W + j] = (v > 0.0f) ? 1.0f : -1.0f;
    }
  }
}

/* Activation helpers */
static float tanh_act(float x) { return tanhf(x); }
static float tanh_deriv(float y) { return 1.0f - y*y; } /* expects y = tanh(x) */

/* ---------- Simple CNN forward/backprop (float) ---------- */

/* conv_forward: input (IMG_SIZE), weights Wconv [CONV_F * K2], biases bconv[CONV_F]
   output conv_out [CONV_F * outW * outH], where outW = IMG_W - K + 1 */
static void conv_forward(const float *in, const float *Wconv, const float *bconv, float *conv_out) {
  int outW = IMG_W - K + 1;
  int outH = IMG_H - K + 1;
  for (int f = 0; f < CONV_F; ++f) {
    for (int oy = 0; oy < outH; ++oy) {
      for (int ox = 0; ox < outW; ++ox) {
        float s = bconv[f];
        for (int ky = 0; ky < K; ++ky)
          for (int kx = 0; kx < K; ++kx) {
            int ix = ox + kx;
            int iy = oy + ky;
            s += Wconv[f*K2 + ky*K + kx] * in[iy*IMG_W + ix];
          }
        conv_out[f*outW*outH + oy*outW + ox] = tanh_act(s);
      }
    }
  }
}

/* conv_backward: compute gradWconv and gradbconv given grad_out (dL/d(conv_out)) and input */
static void conv_backward(const float *in, const float *grad_out, float *gradWconv, float *gradbconv) {
  int outW = IMG_W - K + 1;
  int outH = IMG_H - K + 1;
  memset(gradWconv, 0, sizeof(float)*CONV_F*K2);
  memset(gradbconv, 0, sizeof(float)*CONV_F);
  for (int f = 0; f < CONV_F; ++f) {
    for (int oy = 0; oy < outH; ++oy) {
      for (int ox = 0; ox < outW; ++ox) {
        float g = grad_out[f*outW*outH + oy*outW + ox];
        gradbconv[f] += g;
        for (int ky = 0; ky < K; ++ky)
          for (int kx = 0; kx < K; ++kx) {
            gradWconv[f*K2 + ky*K + kx] += g * in[(oy+ky)*IMG_W + (ox+kx)];
          }
      }
    }
  }
}

/* linear forward: in_n -> out_n */
static void linear_forward(const float *in, const float *W, const float *b, int in_n, int out_n, float *out) {
  for (int o = 0; o < out_n; ++o) {
    float s = b[o];
    for (int i = 0; i < in_n; ++i) s += W[o*in_n + i] * in[i];
    out[o] = tanh_act(s);
  }
}

/* linear_backward: compute gradW and gradb given grad_out and inputs */
static void linear_backward(const float *in, const float *grad_out, float *gradW, float *gradb, int in_n, int out_n) {
  memset(gradW, 0, sizeof(float) * out_n * in_n);
  memset(gradb, 0, sizeof(float) * out_n);
  for (int o = 0; o < out_n; ++o) {
    gradb[o] += grad_out[o];
    for (int i = 0; i < in_n; ++i) gradW[o*in_n + i] += grad_out[o] * in[i];
  }
}

/* softmax + cross-entropy gradient: z values are raw scores (we use them directly) */
static void softmax_grad(const float *z, int label, float *grad_out, int n) {
  float maxv = z[0];
  for (int i = 1; i < n; ++i) if (z[i] > maxv) maxv = z[i];
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) { grad_out[i] = expf(z[i] - maxv); sum += grad_out[i]; }
  for (int i = 0; i < n; ++i) grad_out[i] = grad_out[i] / sum;
  grad_out[label] -= 1.0f;
}

/* ---------- helpers for saving matrices and calling gnuplot ---------- */

/* write matrix (rows x cols) to file as whitespace-separated rows (suitable for 'plot "f" matrix with image') */
static void write_matrix(const char *fname, const float *mat, int rows, int cols) {
  FILE *f = fopen(fname, "w");
  if (!f) { perror(fname); return; }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      fprintf(f, "%g", mat[r*cols + c]);
      if (c + 1 < cols) fputc(' ', f);
    }
    fputc('\n', f);
  }
  fclose(f);
}

/* write 2-column data for accuracy (epoch,acc) */
static void write_xy(const char *fname, const float *x, const float *y, int N) {
  FILE *f = fopen(fname, "w");
  if (!f) { perror(fname); return; }
  for (int i = 0; i < N; ++i) fprintf(f, "%g %g\n", x[i], y[i]);
  fclose(f);
}

/* call gnuplot and send script through pipe */
static void call_gnuplot_script(const char *script) {
  FILE *gp = popen("gnuplot", "w");
  if (!gp) { perror("popen gnuplot"); return; }
  fprintf(gp, "%s\n", script);
  fflush(gp);
  pclose(gp);
}

/* create PNG of matrix using gnuplot 'matrix' data and 'with image' */
static void gplot_matrix_png(const char *datafile, const char *outfile, int rows, int cols, const char *title) {
  char script[4096];
  snprintf(script, sizeof(script),
    "set terminal pngcairo size 480,480 enhanced font 'DejaVuSans,10'\n"
    "set output '%s'\n"
    "set title '%s'\n"
    "unset key\n"
    "set xtics out nomirror\n"
    "set ytics out nomirror\n"
    "set size square\n"
    "set palette rgbformulae 33,13,10\n"
    "plot '%s' matrix with image\n", outfile, title, datafile);
  call_gnuplot_script(script);
}

/* create line plot PNG */
static void gplot_xy_png(const char *datafile, const char *outfile, const char *title, const char *ylabel) {
  char script[2048];
  snprintf(script, sizeof(script),
    "set terminal pngcairo size 640,320 enhanced font 'DejaVuSans,10'\n"
    "set output '%s'\n"
    "set title '%s'\n"
    "set xlabel 'Epoch'\n"
    "set ylabel '%s'\n"
    "set grid\n"
    "plot '%s' using 1:2 with lines lw 2 lc rgb 'blue' title ''\n",
    outfile, title, ylabel, datafile);
  call_gnuplot_script(script);
}

/* ---------- main: create data, train model, save visualizations ---------- */

int main(void) {
  srand((unsigned)time(NULL));
  printf("Training small CNN and visualizing with gnuplot...\n");

  const int train_total = NUM_CLASSES * TRAIN_PER_CLASS;
  const int test_total  = NUM_CLASSES * TEST_PER_CLASS;

  /* allocate datasets (float images where pixels in {-1,+1}) */
  float *trainX = malloc(sizeof(float) * train_total * IMG_SIZE);
  int *trainY    = malloc(sizeof(int) * train_total);
  float *testX  = malloc(sizeof(float) * test_total * IMG_SIZE);
  int *testY     = malloc(sizeof(int) * test_total);

  if (!trainX || !trainY || !testX || !testY) { perror("malloc"); return 1; }

  /* generate train/test examples */
  int idx = 0;
  for (int c = 0; c < NUM_CLASSES; ++c)
    for (int e = 0; e < TRAIN_PER_CLASS; ++e) {
      sample_poly_image(polys[c], 0.15f, &trainX[idx * IMG_SIZE]);
      trainY[idx++] = c;
    }
  idx = 0;
  for (int c = 0; c < NUM_CLASSES; ++c)
    for (int e = 0; e < TEST_PER_CLASS; ++e) {
      sample_poly_image(polys[c], 0.15f, &testX[idx * IMG_SIZE]);
      testY[idx++] = c;
    }

  /* model parameters (float) */
  float *Wconv = malloc(sizeof(float) * CONV_F * K2);
  float *bconv = malloc(sizeof(float) * CONV_F);
  int outW = IMG_W - K + 1;
  int outH = IMG_H - K + 1;
  int conv_out_size = CONV_F * outW * outH;

  float *Wh1 = malloc(sizeof(float) * HIDDEN_N * conv_out_size);
  float *bh1 = malloc(sizeof(float) * HIDDEN_N);
  float *Wh2 = malloc(sizeof(float) * NUM_CLASSES * HIDDEN_N);
  float *bh2 = malloc(sizeof(float) * NUM_CLASSES);

  if (!Wconv || !bconv || !Wh1 || !bh1 || !Wh2 || !bh2) { perror("malloc model"); return 1; }

  /* initialize */
  for (int i = 0; i < CONV_F*K2; ++i) Wconv[i] = 0.2f * randf_signed();
  for (int i = 0; i < CONV_F; ++i) bconv[i] = 0.0f;
  for (int i = 0; i < HIDDEN_N*conv_out_size; ++i) Wh1[i] = 0.2f * randf_signed();
  for (int i = 0; i < HIDDEN_N; ++i) bh1[i] = 0;
  for (int i = 0; i < NUM_CLASSES*HIDDEN_N; ++i) Wh2[i] = 0.2f * randf_signed();
  for (int i = 0; i < NUM_CLASSES; ++i) bh2[i] = 0;

  /* storage for training stats */
  float *ep = malloc(sizeof(float) * EPOCHS);
  float *acc = malloc(sizeof(float) * EPOCHS);

  /* training loop (SGD) */
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    int correct = 0;
    for (int ex = 0; ex < train_total; ++ex) {
      float *x = &trainX[ex * IMG_SIZE];
      int label = trainY[ex];

      /* forward */
      float *conv_out = malloc(sizeof(float) * conv_out_size);
      conv_forward(x, Wconv, bconv, conv_out);

      float *h1 = malloc(sizeof(float) * HIDDEN_N);
      linear_forward(conv_out, Wh1, bh1, conv_out_size, HIDDEN_N, h1);

      /* prepare last-layer "raw" scores (before tanh != ideal for softmax, but workable) */
      float scores[NUM_CLASSES];
      for (int c = 0; c < NUM_CLASSES; ++c) {
        float s = bh2[c];
        for (int i = 0; i < HIDDEN_N; ++i) s += Wh2[c*HIDDEN_N + i] * h1[i];
        scores[c] = s; /* raw */
      }

      /* prediction */
      int pred = 0; float best = scores[0];
      for (int c = 1; c < NUM_CLASSES; ++c) if (scores[c] > best) { best = scores[c]; pred = c; }
      if (pred == label) ++correct;

      /* compute gradients */
      float grad_scores[NUM_CLASSES];
      softmax_grad(scores, label, grad_scores, NUM_CLASSES);

      /* backprop to Wh2, bh2 */
      float *gradWh2 = malloc(sizeof(float) * NUM_CLASSES * HIDDEN_N);
      float *gradbh2 = malloc(sizeof(float) * NUM_CLASSES);
      memset(gradWh2, 0, sizeof(float)*NUM_CLASSES*HIDDEN_N);
      memset(gradbh2, 0, sizeof(float)*NUM_CLASSES);
      for (int o = 0; o < NUM_CLASSES; ++o) {
        gradbh2[o] += grad_scores[o];
        for (int i = 0; i < HIDDEN_N; ++i)
          gradWh2[o*HIDDEN_N + i] += grad_scores[o] * h1[i];
      }

      /* grad into hidden h1 */
      float *grad_h1 = malloc(sizeof(float) * HIDDEN_N);
      memset(grad_h1, 0, sizeof(float)*HIDDEN_N);
      for (int i = 0; i < HIDDEN_N; ++i) {
        float s = 0.0f;
        for (int o = 0; o < NUM_CLASSES; ++o) s += grad_scores[o] * Wh2[o*HIDDEN_N + i];
        grad_h1[i] = s * tanh_deriv(h1[i]);
      }

      /* grad for Wh1/bh1 */
      float *gradWh1 = malloc(sizeof(float) * HIDDEN_N * conv_out_size);
      float *gradbh1 = malloc(sizeof(float) * HIDDEN_N);
      memset(gradWh1, 0, sizeof(float)*HIDDEN_N*conv_out_size);
      memset(gradbh1, 0, sizeof(float)*HIDDEN_N);
      for (int o = 0; o < HIDDEN_N; ++o) {
        gradbh1[o] += grad_h1[o];
        for (int i = 0; i < conv_out_size; ++i)
          gradWh1[o*conv_out_size + i] += grad_h1[o] * conv_out[i];
      }

      /* grad into conv_out (before tanh) */
      float *grad_conv_out = malloc(sizeof(float) * conv_out_size);
      memset(grad_conv_out, 0, sizeof(float)*conv_out_size);
      for (int i = 0; i < conv_out_size; ++i) {
        float s = 0.0f;
        for (int o = 0; o < HIDDEN_N; ++o) s += grad_h1[o] * Wh1[o*conv_out_size + i];
        grad_conv_out[i] = s * (1.0f - conv_out[i]*conv_out[i]); /* derivative of tanh */
      }

      /* conv backward */
      float *gradWconv = malloc(sizeof(float) * CONV_F * K2);
      float *gradbconv = malloc(sizeof(float) * CONV_F);
      conv_backward(x, grad_conv_out, gradWconv, gradbconv);

      /* SGD updates */
      for (int i = 0; i < CONV_F*K2; ++i) Wconv[i] -= LR * gradWconv[i];
      for (int i = 0; i < CONV_F; ++i) bconv[i] -= LR * gradbconv[i];
      for (int i = 0; i < HIDDEN_N*conv_out_size; ++i) Wh1[i] -= LR * gradWh1[i];
      for (int i = 0; i < HIDDEN_N; ++i) bh1[i] -= LR * gradbh1[i];
      for (int i = 0; i < NUM_CLASSES*HIDDEN_N; ++i) Wh2[i] -= LR * gradWh2[i];
      for (int i = 0; i < NUM_CLASSES; ++i) bh2[i] -= LR * gradbh2[i];

      /* free temporary grads */
      free(conv_out);
      free(h1);
      free(gradWh2); free(gradbh2);
      free(grad_h1);
      free(gradWh1); free(gradbh1);
      free(grad_conv_out);
      free(gradWconv); free(gradbconv);
    } /* end examples */

    float train_acc = 100.0f * (float)correct / (float)train_total;
    ep[epoch] = (float)(epoch+1);
    acc[epoch] = train_acc;
    printf("Epoch %2d/%d  train accuracy: %.2f%% (%d/%d)\n", epoch+1, EPOCHS, train_acc, correct, train_total);
  } /* epochs */

  /* Save training accuracy file */
  write_xy("accuracy.dat", ep, acc, EPOCHS);
  gplot_xy_png("accuracy.dat", "accuracy.png", "Training accuracy", "Accuracy (%)");
  printf("Saved accuracy.png\n");

  /* Pick a test sample and compute conv_out & feature maps */
  float *sample = malloc(sizeof(float) * IMG_SIZE);
  memcpy(sample, &testX[0], sizeof(float) * IMG_SIZE);
  float *conv_out_sample = malloc(sizeof(float) * conv_out_size);
  conv_forward(sample, Wconv, bconv, conv_out_sample);

  /* Save sample image as 8x8 matrix (values -1..+1 -> map to 0..1 for plot aesthetics) */
  float sample_mat[IMG_SIZE];
  for (int i = 0; i < IMG_SIZE; ++i) sample_mat[i] = (sample[i] + 1.0f) / 2.0f;
  write_matrix("sample_X.dat", sample_mat, IMG_H, IMG_W);
  gplot_matrix_png("sample_X.dat", "sample.png", IMG_H, IMG_W, "Sample input (0..1)");
  printf("Saved sample.png\n");

  /* Save conv filters as KxK matrices and feature maps (outH x outW) */
  for (int f = 0; f < CONV_F; ++f) {
    float filter_mat[K2];
    for (int i = 0; i < K2; ++i) filter_mat[i] = Wconv[f*K2 + i];
    char fname[128], outpng[128], title[128];
    snprintf(fname, sizeof(fname), "filter_%d.dat", f);
    snprintf(outpng, sizeof(outpng), "filter_%d.png", f);
    snprintf(title, sizeof(title), "Conv filter %d (float weights)", f);
    write_matrix(fname, filter_mat, K, K);
    gplot_matrix_png(fname, outpng, K, K, title);
    printf("Saved %s\n", outpng);

    /* feature map */
    int fmapW = outW, fmapH = outH;
    float fmap[fmapW * fmapH];
    for (int r = 0; r < fmapH; ++r)
      for (int c = 0; c < fmapW; ++c)
        fmap[r*fmapW + c] = conv_out_sample[f* fmapW * fmapH + r*fmapW + c];
    snprintf(fname, sizeof(fname), "feat_%d.dat", f);
    snprintf(outpng, sizeof(outpng), "feat_%d.png", f);
    snprintf(title, sizeof(title), "Feature map %d (tanh)", f);
    write_matrix(fname, fmap, fmapH, fmapW);
    gplot_matrix_png(fname, outpng, fmapH, fmapW, title);
    printf("Saved %s\n", outpng);
  }

  /* cleanup */
  free(trainX); free(trainY); free(testX); free(testY);
  free(Wconv); free(bconv); free(Wh1); free(bh1); free(Wh2); free(bh2);
  free(ep); free(acc);
  free(sample); free(conv_out_sample);

  printf("All PNGs generated in current directory: accuracy.png, sample.png, filter_*.png, feat_*.png\n");
  printf("Open them with your image viewer.\n");
  return 0;
}
