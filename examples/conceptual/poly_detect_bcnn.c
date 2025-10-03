/*
 * poly_detect_bcnn.c
 *
 * True-gradient binary convolutional neural network for polynomial pattern recognition.
 * 
 * Requires: ebnn.h
 * Build: gcc -O2 poly_detect_bcnn.c -lm -o poly_detect_bcnn
 * Run:   ./poly_detect_bcnn
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

#define CONV_F 4
#define K 3
#define K2 (K*K)

#define HIDDEN_N 8
#define NUM_CLASSES 4
#define TRAIN_PER_CLASS 200
#define TEST_PER_CLASS 50
#define EPOCHS 30
#define LR 0.02f

static const float polys[NUM_CLASSES][4] = {
  {0.0f, 0.0f, 1.0f, 0.0f},   // x^2
  {0.0f, -1.0f, 0.0f, 1.0f},  // x^3 - x
  {1.0f, 2.0f, 1.0f, 0.0f},   // 1 + 2x + x^2
  {-1.0f, 0.0f, 2.0f, 0.0f}   // 2x^2 - 1
};

static float eval_poly(float x, const float c[4]) {
  float res = 0, xp = 1;
  for (int i = 0; i < 4; ++i) { res += c[i]*xp; xp *= x; }
  return res;
}

static float randf() { return (rand() / (float)RAND_MAX) * 2 - 1; }

static void sample_poly_image(const float coeffs[4], float noise, float *dst) {
  for (int i=0;i<IMG_H;i++) {
    float x = -1.0f + 2.0f*i/(IMG_H-1);
    for (int j=0;j<IMG_W;j++) {
      float y = -1.0f + 2.0f*j/(IMG_W-1);
      float v = eval_poly(x + y*0.5f, coeffs) + noise*randf();
      dst[i*IMG_W+j] = v>0?1.0f:-1.0f;
    }
  }
}

/* Forward convolution (float) */
static void conv_forward(const float *in, const float *W, const float *b, float *out) {
  int outW = IMG_W - K + 1;
  int outH = IMG_H - K + 1;
  for (int f=0; f<CONV_F; ++f)
    for (int y=0; y<outH; ++y)
      for (int x=0; x<outW; ++x) {
        float s=b[f];
        for (int ky=0;ky<K;ky++)
          for (int kx=0;kx<K;kx++)
            s += W[f*K2+ky*K+kx]*in[(y+ky)*IMG_W+(x+kx)];
        out[f*outW*outH+y*outW+x] = tanhf(s);
      }
}

/* Backprop convolution */
static void conv_backward(const float *in, const float *grad_out, float *gradW, float *gradb) {
  int outW = IMG_W - K + 1;
  int outH = IMG_H - K + 1;
  memset(gradW,0,sizeof(float)*CONV_F*K2);
  memset(gradb,0,sizeof(float)*CONV_F);
  for (int f=0;f<CONV_F;f++) {
    for (int y=0;y<outH;y++)
      for (int x=0;x<outW;x++) {
        float g = grad_out[f*outW*outH+y*outW+x];
        gradb[f]+=g;
        for (int ky=0;ky<K;ky++)
          for (int kx=0;kx<K;kx++)
            gradW[f*K2+ky*K+kx] += g*in[(y+ky)*IMG_W+(x+kx)];
      }
  }
}

/* Linear layer forward */
static void linear_forward(const float *in, const float *W, const float *b,
                           int in_n, int out_n, float *out) {
  for (int o=0;o<out_n;o++){
    float s=b[o];
    for (int i=0;i<in_n;i++) s+=W[o*in_n+i]*in[i];
    out[o]=tanhf(s);
  }
}

/* Linear layer backward */
static void linear_backward(const float *in, const float *W,
                            const float *grad_out, float *gradW,
                            float *gradb, int in_n, int out_n) {
  memset(gradW,0,sizeof(float)*out_n*in_n);
  memset(gradb,0,sizeof(float)*out_n);
  for (int o=0;o<out_n;o++){
    gradb[o]+=grad_out[o];
    for (int i=0;i<in_n;i++)
      gradW[o*in_n+i]+=grad_out[o]*in[i];
  }
}

/* Cross-entropy softmax gradient (simple) */
static void softmax_grad(const float *z, int label, float *grad, int n) {
  float sum=0;
  for(int i=0;i<n;i++) sum+=expf(z[i]);
  for(int i=0;i<n;i++) grad[i]=expf(z[i])/sum;
  grad[label]-=1.0f;
}

int main(void) {
  srand((unsigned)time(NULL));
  printf("True-gradient Binary CNN for Polynomial Detection\n");

  int outW = IMG_W - K + 1;
  int outH = IMG_H - K + 1;
  int conv_out_size = outW*outH*CONV_F;

  /* allocate */
  float *trainX = malloc(NUM_CLASSES*TRAIN_PER_CLASS*IMG_SIZE*sizeof(float));
  int *trainY = malloc(NUM_CLASSES*TRAIN_PER_CLASS*sizeof(int));
  float *testX = malloc(NUM_CLASSES*TEST_PER_CLASS*IMG_SIZE*sizeof(float));
  int *testY = malloc(NUM_CLASSES*TEST_PER_CLASS*sizeof(int));

  for(int c=0,idx=0;c<NUM_CLASSES;c++)
    for(int e=0;e<TRAIN_PER_CLASS;e++){
      sample_poly_image(polys[c],0.2f,&trainX[idx*IMG_SIZE]);
      trainY[idx++]=c;
    }

  for(int c=0,idx=0;c<NUM_CLASSES;c++)
    for(int e=0;e<TEST_PER_CLASS;e++){
      sample_poly_image(polys[c],0.2f,&testX[idx*IMG_SIZE]);
      testY[idx++]=c;
    }

  /* weights */
  float Wconv[CONV_F*K2], bconv[CONV_F];
  float Wh1[HIDDEN_N*conv_out_size], bh1[HIDDEN_N];
  float Wh2[NUM_CLASSES*HIDDEN_N], bh2[NUM_CLASSES];
  for(int i=0;i<CONV_F*K2;i++) Wconv[i]=0.2f*randf();
  for(int i=0;i<HIDDEN_N*conv_out_size;i++) Wh1[i]=0.2f*randf();
  for(int i=0;i<NUM_CLASSES*HIDDEN_N;i++) Wh2[i]=0.2f*randf();
  memset(bconv,0,sizeof(bconv)); memset(bh1,0,sizeof(bh1)); memset(bh2,0,sizeof(bh2));

  /* training */
  for(int epoch=0;epoch<EPOCHS;epoch++){
    int correct=0;
    for(int n=0;n<NUM_CLASSES*TRAIN_PER_CLASS;n++){
      /* forward */
      float conv_out[conv_out_size];
      conv_forward(&trainX[n*IMG_SIZE],Wconv,bconv,conv_out);
      float h1[HIDDEN_N]; linear_forward(conv_out,Wh1,bh1,conv_out_size,HIDDEN_N,h1);
      float out[NUM_CLASSES]; linear_forward(h1,Wh2,bh2,HIDDEN_N,NUM_CLASSES,out);

      /* loss grad */
      float grad_out[NUM_CLASSES];
      softmax_grad(out,trainY[n],grad_out,NUM_CLASSES);

      /* accuracy */
      int pred=0; float maxv=-1e9;
      for(int i=0;i<NUM_CLASSES;i++){ if(out[i]>maxv){maxv=out[i];pred=i;} }
      if(pred==trainY[n]) correct++;

      /* backprop */
      float grad_Wh2[NUM_CLASSES*HIDDEN_N], grad_bh2[NUM_CLASSES];
      linear_backward(h1,Wh2,grad_out,grad_Wh2,grad_bh2,HIDDEN_N,NUM_CLASSES);

      float grad_h1[HIDDEN_N]={0};
      for(int i=0;i<HIDDEN_N;i++)
        for(int o=0;o<NUM_CLASSES;o++)
          grad_h1[i]+=grad_out[o]*Wh2[o*HIDDEN_N+i]*(1-h1[i]*h1[i]);

      float grad_Wh1[HIDDEN_N*conv_out_size], grad_bh1[HIDDEN_N];
      linear_backward(conv_out,Wh1,grad_h1,grad_Wh1,grad_bh1,conv_out_size,HIDDEN_N);

      float grad_conv_out[conv_out_size]={0};
      for(int i=0;i<conv_out_size;i++)
        for(int o=0;o<HIDDEN_N;o++)
          grad_conv_out[i]+=grad_h1[o]*Wh1[o*conv_out_size+i]*(1-conv_out[i]*conv_out[i]);

      float grad_Wconv[CONV_F*K2], grad_bconv[CONV_F];
      conv_backward(&trainX[n*IMG_SIZE],grad_conv_out,grad_Wconv,grad_bconv);

      /* update */
      for(int i=0;i<CONV_F*K2;i++) Wconv[i]-=LR*grad_Wconv[i];
      for(int i=0;i<CONV_F;i++) bconv[i]-=LR*grad_bconv[i];
      for(int i=0;i<HIDDEN_N*conv_out_size;i++) Wh1[i]-=LR*grad_Wh1[i];
      for(int i=0;i<HIDDEN_N;i++) bh1[i]-=LR*grad_bh1[i];
      for(int i=0;i<NUM_CLASSES*HIDDEN_N;i++) Wh2[i]-=LR*grad_Wh2[i];
      for(int i=0;i<NUM_CLASSES;i++) bh2[i]-=LR*grad_bh2[i];
    }
    printf("Epoch %d accuracy: %.2f%%\n",epoch+1,100.0f*correct/(NUM_CLASSES*TRAIN_PER_CLASS));
  }

  /* ---- Binarize & Run EBNN inference ---- */
  uint8_t Fconv[CONV_F*((K2+7)/8)];
  memset(Fconv,0,sizeof(Fconv));
  for(int f=0;f<CONV_F;f++){
    int shift=7,byte=f*((K2+7)/8);
    for(int i=0;i<K2;i++){
      int bit=(Wconv[f*K2+i]>=0);
      Fconv[byte]|=(bit<<shift);
      if(--shift<0){shift=7;byte++;}
    }
  }

  /* Pack linear weights */
  int hbytes=(conv_out_size+7)/8;
  uint8_t Fh1[HIDDEN_N*hbytes]; memset(Fh1,0,sizeof(Fh1));
  for(int o=0;o<HIDDEN_N;o++){
    int shift=7,byte=o*hbytes;
    for(int i=0;i<conv_out_size;i++){
      int bit=(Wh1[o*conv_out_size+i]>=0);
      Fh1[byte]|=(bit<<shift);
      if(--shift<0){shift=7;byte++;}
    }
  }

  int lbytes=(HIDDEN_N+7)/8;
  uint8_t Fh2[NUM_CLASSES*lbytes]; memset(Fh2,0,sizeof(Fh2));
  for(int o=0;o<NUM_CLASSES;o++){
    int shift=7,byte=o*lbytes;
    for(int i=0;i<HIDDEN_N;i++){
      int bit=(Wh2[o*HIDDEN_N+i]>=0);
      Fh2[byte]|=(bit<<shift);
      if(--shift<0){shift=7;byte++;}
    }
  }

  /* Identity BN for simplicity */
  float Bias[NUM_CLASSES]={0},Gamma[NUM_CLASSES],Beta[NUM_CLASSES]={0},Mean[NUM_CLASSES]={0},Std[NUM_CLASSES];
  for(int i=0;i<NUM_CLASSES;i++){Gamma[i]=1;Std[i]=1;}

  uint8_t pred_idx[NUM_CLASSES*TEST_PER_CLASS];
  bconv_layer((uint8_t*)testX,Fconv,NULL,Bias,Gamma,Beta,Mean,Std,
              IMG_W,IMG_H,1,CONV_F,K,K,1,1,0,0,
              IMG_W,IMG_H,1,1);
  blinear_sm_layer((uint8_t*)testX,Fh2,pred_idx,Bias,Gamma,Beta,Mean,Std,
                   NUM_CLASSES*TEST_PER_CLASS,HIDDEN_N,NUM_CLASSES);

  /* Evaluate */
  int correct=0;
  for(int i=0;i<NUM_CLASSES*TEST_PER_CLASS;i++)
    if(pred_idx[i]==testY[i]) correct++;

  printf("\nFinal Test Accuracy: %.2f%%\n",100.0f*correct/(NUM_CLASSES*TEST_PER_CLASS));

  free(trainX); free(trainY); free(testX); free(testY);
  return 0;
}
