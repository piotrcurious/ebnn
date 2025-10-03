/*
 * poly_detect_gnuplot_bin.c
 *
 * Train float CNN, quantize to binary, run binary inference via ebnn.h,
 * and visualize binary filters + feature maps using gnuplot.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ebnn.h"  // binary ops header

#define IMG_W 8
#define IMG_H 8
#define IMG_SIZE (IMG_W*IMG_H)
#define CONV_F 4
#define K 3
#define K2 (K*K)
#define HIDDEN_N 8
#define NUM_CLASSES 4
#define TRAIN_PER_CLASS 200
#define TEST_PER_CLASS 50
#define EPOCHS 25
#define LR 0.02f

static const float polys[NUM_CLASSES][4] = {
  {0.0f, 0.0f, 1.0f, 0.0f},
  {0.0f, -1.0f, 0.0f, 1.0f},
  {1.0f, 2.0f, 1.0f, 0.0f},
  {-1.0f, 0.0f, 2.0f, 0.0f}
};

static float randf_unit() { return rand() / (float)RAND_MAX; }
static float randf_signed() { return randf_unit()*2.0f - 1.0f; }
static float eval_poly(float x, const float c[4]) {
  float xp=1.0f, r=0.0f;
  for(int i=0;i<4;i++){r+=c[i]*xp;xp*=x;}
  return r;
}

static void sample_poly_image(const float coeffs[4], float noise_amp, float *dst) {
  for (int i=0;i<IMG_H;i++){
    float xv=-1.0f+2.0f*i/(IMG_H-1);
    for(int j=0;j<IMG_W;j++){
      float yv=-1.0f+2.0f*j/(IMG_W-1);
      float v=eval_poly(xv+0.5f*yv, coeffs)+noise_amp*randf_signed();
      dst[i*IMG_W+j]=(v>0.0f)?1.0f:-1.0f;
    }
  }
}

static float tanh_act(float x){return tanhf(x);}
static float tanh_deriv(float y){return 1.0f-y*y;}

/* matrix saving helpers */
static void write_matrix(const char *fname,const float *mat,int rows,int cols){
 FILE *f=fopen(fname,"w");if(!f)return;
 for(int r=0;r<rows;r++){for(int c=0;c<cols;c++)fprintf(f,"%g ",mat[r*cols+c]);fputc('\n',f);}fclose(f);
}
static void call_gnuplot_script(const char *script){
 FILE *gp=popen("gnuplot","w");if(!gp)return;fprintf(gp,"%s\n",script);fflush(gp);pclose(gp);
}
static void gplot_matrix_png(const char *data,const char *out,const char *title){
 char s[1024];
 snprintf(s,sizeof(s),
  "set term pngcairo size 400,400;set out '%s';unset key;"
  "set title '%s';set size square;"
  "set palette rgbformulae 33,13,10;"
  "plot '%s' matrix with image",out,title,data);
 call_gnuplot_script(s);
}

/* conv forward float */
static void conv_forward(const float *in,const float *W,const float *b,float *out){
 int ow=IMG_W-K+1,oh=IMG_H-K+1;
 for(int f=0;f<CONV_F;f++){
  for(int oy=0;oy<oh;oy++)for(int ox=0;ox<ow;ox++){
    float s=b[f];
    for(int ky=0;ky<K;ky++)for(int kx=0;kx<K;kx++)
      s+=W[f*K2+ky*K+kx]*in[(oy+ky)*IMG_W+(ox+kx)];
    out[f*ow*oh+oy*ow+ox]=tanh_act(s);
  }
 }
}

int main(void){
 srand(time(NULL));
 printf("Training float CNN, then binary inference via ebnn.h...\n");

 int outW=IMG_W-K+1, outH=IMG_H-K+1, conv_size=CONV_F*outW*outH;
 int train_total=NUM_CLASSES*TRAIN_PER_CLASS;

 float *trainX=malloc(train_total*IMG_SIZE*sizeof(float));
 int *trainY=malloc(train_total*sizeof(int));
 int idx=0;
 for(int c=0;c<NUM_CLASSES;c++)
  for(int i=0;i<TRAIN_PER_CLASS;i++){
    sample_poly_image(polys[c],0.15f,&trainX[idx*IMG_SIZE]);
    trainY[idx++]=c;
  }

 float *Wc=malloc(CONV_F*K2*sizeof(float));
 float *bc=calloc(CONV_F,sizeof(float));
 float *Wh=malloc(HIDDEN_N*conv_size*sizeof(float));
 float *bh=calloc(HIDDEN_N,sizeof(float));
 float *Wo=malloc(NUM_CLASSES*HIDDEN_N*sizeof(float));
 float *bo=calloc(NUM_CLASSES,sizeof(float));
 for(int i=0;i<CONV_F*K2;i++)Wc[i]=0.2f*randf_signed();
 for(int i=0;i<HIDDEN_N*conv_size;i++)Wh[i]=0.2f*randf_signed();
 for(int i=0;i<NUM_CLASSES*HIDDEN_N;i++)Wo[i]=0.2f*randf_signed();

 float *conv_out=malloc(conv_size*sizeof(float));
 float *h=malloc(HIDDEN_N*sizeof(float));
 for(int ep=0;ep<EPOCHS;ep++){
  int correct=0;
  for(int ex=0;ex<train_total;ex++){
    float *x=&trainX[ex*IMG_SIZE];
    int label=trainY[ex];
    conv_forward(x,Wc,bc,conv_out);
    for(int i=0;i<HIDDEN_N;i++){
      float s=bh[i];
      for(int j=0;j<conv_size;j++)s+=Wh[i*conv_size+j]*conv_out[j];
      h[i]=tanh_act(s);
    }
    float z[NUM_CLASSES];
    for(int o=0;o<NUM_CLASSES;o++){
      float s=bo[o];
      for(int i=0;i<HIDDEN_N;i++)s+=Wo[o*HIDDEN_N+i]*h[i];
      z[o]=s;
    }
    int pred=0;float best=z[0];for(int i=1;i<NUM_CLASSES;i++)if(z[i]>best){best=z[i];pred=i;}
    if(pred==label)correct++;
    float gradz[NUM_CLASSES];
    float sumexp=0,maxv=z[0];for(int i=1;i<NUM_CLASSES;i++)if(z[i]>maxv)maxv=z[i];
    for(int i=0;i<NUM_CLASSES;i++){gradz[i]=expf(z[i]-maxv);sumexp+=gradz[i];}
    for(int i=0;i<NUM_CLASSES;i++){gradz[i]/=sumexp;gradz[i]-=(i==label);}
    for(int o=0;o<NUM_CLASSES;o++){bo[o]-=LR*gradz[o];for(int i=0;i<HIDDEN_N;i++)Wo[o*HIDDEN_N+i]-=LR*gradz[o]*h[i];}
    float gradh[HIDDEN_N]={0};
    for(int i=0;i<HIDDEN_N;i++){float s=0;for(int o=0;o<NUM_CLASSES;o++)s+=gradz[o]*Wo[o*HIDDEN_N+i];gradh[i]=s*tanh_deriv(h[i]);}
    for(int i=0;i<HIDDEN_N;i++){bh[i]-=LR*gradh[i];for(int j=0;j<conv_size;j++)Wh[i*conv_size+j]-=LR*gradh[i]*conv_out[j];}
    float gradc[conv_size]={0};
    for(int j=0;j<conv_size;j++){float s=0;for(int i=0;i<HIDDEN_N;i++)s+=gradh[i]*Wh[i*conv_size+j];gradc[j]=s*(1-conv_out[j]*conv_out[j]);}
    int ow=outW,oh=outH;
    for(int f=0;f<CONV_F;f++){
      float gb=0;
      for(int oy=0;oy<oh;oy++)for(int ox=0;ox<ow;ox++){
        float g=gradc[f*ow*oh+oy*ow+ox];gb+=g;
        for(int ky=0;ky<K;ky++)for(int kx=0;kx<K;kx++)
          Wc[f*K2+ky*K+kx]-=LR*g*x[(oy+ky)*IMG_W+(ox+kx)];
      }
      bc[f]-=LR*gb;
    }
  }
  printf("Epoch %d acc=%.1f%%\n",ep+1,100.0f*correct/train_total);
 }

 /* Quantize weights -> binary */
 float *Wbconv=malloc(CONV_F*K2*sizeof(float));
 float *Bbconv=malloc(CONV_F*sizeof(float));
 float *Wbhid=malloc(HIDDEN_N*conv_size*sizeof(float));
 float *Bbhid=malloc(HIDDEN_N*sizeof(float));
 float *Wbout=malloc(NUM_CLASSES*HIDDEN_N*sizeof(float));
 float *Bbout=malloc(NUM_CLASSES*sizeof(float));
 for(int i=0;i<CONV_F*K2;i++)Wbconv[i]=signf(Wc[i]);
 memcpy(Bbconv,bc,sizeof(float)*CONV_F);
 for(int i=0;i<HIDDEN_N*conv_size;i++)Wbhid[i]=signf(Wh[i]);
 memcpy(Bbhid,bh,sizeof(float)*HIDDEN_N);
 for(int i=0;i<NUM_CLASSES*HIDDEN_N;i++)Wbout[i]=signf(Wo[i]);
 memcpy(Bbout,bo,sizeof(float)*NUM_CLASSES);

 /* Binary inference on one sample */
 float sample[IMG_SIZE]; sample_poly_image(polys[2],0.0f,sample);
 int outWb=IMG_W-K+1,outHb=IMG_H-K+1;
 float *featb=malloc(CONV_F*outWb*outHb*sizeof(float));
 bconv_layer(sample,Wbconv,Bbconv,IMG_W,IMG_H,K,CONV_F,featb);

 /* save binary filters & features */
 for(int f=0;f<CONV_F;f++){
   float filt[K2];for(int i=0;i<K2;i++)filt[i]=(Wbconv[f*K2+i]+1.0f)/2.0f;
   char fn[64];sprintf(fn,"filter_bin_%d.dat",f);
   write_matrix(fn,filt,K,K);
   char outpng[64];sprintf(outpng,"filter_bin_%d.png",f);
   gplot_matrix_png(fn,outpng,"Binary conv filter");
   float fmap[outWb*outHb];
   for(int i=0;i<outWb*outHb;i++)fmap[i]=(featb[f*outWb*outHb+i]+1.0f)/2.0f;
   sprintf(fn,"feat_bin_%d.dat",f);
   write_matrix(fn,fmap,outHb,outWb);
   sprintf(outpng,"feat_bin_%d.png",f);
   gplot_matrix_png(fn,outpng,"Binary feature map");
 }

 printf("Binary feature maps and filters visualized via gnuplot PNGs.\n");
 return 0;
}
