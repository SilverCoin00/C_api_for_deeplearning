#ifndef KERAS_CORE_H
#define KERAS_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Python.h"

typedef struct Input {
    int* shape;
    int dim;
} Input;
typedef struct IFlatten {
} IFlatten;
typedef struct IDense {
    int nodes;
    char* activation;
} IDense;
typedef struct IDropout {
    float drop;
} IDropout;
typedef struct {
    int type;
    void* layer;
} Keras_layer;
typedef struct IConv {
    int channel;
    int row, col;
    int stride_row, stride_col;
    int padding_row, padding_col;
    char* activation;
} IConv;
typedef struct IMPool {
    int row, col;
    int stride_row, stride_col;
    int padding_row, padding_col;
} IMPool;
typedef struct IAPool {
    int row, col;
    int stride_row, stride_col;
    int padding_row, padding_col;
} IAPool;

#define Input_(...) &(Keras_layer){0, (float[]){__VA_ARGS__}}
#define Flatten_() &(Keras_layer){2, &(IFlatten){}}
#define Dense_(num_nodes, activate) &(Keras_layer){1, &(IDense){.nodes = num_nodes, .activation = activate}}
#define Dropout_(ratio) &(Keras_layer){-1, &(IDropout){.drop = ratio}}
#define Conv_(height, width, channels, ver_stride, hor_stride, pad, activate) &(Keras_layer){3, &(IConv){.row = height, .col = width, .channel = channels, .stride_row = ver_stride, .stride_col = hor_stride, .padding_row = pad, .padding_col = pad, .activation = activate}}
#define MaxPooling_(height, width, ver_stride, hor_stride, pad) &(Keras_layer){4, &(IMPool){.row = height, .col = width, .stride_row = ver_stride, .stride_col = hor_stride, .padding_row = pad, .padding_col = pad}}
#define AveragePooling_(height, width, ver_stride, hor_stride, pad) &(Keras_layer){5, &(IAPool){.row = height, .col = width, .stride_row = ver_stride, .stride_col = hor_stride, .padding_row = pad, .padding_col = pad}}
#define tf_keras_layers_(...) (Keras_layer*[]){__VA_ARGS__}
const char MODEL_KEY[] = "$tf_keras_sequential_model_file$";

typedef struct Weights {
    Matrix* w;            // front_num_units x cur_num_units
    float* b;             // cur_num_units x 1
} Weights;
typedef struct Flatten {
    int row, col, depth;
    Matrix* flatted;      // samples x num_features
} Flatten;
typedef struct Dense {
    Weights* weights;
    Matrix* a;            // sizeof(samples x num_units)
    Matrix* z;            // = sizeof(a)
    int activation;
    Weights* deriv;
    float* drop;          // drop[0] = num_drop, then drop_ratios
    Weights* pre_velo;
    Weights* acc_grad;
} Dense;

typedef struct Kernel {
    Tensor** w;           // out_channels x (row x col x in_channels)
    float* b;             // out_channels x 1
    int channel;
} Kernel;
typedef struct Conv {
    Kernel* filter;
    int out_channels;
    int activation;
    int* stride;          // [stride_row, stride_col]
    int* padding;         // [padding_row, padding_col]
    Tensor** a;           // batchsize x (row x col x out_channels)
    Tensor** z;           // = sizeof(a)
    Kernel* deriv;
    float* drop;
    Kernel* pre_velo;
    Kernel* acc_grad;
} Conv;
typedef struct MaxPooling {
    int row, col;         // pool_sizes
    int* stride;
    int* padding;
    Tensor** pool;        // pooled_data
    Tensor** mask_row;    // max_position_row
    Tensor** mask_col;    // max_position_col
    float* drop;
} MaxPooling;
typedef struct AveragePooling {
    int row, col;
    int* stride;
    int* padding;
    Tensor** pool;
    float* drop;
} AveragePooling;

typedef struct {
    int type;
    float learning_rate;
    float momentum;
    int nesterov;
    float beta_1, beta_2;
    float rho;
    float epsilon;
    float init_accumulator_grad;
} Optimizer;
typedef struct {
    int monitor;
    float baseline;
    int patience;
    float min_delta;
    int restore_best_weights;
    float last_monitor_val;
    float best_monitor_val;
} Early_Stopping;
typedef struct Model_Compiler {
    Optimizer* optimize;
    int loss_type, metrics_type;
} Model_Compiler;

typedef struct Sequential {
    int num_layers;
    Keras_layer** layer;
    Model_Compiler* compiler;
    int batch_size;
} Sequential;

#include "D:\Data\code_doc\Keras\Weights.h"
#include "D:\Data\code_doc\Keras\Kernels.h"
#include "D:\Data\code_doc\Keras\Optimizer.h"
#include "D:\Data\code_doc\Keras\Dense_layer.h"
#include "D:\Data\code_doc\Keras\Conv_layer.h"
#include "D:\Data\code_doc\Keras\Pooling_layer.h"
#include "D:\Data\code_doc\Keras\Reshaping_layer.h"
#include "D:\Data\code_doc\Keras\Sequential_class.h"

#endif
