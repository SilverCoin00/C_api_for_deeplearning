#pragma once
#include "D:\Data\code_doc\Keras\Keras_core.h"

Kernel* init_kernels(int row, int col, int out_channel, int in_channel, int random_state) {
    Kernel* newk = (Kernel*)malloc(sizeof(Kernel));
    newk->w = (Tensor**)malloc(out_channel* sizeof(Tensor*));
    int i, j, k, h;
    for (i = 0; i < out_channel; i++) 
        newk->w[i] = new_tensor(row, col, in_channel);
    newk->b = (float*)calloc(out_channel, sizeof(float));
    newk->channel = out_channel;
    if (random_state) {
        srand(random_state);
        for (i = 0; i < out_channel; i++) {
            newk->b[i] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
            for (j = 0; j < in_channel; j++) 
                for (k = 0; k < row; k++) 
                    for (h = 0; h < col; h++) newk->w[i]->mat[j]->val[k][h] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
        }
    }
    return newk;
}
void copy_kernels(Kernel* dest, Kernel* sauce) {
    if (dest->channel < sauce->channel) {
        printf("Warning: Inapproriate type of kernels !!");
        return ;
    }
    for (int i = 0; i < sauce->channel; i++) copy_tensor(dest->w[i], sauce->w[i]);
    copy_vector(dest->b, sauce->b, sauce->channel);
}
void fill_kernels(Kernel* filter, float fill_value) {
    if (!filter) return ;
    for (int i = 0; i < filter->channel; i++) {
        fill_tensor(filter->w[i], fill_value);
        filter->b[i] = fill_value;
    }
}
Kernel* get_copy_kernels(Kernel* sauce) {
    Kernel* newk = init_kernels(sauce->w[0]->mat[0]->row, sauce->w[0]->mat[0]->col, sauce->channel, sauce->w[0]->depth, 0);
    copy_kernels(newk, sauce);
    return newk;
}
Kernel* rot_kernels(Kernel* sauce) {
    Kernel* rot = (Kernel*)malloc(sizeof(Kernel));
    rot->channel = sauce->w[0]->depth;
    rot->w = (Tensor**)malloc(rot->channel* sizeof(Tensor*));
    int i, j, k, h;
    for (i = 0; i < rot->channel; i++) {
        rot->w[i] = new_tensor(sauce->w[0]->mat[0]->row, sauce->w[0]->mat[0]->col, sauce->channel);
        for (j = 0; j < sauce->channel; j++) get_rot_180m(rot->w[i]->mat[j], sauce->w[j]->mat[i]);
    }
    rot->b = (float*)calloc(rot->channel, sizeof(float));
    return rot;
}
void free_kernels(Kernel* k) {
    if (!k) return ;
    if (k->w) {
        for (int i = 0; i < k->channel; i++) if (k->w[i]) free_tensor(k->w[i]);
        free(k->w);
    }
    if (k->b) free(k->b);
    free(k);
}
int is_valid_point(Matrix* x, int row_point, int col_point) {
    return (row_point >= 0 && col_point >= 0 && row_point < x->row && col_point < x->col);
}
static float convolutional_point(Matrix* x, Matrix* w, int row_point, int col_point) {
    float y_point = 0;
    for (int i = 0, j, row, col; i < w->row; i++) 
        for (j = 0; j < w->col; j++) {
            row = row_point + i, col = col_point + j;
            if (is_valid_point(x, row, col)) y_point += w->val[i][j]* x->val[row][col];
        }
    return y_point;
}
static void convolutional_func(Matrix* y, Matrix* x, Matrix* w, int* stride, int* padding) {  // y = x @ w
    int start_row = -padding[0], start_col = -padding[1];
    for (int i = 0, ci = 0, j, cj; i < y->row; i++, ci += stride[0]) {
        for (j = 0, cj = 0; j < y->col; j++, cj += stride[1]) 
            y->val[i][j] += convolutional_point(x, w, start_row + ci, start_col + cj);
    }
}
static void plus_bias_2D(Matrix* xw, float bias) {
    for (int i = 0, j; i < xw->row; i++) 
        for (j = 0; j < xw->col; j++) xw->val[i][j] += bias;
}
Tensor* kernel_func(Kernel* filter, Tensor* x, int* stride, int* padding) {
    if (filter->w[0]->depth != x->depth) {
        printf("Warning: Inappropriate size of filters !!");
        return NULL;
    }
    int row = (x->mat[0]->row - filter->w[0]->mat[0]->row + 2* padding[0]) / stride[0] + 1, 
        col = (x->mat[0]->col - filter->w[0]->mat[0]->col + 2* padding[1]) / stride[1] + 1;
    Tensor* z = new_tensor(row, col, filter->channel);
    for (int i = 0, j; i < z->depth; i++) {
        for (j = 0; j < x->depth; j++) convolutional_func(z->mat[i], x->mat[j], filter->w[i]->mat[j], stride, padding);
        plus_bias_2D(z->mat[i], filter->b[i]);
    }
    return z;
}

static void conv_grad_point(Matrix* dL_dw, Matrix* x, float dL_dz, int row_point, int col_point) {
    for (int i = 0, j, row, col; i < dL_dw->row; i++) 
        for (j = 0; j < dL_dw->col; j++) {
            row = row_point + i, col = col_point + j;
            if (is_valid_point(x, row, col)) dL_dw->val[i][j] += dL_dz* x->val[row][col];
        }
}
static void conv_grad_func(Matrix* dL_dw, float* dL_db, Matrix* x, Matrix* dL_dz, int* stride, int* padding) {
    int start_row = -padding[0], start_col = -padding[1];
    for (int i = 0, ci = 0, j, cj; i < dL_dz->row; i++, ci += stride[0]) {
        for (j = 0, cj = 0; j < dL_dz->col; j++, cj += stride[1]) {
            conv_grad_point(dL_dw, x, dL_dz->val[i][j], start_row + ci, start_col + cj);
            *dL_db += dL_dz->val[i][j];
        }
    }
}
void kernel_derivative(Kernel* deriv, Tensor** front_a, int batch, Tensor** delta, int* stride, int* padding) {  // = 1/n. a(l-1)*&z(l)
    int i, j, k;
    for (j = 0; j < deriv->channel; j++) {
        deriv->b[j] = 0;
        for (k = 0; k < front_a[0]->depth; k++) {
            fill_matrix(deriv->w[j]->mat[k], 0);
            for (i = 0; i < batch; i++) 
                conv_grad_func(deriv->w[j]->mat[k], &(deriv->b[j]), front_a[i]->mat[k], delta[i]->mat[j], stride, padding);
            scalar_multiply(deriv->w[j]->mat[k], 1.0f / batch);
        }
        deriv->b[j] /= (batch* front_a[0]->depth);
    }
}
void binary_write_kernels(FILE* f, Kernel* filter) {
    if (!f || !filter) return ;
    int depth = filter->w[0]->depth, row = filter->w[0]->mat[0]->row, col = filter->w[0]->mat[0]->col, chk, i, j, k;
    for (i = 0; i < filter->channel; i++) 
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                if (fwrite(filter->w[i]->mat[j]->val[k], sizeof(float), col, f) != col) 
                    printf("Warning: Writing filter to file failed !!");
    if (fwrite(filter->b, sizeof(float), filter->channel, f) != filter->channel) 
        printf("Warning: Writing bias to file failed !!");
}
void binary_read_kernels(FILE* f, Kernel* filter) {
    if (!f || !filter) return ;
    int depth = filter->w[0]->depth, row = filter->w[0]->mat[0]->row, col = filter->w[0]->mat[0]->col, chk, i, j, k;
    for (i = 0; i < filter->channel; i++) 
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                if (fread(filter->w[i]->mat[j]->val[k], sizeof(float), col, f) != col) 
                    printf("Warning: Reading filter from file failed !!");
    if (fread(filter->b, sizeof(float), filter->channel, f) != filter->channel) 
        printf("Warning: Reading bias from file failed !!");
}