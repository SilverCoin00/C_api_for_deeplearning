#pragma once
#include "Keras_core.h"

void free_flatten_layer(Flatten* layer) {
    free_matrix(layer->flatted);
    free(layer);
}
Flatten* new_flatten_layer(int* flatten_size) {
    Flatten* layer = (Flatten*)calloc(1, sizeof(Flatten));
    flatten_size[0] *= flatten_size[1];
    flatten_size[0] *= flatten_size[2];
    return layer;
}
void flatten_forward(Flatten* layer, Tensor** x, int batch, Matrix** y) {
    free_matrix(layer->flatted);
    layer->row = x[0]->mat[0]->row, layer->col = x[0]->mat[0]->col;
    layer->depth = x[0]->depth;
    layer->flatted = new_matrix(batch, layer->depth* layer->row* layer->col);
    for (int i = 0, j, k, h; i < batch; i++) {
        for (j = 0; j < layer->depth; j++) 
            for (k = 0; k < layer->row; k++) 
                for (h = 0; h < layer->col; h++) 
                    layer->flatted->val[i][j* layer->row* layer->col + k* layer->col + h] = x[i]->mat[j]->val[k][h];
    }
    *y = layer->flatted;
}
void flatten_backprop(Flatten* layer, Matrix** back_delta, Tensor*** return_delta) {
    *return_delta = (Tensor**)malloc(layer->flatted->row* sizeof(Tensor*));
    for (int i = 0, j, k, h; i < layer->flatted->row; i++) {
        (*return_delta)[i] = new_tensor(layer->row, layer->col, layer->depth);
        for (j = 0; j < layer->depth; j++) 
            for (k = 0; k < layer->row; k++) 
                for (h = 0; h < layer->col; h++) 
                    (*return_delta)[i]->mat[j]->val[k][h] = (*back_delta)->val[i][j* layer->row* layer->col + k* layer->col + h];
    }
    free_matrix(*back_delta);
    *back_delta = NULL;
}
