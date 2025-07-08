#pragma once
#include "Keras_core.h"

void free_max_pooling_data(MaxPooling* layer, int batch) {
    int i;
    if (layer->pool) {
        for (i = 0; i < batch; i++) if (layer->pool[i]) free_tensor(layer->pool[i]);
        free(layer->pool);
        layer->pool = NULL;
    }
    if (layer->mask_row) {
        for (i = 0; i < batch; i++) if (layer->mask_row[i]) free_tensor(layer->mask_row[i]);
        free(layer->mask_row);
        layer->mask_row = NULL;
    }
    if (layer->mask_col) {
        for (i = 0; i < batch; i++) if (layer->mask_col[i]) free_tensor(layer->mask_col[i]);
        free(layer->mask_col);
        layer->mask_col = NULL;
    }
}
void free_max_pooling_layer(MaxPooling* layer, int batch) {
    free_max_pooling_data(layer, batch);
    if (layer->drop) free(layer->drop);
    if (layer->stride) free(layer->stride);
    if (layer->padding)free(layer->padding);
    free(layer);
}
MaxPooling* new_max_pooling_layer(IMPool* input, int* pooled_size, FNode* drop) {
    MaxPooling* newmp = (MaxPooling*)calloc(1, sizeof(MaxPooling));
    newmp->row = input->row, newmp->col = input->col;
    newmp->stride = (int*)malloc(2* sizeof(int));
    newmp->stride[0] = input->stride_row, newmp->stride[1] = input->stride_col;
    newmp->padding = (int*)malloc(2* sizeof(int));
    newmp->padding[0] = input->padding_row, newmp->padding[1] = input->padding_col;
    newmp->drop = (float*)malloc(((int) drop->data + 1)* sizeof(float));
    for (int i = 0; drop; i++, drop = drop->next) newmp->drop[i] = drop->data;

    pooled_size[0] = (pooled_size[0] - input->row + 2* newmp->padding[0]) / newmp->stride[0] + 1;
    pooled_size[1] = (pooled_size[1] - input->col + 2* newmp->padding[1]) / newmp->stride[1] + 1;
    return newmp;
}
static float max_pool_point(Matrix* x, int pool_h, int pool_w, int row_point, int col_point, float* pos_i, float* pos_j) {
    float max = -1.0f / 0.0f;
    for (int i = 0, j, row, col; i < pool_h; i++) 
        for (j = 0; j < pool_w; j++) {
            row = row_point + i, col = col_point + j;
            if (is_valid_point(x, row, col)) 
                if (max < x->val[row][col]) {
                    max = x->val[row][col];
                    *pos_i = row, *pos_j = col;
                }
        }
    return max;
}
static void max_pool_func(Matrix* x, Matrix* pool, int pool_h, int pool_w, 
                            Matrix* mask_row, Matrix* mask_col, int* stride, int* padding) {
    int start_row = -padding[0], start_col = -padding[1];
    for (int i = 0, ci = 0, j, cj; i < pool->row; i++, ci += stride[0]) {
        for (j = 0, cj = 0; j < pool->col; j++, cj += stride[1]) 
            pool->val[i][j] = max_pool_point(x, pool_h, pool_w, start_row + ci, start_col + cj, 
                                            &(mask_row->val[i][j]), &(mask_col->val[i][j]));
    }
}
void max_pool_forward(Tensor** x, MaxPooling* layer, int batch, int is_training, Tensor*** y) {
    free_max_pooling_data(layer, batch);
    layer->pool = (Tensor**)malloc(batch* sizeof(Tensor*));
    layer->mask_row = (Tensor**)malloc(batch* sizeof(Tensor*));
    layer->mask_col = (Tensor**)malloc(batch* sizeof(Tensor*));
    int i, j;
    int row = (x[0]->mat[0]->row - layer->row + 2* layer->padding[0]) / layer->stride[0] + 1, 
        col = (x[0]->mat[0]->col - layer->col + 2* layer->padding[1]) / layer->stride[1] + 1;
    for (i = 0; i < batch; i++) {
        layer->pool[i] = new_tensor(row, col, x[0]->depth);
        layer->mask_row[i] = new_tensor(row, col, x[0]->depth);
        layer->mask_col[i] = new_tensor(row, col, x[0]->depth);
        for (j = 0; j < x[i]->depth; j++) 
            max_pool_func(x[i]->mat[j], layer->pool[i]->mat[j], layer->row, layer->col,
                layer->mask_row[i]->mat[j], layer->mask_col[i]->mat[j], layer->stride, layer->padding);
    }
    if (is_training) {
        int nodes = layer->pool[0]->depth;
        int* drop_i = (int*)malloc(nodes* sizeof(int));
        for (i = 0; i < nodes; i++) drop_i[i] = i;
        for (i = 1; i < layer->drop[0]; i++) drop_conv_neural(layer->pool, batch, drop_i, &nodes, layer->drop[i]);
        free(drop_i);
    }
    *y = layer->pool;
}
void max_pool_backprop(Tensor** front_a, MaxPooling* layer, int batch, Tensor*** dL_da) {
    Tensor** back_delta = (Tensor**)malloc(batch* sizeof(Tensor*));
    int i, j, k, h, max_r, max_c;
    for (i = 0; i < batch; i++) {
        back_delta[i] = new_tensor(front_a[i]->mat[0]->row, front_a[i]->mat[0]->col, front_a[i]->depth);
        for (j = 0; j < layer->pool[i]->depth; j++) 
            for (k = 0; k < layer->pool[i]->mat[j]->row; k++) 
                for (h = 0; h < layer->pool[i]->mat[j]->col; h++) {
                    max_r = (int) layer->mask_row[i]->mat[j]->val[k][h];
                    max_c = (int) layer->mask_col[i]->mat[j]->val[k][h];
                    back_delta[i]->mat[j]->val[max_r][max_c] += (*dL_da)[i]->mat[j]->val[k][h];
                }
        free_tensor((*dL_da)[i]);
    }
    free(*dL_da);
    *dL_da = back_delta;
}
void binary_write_max_pool(FILE* f, MaxPooling* layer) {
    fwrite(&(layer->row), sizeof(int), 1, f);
    fwrite(&(layer->col), sizeof(int), 1, f);
    fwrite(layer->stride, sizeof(int), 2, f);
    fwrite(layer->padding, sizeof(int), 2, f);
    fwrite(layer->drop, sizeof(float), (int) layer->drop[0] + 1, f);
}
void binary_read_max_pool(FILE* f, MaxPooling* layer) {
    fread(&(layer->row), sizeof(int), 1, f);
    fread(&(layer->col), sizeof(int), 1, f);
    layer->stride = (int*)malloc(2* sizeof(int));
    layer->padding = (int*)malloc(2* sizeof(int));
    fread(layer->stride, sizeof(int), 2, f);
    fread(layer->padding, sizeof(int), 2, f);
    float drop;
    fread(&drop, sizeof(float), 1, f);
    layer->drop = (float*)malloc(((int) drop + 1)* sizeof(float));
    layer->drop[0] = drop;
    fread(layer->drop + 1, sizeof(float), (int) drop, f);
}

void free_average_pooling_data(AveragePooling* layer, int batch) {
    int i;
    if (layer->pool) {
        for (i = 0; i < batch; i++) if (layer->pool[i]) free_tensor(layer->pool[i]);
        free(layer->pool);
        layer->pool = NULL;
    }
}
void free_average_pooling_layer(AveragePooling* layer, int batch) {
    free_average_pooling_data(layer, batch);
    if (layer->drop) free(layer->drop);
    if (layer->stride) free(layer->stride);
    if (layer->padding)free(layer->padding);
    free(layer);
}
AveragePooling* new_average_pooling_layer(IAPool* input, int* pooled_size, FNode* drop) {
    AveragePooling* newap = (AveragePooling*)calloc(1, sizeof(AveragePooling));
    newap->row = input->row, newap->col = input->col;
    newap->stride = (int*)malloc(2* sizeof(int));
    newap->stride[0] = input->stride_row, newap->stride[1] = input->stride_col;
    newap->padding = (int*)malloc(2* sizeof(int));
    newap->padding[0] = input->padding_row, newap->padding[1] = input->padding_col;
    newap->drop = (float*)malloc(((int) drop->data + 1)* sizeof(float));
    for (int i = 0; drop; i++, drop = drop->next) newap->drop[i] = drop->data;

    pooled_size[0] = (pooled_size[0] - input->row + 2* newap->padding[0]) / newap->stride[0] + 1;
    pooled_size[1] = (pooled_size[1] - input->col + 2* newap->padding[1]) / newap->stride[1] + 1;
    return newap;
}
static float average_pool_point(Matrix* x, int pool_h, int pool_w, int row_point, int col_point) {
    if (!pool_h || !pool_w) return 0;
    float sum = 0;
    for (int i = 0, j, row, col; i < pool_h; i++) 
        for (j = 0; j < pool_w; j++) {
            row = row_point + i, col = col_point + j;
            if (is_valid_point(x, row, col)) sum += x->val[row][col];
        }
    return sum / pool_h / pool_w;
}
static void average_pool_func(Matrix* x, Matrix* pool, int pool_h, int pool_w, int* stride, int* padding) {
    int start_row = -padding[0], start_col = -padding[1];
    for (int i = 0, ci = 0, j, cj; i < pool->row; i++, ci += stride[0]) {
        for (j = 0, cj = 0; j < pool->col; j++, cj += stride[1]) 
            pool->val[i][j] = average_pool_point(x, pool_h, pool_w, start_row + ci, start_col + cj);
    }
}
void average_pool_forward(Tensor** x, AveragePooling* layer, int batch, int is_training, Tensor*** y) {
    free_average_pooling_data(layer, batch);
    layer->pool = (Tensor**)malloc(batch* sizeof(Tensor*));
    int i, j;
    int row = (x[0]->mat[0]->row - layer->row + 2* layer->padding[0]) / layer->stride[0] + 1, 
        col = (x[0]->mat[0]->col - layer->col + 2* layer->padding[1]) / layer->stride[1] + 1;
    for (i = 0; i < batch; i++) {
        layer->pool[i] = new_tensor(row, col, x[0]->depth);
        for (j = 0; j < x[i]->depth; j++) 
            average_pool_func(x[i]->mat[j], layer->pool[i]->mat[j], layer->row, layer->col, layer->stride, layer->padding);
    }
    if (is_training) {
        int nodes = layer->pool[0]->depth;
        int* drop_i = (int*)malloc(nodes* sizeof(int));
        for (i = 0; i < nodes; i++) drop_i[i] = i;
        for (i = 1; i < layer->drop[0]; i++) drop_conv_neural(layer->pool, batch, drop_i, &nodes, layer->drop[i]);
        free(drop_i);
    }
    *y = layer->pool;
}

static void average_repool_point(Matrix* x, float grad, int pool_h, int pool_w, int row_point, int col_point) {
    for (int i = 0, j, row, col; i < pool_h; i++) 
        for (j = 0; j < pool_w; j++) {
            row = row_point + i, col = col_point + j;
            if (is_valid_point(x, row, col)) x->val[row][col] += grad / pool_h / pool_w;
        }
}
static void average_repool_func(Matrix* x, Matrix* pool, int pool_h, int pool_w, int* stride, int* padding) {
    int start_row = -padding[0], start_col = -padding[1];
    for (int i = 0, ci = 0, j, cj; i < pool->row; i++, ci += stride[0]) {
        for (j = 0, cj = 0; j < pool->col; j++, cj += stride[1]) 
            average_repool_point(x, pool->val[i][j], pool_h, pool_w, start_row + ci, start_col + cj);
    }
}
void average_pool_backprop(Tensor** front_a, AveragePooling* layer, int batch, Tensor*** dL_da) {
    Tensor** back_delta = (Tensor**)malloc(batch* sizeof(Tensor*));
    int i, j;
    for (i = 0; i < batch; i++) {
        back_delta[i] = new_tensor(front_a[i]->mat[0]->row, front_a[i]->mat[0]->col, front_a[i]->depth);
        for (j = 0; j < layer->pool[i]->depth; j++) 
            average_repool_func(back_delta[i]->mat[j], (*dL_da)[i]->mat[j], layer->row, layer->col, layer->stride, layer->padding);
        free_tensor((*dL_da)[i]);
    }
    free(*dL_da);
    *dL_da = back_delta;
}
void binary_write_average_pool(FILE* f, AveragePooling* layer) {
    fwrite(&(layer->row), sizeof(int), 1, f);
    fwrite(&(layer->col), sizeof(int), 1, f);
    fwrite(layer->stride, sizeof(int), 2, f);
    fwrite(layer->padding, sizeof(int), 2, f);
    fwrite(layer->drop, sizeof(float), (int) layer->drop[0] + 1, f);
}
void binary_read_average_pool(FILE* f, AveragePooling* layer) {
    fread(&(layer->row), sizeof(int), 1, f);
    fread(&(layer->col), sizeof(int), 1, f);
    layer->stride = (int*)malloc(2* sizeof(int));
    layer->padding = (int*)malloc(2* sizeof(int));
    fread(layer->stride, sizeof(int), 2, f);
    fread(layer->padding, sizeof(int), 2, f);
    float drop;
    fread(&drop, sizeof(float), 1, f);
    layer->drop = (float*)malloc(((int) drop + 1)* sizeof(float));
    layer->drop[0] = drop;
    fread(layer->drop + 1, sizeof(float), (int) drop, f);
}
