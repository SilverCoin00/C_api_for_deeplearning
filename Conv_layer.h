#pragma once
#include "Keras_core.h"

static void free_conv_activated_data(Conv* layer, int batch) {
    int i;
    if (layer->a) {
        for (i = 0; i < batch; i++) if (layer->a[i]) free_tensor(layer->a[i]);
        free(layer->a);
    }
    if (layer->z) {
        for (i = 0; i < batch; i++) if (layer->z[i]) free_tensor(layer->z[i]);
        free(layer->z);
    }
}
void free_conv_layer(Conv* layer, int batch) {
    free_kernels(layer->filter);
    free_kernels(layer->deriv);
    free_kernels(layer->pre_velo);
    free_conv_activated_data(layer, batch);
    if (layer->stride) free(layer->stride);
    if (layer->padding)free(layer->padding);
    if (layer->drop) free(layer->drop);
    free(layer);
}
Conv* new_conv_layer(IConv* input, int* input_sizes, FNode* drop) {
    Conv* layer = (Conv*)calloc(1, sizeof(Conv));
    layer->activation = activation_encode(input->activation);
    layer->out_channels = input->channel;
    layer->filter = init_kernels(input->row, input->col, input->channel, input_sizes[2], input->row* input->col* input->channel);
    layer->stride = (int*)malloc(2* sizeof(int));
    layer->stride[0] = input->stride_row, layer->stride[1] = input->stride_col;
    layer->padding = (int*)malloc(2* sizeof(int));
    layer->padding[0] = input->padding_row, layer->padding[1] = input->padding_col;
    layer->drop = (float*)malloc(((int) drop->data + 1)* sizeof(float));
    for (int i = 0; drop; i++, drop = drop->next) layer->drop[i] = drop->data;
    
    input_sizes[0] = (input_sizes[0] - input->row + 2* layer->padding[0]) / layer->stride[0] + 1;
    input_sizes[1] = (input_sizes[1] - input->col + 2* layer->padding[1]) / layer->stride[1] + 1;
    input_sizes[2] = input->channel;
    return layer;
}
void drop_conv_neural(Tensor** a, int batch, int* drop_i, int* cur_num_units, float drop_ratio) {
    if (!(*cur_num_units) || !drop_ratio) return ;
    int num_drop = round(drop_ratio* *cur_num_units);
    shuffle_index(drop_i, *cur_num_units, time(NULL));
    int i, j;
    for (i = 0; i < batch; i++) {
        for (j = 1; j <= num_drop; j++) fill_matrix(a[i]->mat[drop_i[*cur_num_units - j]], 0);
        for ( ; j <= *cur_num_units; j++) scalar_multiply(a[i]->mat[drop_i[*cur_num_units - j]], 1.0f / (1 - drop_ratio));
    }
    *cur_num_units -= num_drop;
}
void conv_forward(Conv* layer, Tensor** x, int batch, int is_training, Tensor*** y) {
    int i;
    free_conv_activated_data(layer, batch);
    layer->z = (Tensor**)malloc(batch* sizeof(Tensor*));
    for (i = 0; i < batch; i++) layer->z[i] = kernel_func(layer->filter, x[i], layer->stride, layer->padding);
    layer->a = (Tensor**)malloc(batch* sizeof(Tensor*));
    for (i = 0; i < batch; i++) layer->a[i] = activation_func_2D(layer->z[i], layer->activation);
    if (is_training) {
        int nodes = layer->a[0]->depth;
        int* drop_i = (int*)malloc(nodes* sizeof(int));
        for (i = 0; i < nodes; i++) drop_i[i] = i;
        for (i = 1; i < layer->drop[0]; i++) drop_conv_neural(layer->a, batch, drop_i, &nodes, layer->drop[i]);
        free(drop_i);
    }
    *y = layer->a;
}
static Tensor** cal_conv_delta_a(Tensor** dL_dz, Kernel* filter, int batch, int* stride, int* padding) {
    /* &a(l-1) = (&a(l).f'(z(l)))* rot(180)(w(l))
       pad(b) - pad(f) = (in_size - out_size).(stride + 1) / 2   */
    int row = (dL_dz[0]->mat[0]->row - 1)* stride[0] + filter->w[0]->mat[0]->row - 2* padding[0], 
        col = (dL_dz[0]->mat[0]->col - 1)* stride[1] + filter->w[0]->mat[0]->col - 2* padding[1];
    int* padding_b = (int*)malloc(2* sizeof(int));
    padding_b[0] = padding[0] + (row - dL_dz[0]->mat[0]->row)* (stride[0] + 1) / 2;
    padding_b[1] = padding[1] + (col - dL_dz[0]->mat[0]->col)* (stride[1] + 1) / 2;
    Tensor** dL_da = (Tensor**)malloc(batch* sizeof(Tensor*));
    Kernel* rot_filter = rot_kernels(filter);
    for (int i = 0; i < batch; i++) dL_da[i] = kernel_func(rot_filter, dL_dz[i], stride, padding_b);
    free_kernels(rot_filter);
    free(padding_b);
    return dL_da;
}
void conv_backprop(Tensor** front_a, Conv* layer, int batch, int is_last, Tensor*** dL_da, Model_Compiler* cpl) {
    int i, j;
    if (!layer->deriv) layer->deriv = init_kernels(layer->filter->w[0]->mat[0]->row, layer->filter->w[0]->mat[0]->col, 
                                                    layer->filter->channel, layer->filter->w[0]->depth, 0);
    Tensor** a_deriv = (Tensor**)malloc(batch* sizeof(Tensor*));
    Tensor** dL_dz = (Tensor**)malloc(batch* sizeof(Tensor*));
    for (i = 0; i < batch; i++) a_deriv[i] = activation_derivative_2D(layer->a[i], layer->z[i], layer->activation);
    Kernel* temp = get_copy_kernels(layer->filter);
    check_nestorov_2D(temp, layer->pre_velo, cpl->optimize);

    if (!is_last) for (i = 0; i < batch; i++) dL_dz[i] = tensor_ewise_multiply((*dL_da)[i], a_deriv[i]);
    else {
        for (i = 0; i < batch; i++) {
            dL_dz[i] = minust(layer->a[i], (*dL_da)[i]);
            tensor_scalar_multiply(dL_dz[i], 1.0f / batch);
            if (cpl->loss_type == 1 && layer->activation != 0) 
                get_tensor_ewise_multiply(dL_dz[i], a_deriv[i]);
        }
    }

    for (i = 0; i < batch; i++) {
        free_tensor((*dL_da)[i]);
        free_tensor(a_deriv[i]);
    }
    free(a_deriv);
    free(*dL_da);
    kernel_derivative(layer->deriv, front_a, batch, dL_dz, layer->stride, layer->padding);

    *dL_da = cal_conv_delta_a(dL_dz, layer->filter, batch, layer->stride, layer->padding);
    for (i = 0; i < batch; i++) free_tensor(dL_dz[i]);
    free(dL_dz);
    free_kernels(temp);
}
void binary_write_conv(FILE* f, Conv* layer) {
    fwrite(&(layer->activation), sizeof(int), 1, f);
    fwrite(&(layer->out_channels), sizeof(int), 1, f);
    fwrite(&(layer->filter->w[0]->depth), sizeof(int), 1, f);
    fwrite(&(layer->filter->w[0]->mat[0]->row), sizeof(int), 1, f);
    fwrite(&(layer->filter->w[0]->mat[0]->col), sizeof(int), 1, f);
    binary_write_kernels(f, layer->filter);
    fwrite(layer->stride, sizeof(int), 2, f);
    fwrite(layer->padding, sizeof(int), 2, f);
    fwrite(layer->drop, sizeof(float), (int) layer->drop[0], f);
}
void binary_read_conv(FILE* f, Conv* layer) {
    fread(&(layer->activation), sizeof(int), 1, f);
    fread(&(layer->out_channels), sizeof(int), 1, f);
    int depth, row, col;
    fread(&depth, sizeof(int), 1, f);
    fread(&row, sizeof(int), 1, f);
    fread(&col, sizeof(int), 1, f);
    layer->filter = init_kernels(row, col, layer->out_channels, depth, row + col + depth);
    binary_read_kernels(f, layer->filter);
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
