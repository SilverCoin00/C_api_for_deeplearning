#pragma once
#include "D:\Data\code_doc\Keras\Keras_core.h"

static void free_dense_activated_data(Dense* layer) {
    if (layer->a) free_matrix(layer->a);
    if (layer->z) free_matrix(layer->z);
}
void free_dense_layer(Dense* layer) {
    free_weights(layer->weights);
    free_weights(layer->deriv);
    free_dense_activated_data(layer);
    if (layer->drop) free(layer->drop);
    free_weights(layer->pre_velo);
    free_weights(layer->acc_grad);
    free(layer);
}
Dense* new_dense_layer(IDense* input, int* front_units, FNode* drop) {
    int i, j;
    Dense* layer = (Dense*)calloc(1, sizeof(Dense));
    layer->activation = activation_encode(input->activation);
    layer->weights = init_weights(front_units[0], input->nodes, front_units[0] + input->nodes + drop->data);
    layer->drop = (float*)malloc(((int) drop->data + 1)* sizeof(float));
    for (i = 0; drop; i++, drop = drop->next) layer->drop[i] = drop->data;
    front_units[0] = input->nodes;
    return layer;
}
static void drop_dense_neural(Matrix* a, int* drop_i, int* cur_num_units, float drop_ratio) {
    if (!(*cur_num_units) || !drop_ratio) return ;
    int num_drop = round(drop_ratio* *cur_num_units);
    shuffle_index(drop_i, *cur_num_units, time(NULL));
    int i, j;
    for (i = 1; i <= num_drop; i++) {
        for (j = 0; j < a->row; j++) a->val[j][drop_i[*cur_num_units - i]] = 0;
    }
    for ( ; i <= *cur_num_units; i++) {
        for (j = 0; j < a->row; j++) a->val[j][drop_i[*cur_num_units - i]] /= (1 - drop_ratio);
    }
    *cur_num_units -= num_drop;
}
void dense_forward(Dense* layer, Matrix* x, int is_training, Matrix** y) {
    free_dense_activated_data(layer);
    layer->z = multiply(x, layer->weights->w);
    plus_bias(layer->z, layer->weights->b);
    layer->a = activation_func(layer->z, layer->activation);
    if (is_training) {
        int i, col = layer->a->col;
        int* drop_i = (int*)malloc(col* sizeof(int));
        for (i = 0; i < col; i++) drop_i[i] = i;
        for (i = 1; i <= layer->drop[0]; i++) 
            drop_dense_neural(layer->a, drop_i, &col, layer->drop[i]);
        free(drop_i);
    }
    *y = layer->a;
}
static Matrix* cal_dense_delta_a(Matrix* dL_dz, Matrix* w) {  // &a(l-1) = (&a(l) @ f'(z(l))). w(l)(T) = dL|da
    Matrix* w_T = transpose(w);
    Matrix* dL_da = multiply(dL_dz, w_T);
    free_matrix(w_T);
    return dL_da;
}
void dense_backprop(Matrix* front_a, Dense* layer, int is_last, Matrix** dL_da, Model_Compiler* cpl) {
    if (!layer->deriv) 
        layer->deriv = init_weights(layer->weights->w->row, layer->weights->w->col, 0);
    Matrix* a_deriv = activation_derivative(layer->a, layer->z, layer->activation);
    Matrix* dL_dz;
    Weights* temp = get_copy_weights(layer->weights);
    check_nesterov(temp, layer->pre_velo, cpl->optimize);

    if (!is_last) dL_dz = ewise_multiply(*dL_da, a_deriv);
    else {
        dL_dz = minusm(layer->a, *dL_da);
        scalar_multiply(dL_dz, 1.0f / (*dL_da)->row);
        if (cpl->loss_type == 1 && layer->activation != 0) 
            get_ewise_multiply(dL_dz, a_deriv);
    }
    
    free_matrix(a_deriv);
    free_matrix(*dL_da);
    weights_derivative(layer->deriv, front_a, dL_dz);
    *dL_da = cal_dense_delta_a(dL_dz, temp->w);
    free_weights(temp);
}
void binary_write_dense(FILE* f, Dense* layer) {
    fwrite(&(layer->activation), sizeof(int), 1, f);
    fwrite(&(layer->weights->w->row), sizeof(int), 1, f);
    fwrite(&(layer->weights->w->col), sizeof(int), 1, f);
    binary_write_weights(f, layer->weights);
    fwrite(layer->drop, sizeof(float), (int) layer->drop[0], f);
}
void binary_read_dense(FILE* f, Dense* layer) {
    fread(&(layer->activation), sizeof(int), 1, f);
    int row, col;
    fread(&row, sizeof(int), 1, f);
    fread(&col, sizeof(int), 1, f);
    layer->weights = init_weights(row, col, 0);
    binary_read_weights(f, layer->weights);
    float drop;
    fread(&drop, sizeof(float), 1, f);
    layer->drop = (float*)malloc(((int) drop + 1)* sizeof(float));
    layer->drop[0] = drop;
    fread(layer->drop + 1, sizeof(float), (int) drop, f);
}