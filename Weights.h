#pragma once
#include "Keras_core.h"

Weights* init_weights(int row, int col, int random_state) {
    Weights* neww = (Weights*)malloc(sizeof(Weights));
    int i, j;
    neww->w = new_matrix(row, col);
    neww->b = (float*)calloc(col, sizeof(float));
    if (random_state) {
        srand(random_state);
        for (i = 0; i < col; i++) {
            for (j = 0; j < row; j++) neww->w->val[j][i] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
            neww->b[i] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
        }
    }
    return neww;
}
void copy_weights(Weights* dest, Weights* sauce) {
    if (!dest) {
        printf("Error: No available weights space for copying !!");
        return ;
    }
    if (dest->w->row < sauce->w->row || dest->w->col < sauce->w->col) {
        printf("Warning: Inappropriate size of weights !!");
        return ;
    }
    for (int i = 0, j; i < sauce->w->col; i++) {
        for (j = 0; j < sauce->w->row; j++) dest->w->val[j][i] = sauce->w->val[j][i];
        dest->b[i] = sauce->b[i];
    }
}
Weights* get_copy_weights(Weights* sauce) {
    Weights* neww = init_weights(sauce->w->row, sauce->w->col, 0);
    copy_weights(neww, sauce);
    return neww;
}
void fill_weights(Weights* w, float fill_value) {
    for (int i = 0, j; i < w->w->col; i++) {
        for (j = 0; j < w->w->row; j++) w->w->val[j][i] = fill_value;
        w->b[i] = fill_value;
    }
}
void free_weights(Weights* w) {
    if (!w) return ;
    if (w->w) free_matrix(w->w);
    if (w->b) free(w->b);
    free(w);
}
void plus_bias(Matrix* xw, float* bias) {
    for (int i = 0, j; i < xw->row; i++) {
        for (j = 0; j < xw->col; j++) xw->val[i][j] += bias[j];
    }
}
void weights_derivative(Weights* deriv, Matrix* front_a, Matrix* delta) {  // = 1/n. a(l-1)(T).&z(l)
    Matrix* a_T = transpose(front_a);
    get_multiply(deriv->w, a_T, delta);
    scalar_multiply(deriv->w, 1.0f / delta->row);
    free_matrix(a_T);
    for (int i = 0, j; i < delta->col; i++) {
        deriv->b[i] = 0;
        for (j = 0; j < delta->row; j++) deriv->b[i] += delta->val[j][i];
        deriv->b[i] /= delta->row;
    }
}
void binary_write_weights(FILE* f, Weights* w) {
    if (!f || !w) return ;
    int i, row = w->w->row, col = w->w->col;
    for (i = 0; i < row; i++) 
        if (fwrite(w->w->val[i], sizeof(float), col, f) != col) 
            printf("Warning: Writing weights to file failed !!");
    if (fwrite(w->b, sizeof(float), col, f) != col) 
        printf("Warning: Writing bias to file failed !!");
}
void binary_read_weights(FILE* f, Weights* w) {
    if (!f || !w) return ;
    int i, row = w->w->row, col = w->w->col;
    for (i = 0; i < row; i++) 
        if (fread(w->w->val[i], sizeof(float), col, f) != col) 
            printf("Warning: Reading weights from file failed !!");
    if (fread(w->b, sizeof(float), col, f) != col) 
        printf("Warning: Reading bias from file failed !!");
}
