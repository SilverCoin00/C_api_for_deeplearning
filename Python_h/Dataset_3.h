#pragma once
#include "Python.h"

typedef struct Dataset_3 {
    Tensor** x;
    Matrix* y;
    int samples;
    int channel;
} Dataset_3;

Dataset_3* trans_dframe_to_dset3(Data_Frame* df, int row_size, int col_size, int channel) {
    if (row_size* col_size* channel != df->col - 1) {
        printf("Warning: Mismatch between image and DataFrame sizes !!");
        return NULL;
    }
    Dataset_3* newd = (Dataset_3*)malloc(sizeof(Dataset_3));
    newd->samples = df->row;
    newd->x = (Tensor**)malloc(df->row* sizeof(Tensor*));
    for (int i = 0, j, k, h; i < df->row; i++) {
        newd->x[i] = new_tensor(row_size, col_size, channel);
        for (j = 0; j < channel; j++) 
            for (k = 0; k < row_size; k++) 
                for (h = 0; h < col_size; h++) newd->x[i]->mat[j]->val[k][h] = df->data[i][j + channel* (k* col_size + h)];
    }
    newd->y = (Matrix*)malloc(sizeof(Matrix));
    newd->y->row = df->row;
    newd->y->val = label_to_one_hot_encode(df->data, df->col - 1, NULL, df->row, &(newd->y->col));
    newd->channel = channel;
    return newd;
}
void dataset3_sample_copy(Dataset_3* ds, int ds_sample_index, Dataset_3* copy, int copy_sample_index) {
    int i;
    copy_tensor(copy->x[copy_sample_index], ds->x[ds_sample_index]);
    for (i = 0; i < ds->y->col && i < copy->y->col; i++)
		copy->y->val[copy_sample_index][i] = ds->y->val[ds_sample_index][i];
}
Dataset_3* dataset3_samples_order_copy(Dataset_3* ds, int* order, int order_begin_index, int order_end_index) {
	Dataset_3* newd = (Dataset_3*)malloc(sizeof(Dataset_3));
    int i, j = order_end_index - order_begin_index;
    newd->x = (Tensor**)malloc(j* sizeof(Tensor*));
    newd->samples = j;
    newd->channel = ds->x[0]->depth;
    for (i = 0; i < j; i++) 
        newd->x[i] = new_tensor(ds->x[i]->mat[0]->row, ds->x[i]->mat[0]->col, ds->x[i]->depth);
	newd->y = new_matrix(j, ds->y->col);
	for (i = order_begin_index; i < order_end_index; i++) {
		dataset3_sample_copy(ds, order[i], newd, i - order_begin_index);
	}
	return newd;
}
void free_dataset3(Dataset_3* ds) {
    if (!ds) return ;
    for (int i = 0; i < ds->samples; i++) free_tensor(ds->x[i]);
    free_matrix(ds->y);
    free(ds);
}
void print_dataset3(Dataset_3* ds, int decimal, int col_space, int num_of_rows) {
    if (!ds) return ;
    if (num_of_rows < 0 || num_of_rows > ds->samples) num_of_rows = ds->samples;
    printf(" Row\n");
    for (int i = 0, j, k, h, row = ds->x[0]->mat[0]->row, col = ds->x[0]->mat[0]->col; i < num_of_rows; i++) {
        printf("%4d", i + 1);
        for (j = 0; j < ds->channel; j++) {
            printf("\t[ ");
            for (h = 0; h < col; h++) printf("%*.*f ", col_space, decimal, ds->x[i]->mat[j]->val[0][h]);
            for (k = 1; k < row; k++) {
                printf("\n\t  ");
                for (h = 0; h < col; h++) printf("%*.*f ", col_space, decimal, ds->x[i]->mat[j]->val[k][h]);
            }
            printf("]");
            if (j < ds->channel - 1) printf("\n");
        }
        printf("\t|   [ ");
        for (j = 0; j < ds->y->col; j++) printf("%*.*f ", col_space, decimal, ds->y->val[i][j]);
        printf("]\n");
    }
}
void train_test_split_ds3(Dataset_3* data, Dataset_3* train, Dataset_3* test, float test_size, int random_state) {
	int i, j;
    i = round(test_size* data->samples);
    test->samples = i;
    test->channel = data->channel;
    test->x = (Tensor**)malloc(i* sizeof(Tensor*));
    for (j = 0; j < i; j++) test->x[j] = new_tensor(data->x[j]->mat[0]->row, data->x[j]->mat[0]->col, data->x[j]->depth);
    test->y = new_matrix(i, data->y->col);
	i = data->samples - i;
    train->samples = i;
    train->channel = data->channel;
    train->x = (Tensor**)malloc(i* sizeof(Tensor*));
    for (j = 0; j < i; j++) train->x[j] = new_tensor(data->x[j]->mat[0]->row, data->x[j]->mat[0]->col, data->x[j]->depth);
    train->y = new_matrix(i, data->y->col);

    int* random_i = (int*)malloc(data->samples* sizeof(int));
    for (i = 0; i < data->samples; i++) random_i[i] = i;
    shuffle_index(random_i, data->samples, random_state);
    
    for (i = 0; i < test->samples; i++) 
        dataset3_sample_copy(data, random_i[i], test, i);
    for (j = 0; i < data->samples; i++, j++) 
        dataset3_sample_copy(data, random_i[i], train, j);
    free(random_i);
}
