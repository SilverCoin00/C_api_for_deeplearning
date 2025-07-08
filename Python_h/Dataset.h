#pragma once
#include "D:\Std_gcc_&_g++_plus\Python.h"

typedef struct {
    int type;
    void* data;
} Dataset;

Dataset* trans_dframe_to_dset(int type, Data_Frame* df, int* sizes, const char* output_col) {
    Dataset* ds = (Dataset*)malloc(sizeof(Dataset));
    ds->type = type;
    if (type == 2) ds->data = (Dataset_2*)trans_dframe_to_dset2(df, output_col);
    else if (type == 3) ds->data = (Dataset_3*)trans_dframe_to_dset3(df, sizes[0], sizes[1], sizes[2]);
    return ds;
}
void train_test_split(Dataset* data, Dataset* train, Dataset* test, float test_size, int random_state) {
    train->type = test->type = data->type;
    if (data->type == 2) {
        train->data = (Dataset_2*)malloc(sizeof(Dataset_2));
        test->data = (Dataset_2*)malloc(sizeof(Dataset_2));
        train_test_split_ds2((Dataset_2*) data->data, (Dataset_2*) train->data, (Dataset_2*) test->data, test_size, random_state);
    } else if (data->type == 3) {
        train->data = (Dataset_3*)malloc(sizeof(Dataset_3));
        test->data = (Dataset_3*)malloc(sizeof(Dataset_3));
        train_test_split_ds3((Dataset_3*) data->data, (Dataset_3*) train->data, (Dataset_3*) test->data, test_size, random_state);
    }
}
int get_ds_num_samples(Dataset* ds) {
    if (ds->type == 2) return ((Dataset_2*) ds->data)->x->row;
    if (ds->type == 3) return ((Dataset_3*) ds->data)->samples;
}
void* get_ds_x(Dataset* ds) {
    if (ds->type == 2) return ((Dataset_2*) ds->data)->x;
    if (ds->type == 3) return ((Dataset_3*) ds->data)->x;
}
void* get_ds_y(Dataset* ds) {
    if (ds->type == 2) return ((Dataset_2*) ds->data)->y;
    if (ds->type == 3) return ((Dataset_3*) ds->data)->y;
}
Dataset* dataset_samples_order_copy(const Dataset* ds, int* order, int order_begin_index, int order_end_index) {
    Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
    newd->type = ds->type;
    if (ds->type == 2) newd->data = dataset2_samples_order_copy((Dataset_2*) ds->data, order, order_begin_index, order_end_index);
    else if (ds->type == 3) newd->data = dataset3_samples_order_copy((Dataset_3*) ds->data, order, order_begin_index, order_end_index);
    return newd;
}
void free_dataset(Dataset* ds) {
    if (!ds) return ;
    if (ds->type == 2) free_dataset2((Dataset_2*) ds->data);
    else if (ds->type == 3) free_dataset3((Dataset_3*) ds->data);
    free(ds);
}
void print_dataset(Dataset* ds, int decimal, int col_space, int num_of_rows) {
    if (!ds) return ;
    if (ds->type == 2) print_dataset2((Dataset_2*) ds->data, decimal, col_space, num_of_rows);
    else if (ds->type == 3) print_dataset3((Dataset_3*) ds->data, decimal, col_space, num_of_rows);
}