#pragma once
#include "Python.h"

typedef struct Dataset_2 {
	Matrix* x;
	Matrix* y;
} Dataset_2;

Dataset_2* new_dataset2(float** x, float** y, int num_of_features, int num_of_samples, int num_of_output_classes) {
	Dataset_2* newd = (Dataset_2*)malloc(sizeof(Dataset_2));
	newd->x = new_matrix(num_of_samples, num_of_features);
	if (y) newd->y = new_matrix(num_of_samples, num_of_output_classes);
	int i, j;
	for (i = 0; i < num_of_samples; i++) {
		for (j = 0; j < num_of_features; j++) newd->x->val[i][j] = x[i][j];
		if (y) for (j = 0; j < num_of_output_classes; j++) newd->y->val[i][j] = y[i][j];
	}
	return newd;
}
Dataset_2* trans_dframe_to_dset2(Data_Frame* df, const char* predict_feature_col) {
    float** enc_sdata = NULL;
	int y_col = strtoi(predict_feature_col), i, j, k, is_y_str = 0;
	if (df->str_cols[0] != 0) {
		enc_sdata = (float**)malloc(df->str_cols[0]* sizeof(float*));
		Label_encoder* encoder = (Label_encoder*)malloc(sizeof(Label_encoder));
		for (i = 0; i < df->str_cols[0]; i++) {
			encoder_fit(df->str_data, df->row, i, encoder, "Label_encoder");
			enc_sdata[i] = (float*) encoder_transform(df->str_data, df->row, i, encoder, "Label_encoder");
			free_set(encoder->sample_types);
		}
		free(encoder);
	}
	if (y_col < 0) {
		j = df->col + df->str_cols[0];
		for (i = 0; i < j; i++) 
			if (!strcmp(df->features[i], predict_feature_col)) {
				for (k = 1; k <= df->str_cols[0]; k++)
					if (df->str_cols[k] == i) {
						y_col = k - 1;
						is_y_str = 1;
						goto out;
					}
				y_col = i;
				break;
			}
	}
	out:;
	Dataset_2* newd = (Dataset_2*)malloc(sizeof(Dataset_2));
	newd->y = (Matrix*)malloc(sizeof(Matrix));
	if (is_y_str) newd->y->val = label_to_one_hot_encode(NULL, 0, enc_sdata[y_col], df->row, &(newd->y->col));
	else {
		if (y_col >= 0) newd->y->val = label_to_one_hot_encode(df->data, y_col, NULL, df->row, &(newd->y->col));
	}
	newd->y->row = df->row;
	newd->x = new_matrix(df->row, df->col + df->str_cols[0] - 1);
	for (i = 0; i < df->row; i++) {
		for (j = 0, k = 0; k < df->col; k++) {
			if (!is_y_str) if (k == y_col) continue;
			newd->x->val[i][j++] = df->data[i][k];
		}
		for (k = 0; k < df->str_cols[0]; k++) {
			if (is_y_str) if (k == y_col) continue;
			newd->x->val[i][j++] = enc_sdata[k][i];
		}
	}
	if (enc_sdata) {
		for (i = 0; i < df->str_cols[0]; i++) free(enc_sdata[i]);
		free(enc_sdata);
	}
	return newd;
}
void dataset2_sample_copy(const Dataset_2* ds, int ds_sample_index, Dataset_2* copy, int copy_sample_index) {
	int i;
	for (i = 0; i < ds->x->col && i < copy->x->col; i++)
		copy->x->val[copy_sample_index][i] = ds->x->val[ds_sample_index][i];
	for (i = 0; i < ds->y->col && i < copy->y->col; i++)
		copy->y->val[copy_sample_index][i] = ds->y->val[ds_sample_index][i];
}
Dataset_2* dataset2_samples_order_copy(const Dataset_2* ds, int* order, int order_begin_index, int order_end_index) {
	Dataset_2* newd = (Dataset_2*)malloc(sizeof(Dataset_2));
	newd->x = new_matrix(order_end_index - order_begin_index, ds->x->col);
	newd->y = new_matrix(order_end_index - order_begin_index, ds->y->col);
	for (int i = order_begin_index; i < order_end_index; i++) {
		dataset2_sample_copy(ds, order[i], newd, i - order_begin_index);
	}
	return newd;
}
void print_dataset2(Dataset_2* ds, int decimal, int col_space, int num_of_rows) {
	if (!ds) return ;
	if (num_of_rows < 0 || num_of_rows > ds->x->row) num_of_rows = ds->x->row;
	printf(" Row\n");
	for (int i = 0, j; i < num_of_rows; i++) {
		printf("%4d\t", i + 1);
		for (j = 0; j < ds->x->col; j++) {
			printf("%*.*f ", col_space, decimal, ds->x->val[i][j]);
		}
        printf("\t|   [ ");
        for (j = 0; j < ds->y->col; j++) {
		    printf("%.1f ", ds->y->val[i][j]);
        }
        printf("]\n");
	}
}
void free_dataset2(Dataset_2* ds) {
	if (!ds) return ;
	free_matrix(ds->y);
	free_matrix(ds->x);
	free(ds);
}
void train_test_split_ds2(Dataset_2* data, Dataset_2* train, Dataset_2* test, float test_size, int random_state) {
	int i;
    i = round(test_size* data->x->row);
    test->x = new_matrix(i, data->x->col);
    test->y = new_matrix(i, data->y->col);
	i = data->x->row - i;
    train->x = new_matrix(i, data->x->col);
    train->y = new_matrix(i, data->y->col);

    int* random_i = (int*)malloc(data->x->row* sizeof(int));
    for (i = 0; i < data->x->row; i++) random_i[i] = i;
    shuffle_index(random_i, data->x->row, random_state);
    
    for (i = 0; i < test->x->row; i++) 
        dataset2_sample_copy(data, random_i[i], test, i);
    for (int e = 0; i < data->x->row; i++, e++) 
        dataset2_sample_copy(data, random_i[i], train, e);
    free(random_i);
}
