#pragma once
#include "Python.h"

typedef struct Standard_scaler {
    int features;
    float* mean;
    float* deviation;
} Standard_scaler;
typedef struct Min_max_scaler {
    int features;
    float* min;
    float* max;
} Min_max_scaler;
typedef struct One_hot_encoder {
    Set* sample_types;
} One_hot_encoder;
typedef struct Label_encoder {
    Set* sample_types;
} Label_encoder;
typedef struct Simple_Imputer {
    float* digit_data;
    char** str_data;
} Simple_Imputer;


void* new_scaler(char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
        return (void*) scaler;
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scaler = (Min_max_scaler*)malloc(sizeof(Min_max_scaler));
        return (void*) scaler;
    }
    return NULL;
}
void scaler_fit_1D(float* x, int samples, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        scl->features = 1;
        scl->mean = (float*)malloc(scl->features* sizeof(float));
        scl->deviation = (float*)malloc(scl->features* sizeof(float));
        float sum;
        int i, j;
        for (i = 0, sum = 0; i < samples; i++) sum += x[i];
        scl->mean[0] = sum / samples;
        for (i = 0, sum = 0; i < samples; i++) 
            sum += (scl->mean[0] - x[i])* (scl->mean[0] - x[i]);
        scl->deviation[0] = sqrt(sum / (samples - 1));
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        scl->features = 1;
        scl->min = (float*)malloc(scl->features* sizeof(float));
        scl->max = (float*)malloc(scl->features* sizeof(float));
        int i, j;
        scl->max[0] = scl->min[0] = x[0];
        for (i = 1; i < samples; i++) {
            if (scl->max[0] < x[i]) scl->max[0] = x[i];
            if (scl->min[0] > x[i]) scl->min[0] = x[i];
        }
    }
}
void scaler_fit_2D(Matrix* x, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        scl->features = x->col;
        scl->mean = (float*)malloc(scl->features* sizeof(float));
        scl->deviation = (float*)malloc(scl->features* sizeof(float));
        float sum;
        int i, j;
        for (i = 0; i < scl->features; i++) {
            sum = 0;
            for (j = 0; j < x->row; j++) sum += x->val[j][i];
            scl->mean[i] = sum / x->row;
            sum = 0;
            for (j = 0; j < x->row; j++) sum += (scl->mean[i] - x->val[j][i])* (scl->mean[i] - x->val[j][i]);
            scl->deviation[i] = sqrt(sum / (x->row - 1));
        }
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        scl->features = x->col;
        scl->min = (float*)malloc(scl->features* sizeof(float));
        scl->max = (float*)malloc(scl->features* sizeof(float));
        int i, j;
        for (i = 0; i < scl->features; i++) {
            scl->max[i] = scl->min[i] = x->val[0][i];
            for (j = 1; j < x->row; j++) {
                if (scl->max[i] < x->val[j][i]) scl->max[i] = x->val[j][i];
                if (scl->min[i] > x->val[j][i]) scl->min[i] = x->val[j][i];
            }
        }
    }
}
void scaler_fit_4D(Tensor** x, int samples, void* scaler, char* scaler_type) {
    int depth = x[0]->depth, row = x[0]->mat[0]->row, col = x[0]->mat[0]->col;
    int i, j, k, h;
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        scl->features = depth* row* col;
        scl->mean = (float*)calloc(scl->features, sizeof(float));
        scl->deviation = (float*)calloc(scl->features, sizeof(float));
        float dev;
        for (i = 0; i < samples; i++) {
            for (j = 0; j < depth; j++) 
                for (k = 0; k < row; k++) 
                    for (h = 0; h < col; h++) 
                        scl->mean[j* row* col + k* col + h] += x[i]->mat[j]->val[k][h];
        }
        for (i = 0; i < scl->features; i++) scl->mean[i] /= samples;
        for (i = 0; i < samples; i++) {
            for (j = 0; j < depth; j++) 
                for (k = 0; k < row; k++) 
                    for (h = 0; h < col; h++) {
                        dev = scl->mean[j* row* col + k* col + h] - x[i]->mat[j]->val[k][h];
                        scl->deviation[j* row* col + k* col + h] += dev* dev;
                    }
        }
        for (i = 0; i < scl->features; i++) 
            scl->deviation[i] = sqrt(scl->deviation[i] / (samples - 1));
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        scl->features = depth* row* col;
        scl->min = (float*)malloc(scl->features* sizeof(float));
        scl->max = (float*)malloc(scl->features* sizeof(float));
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) {
                    scl->max[j* row* col + k* col + h] = scl->min[j* row* col + k* col + h] = x[0]->mat[j]->val[k][h];
                    for (i = 1; i < samples; i++) {
                        if (scl->max[j* row* col + k* col + h] < x[i]->mat[j]->val[k][h]) scl->max[j* row* col + k* col + h] = x[i]->mat[j]->val[k][h];
                        if (scl->min[j* row* col + k* col + h] > x[i]->mat[j]->val[k][h]) scl->min[j* row* col + k* col + h] = x[i]->mat[j]->val[k][h];
                    }
                }
    }
}
void scaler_fit(int dim, void* x, int samples, void* scaler, char* scaler_type) {
    if (!x) return ;
    if (dim == 1) scaler_fit_1D((float*) x, samples, scaler, scaler_type);
    else if (dim == 2) scaler_fit_2D((Matrix*) x, scaler, scaler_type);
    else if (dim == 4) scaler_fit_4D((Tensor**) x, samples, scaler, scaler_type);
}
void scaler_transform_1D(float* x, int samples, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        for (int i = 0; i < samples; i++) 
            x[i] = (x[i] - scl->mean[0]) / scl->deviation[0];
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        for (int i = 0; i < samples; i++) 
            x[i] = (x[i] - scl->min[0]) / (scl->max[0] - scl->min[0]);
    }
}
void scaler_transform_2D(Matrix* x, void* scaler, char* scaler_type) {
    int i, j;
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        for (i = 0; i < x->col; i++) {
            for (j = 0; j < x->row; j++) x->val[j][i] = (x->val[j][i] - scl->mean[i]) / scl->deviation[i];
        }
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        for (i = 0; i < x->col; i++) {
            for (j = 0; j < x->row; j++) x->val[j][i] = (x->val[j][i] - scl->min[i]) / (scl->max[i] - scl->min[i]);
        }
    }
}
void scaler_transform_4D(Tensor** x, int samples, void* scaler, char* scaler_type) {
    int depth = x[0]->depth, row = x[0]->mat[0]->row, col = x[0]->mat[0]->col;
    int i, j, k, h;
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        for (i = 0; i < samples; i++) {
            for (j = 0; j < depth; j++) 
                for (k = 0; k < row; k++) 
                    for (h = 0; h < col; h++) 
                        x[i]->mat[j]->val[k][h] = (x[i]->mat[j]->val[k][h] - scl->mean[j* row* col + k* col + h]) / scl->deviation[j* row* col + k* col + h];
        }
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        for (i = 0; i < samples; i++) {
            for (j = 0; j < depth; j++) 
                for (k = 0; k < row; k++) 
                    for (h = 0; h < col; h++) 
                        x[i]->mat[j]->val[k][h] = (x[i]->mat[j]->val[k][h] - scl->min[j* row* col + k* col + h]) / (scl->max[j* row* col + k* col + h] - scl->min[j* row* col + k* col + h]);
        }
    }
}
void scaler_transform(int dim, void* x, int samples, void* scaler, char* scaler_type) {
    if (!x) return ;
    if (dim == 1) scaler_transform_1D((float*) x, samples, scaler, scaler_type);
    else if (dim == 2) scaler_transform_2D((Matrix*) x, scaler, scaler_type);
    else if (dim == 4) scaler_transform_4D((Tensor**) x, samples, scaler, scaler_type);
}
void free_scaler(void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        free(scl->mean);
        free(scl->deviation);
        free(scl);
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        free(scl->max);
        free(scl->min);
        free(scl);
    }
}
void scaler_save(const char* file_name, void* scaler, char* scaler_type) {
    FILE* file = fopen(file_name, "w");
    if (!file) {
        printf("Error: Cannot open file %s !!", file_name);
        return ;
    }
    int i;
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        fprintf(file, "Standard_scaler\n");
        fprintf(file, "%d\n", scl->features);
        for (i = 0; i < scl->features; i++) 
            fprintf(file, "%.20f %.20f\n", scl->mean[i], scl->deviation[i]);
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        fprintf(file, "Min_max_scaler\n");
        fprintf(file, "%d\n", scl->features);
        for (i = 0; i < scl->features; i++) 
            fprintf(file, "%.20f %.20f\n", scl->min[i], scl->max[i]);
    }
    fclose(file);
}
void* load_scaler(const char* file_name, char* scaler_type) {
    FILE* file = fopen(file_name, "r");
    if (!file) {
        printf("Error: Cannot open file %s !!", file_name);
        return NULL;
    }
    int i, j;
    char* s = (char*)malloc(200* sizeof(char));
    fgets(s, 200, file);
    s[strcspn(s, "\n")] = '\0';
    if (strcmp(scaler_type, s)) {
        printf("Warning: Inappropriated scaler type !!");
        return NULL;
    }
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
        fscanf(file, "%d", &(scaler->features));
        scaler->mean = (float*)malloc(scaler->features* sizeof(float));
        scaler->deviation = (float*)malloc(scaler->features* sizeof(float));
        for (i = 0; i < scaler->features; i++) {
            while ((j = fgetc(file)) != '\n' && j != EOF);
            fscanf(file, "%f %f", &(scaler->mean[i]), &(scaler->deviation[i]));
        }
        fclose(file);
        return (void*) scaler;
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scaler = (Min_max_scaler*)malloc(sizeof(Min_max_scaler));
        fscanf(file, "%d", &(scaler->features));
        scaler->min = (float*)malloc(scaler->features* sizeof(float));
        scaler->max = (float*)malloc(scaler->features* sizeof(float));
        for (i = 0; i < scaler->features; i++) {
            while ((j = fgetc(file)) != '\n' && j != EOF);
            fscanf(file, "%f %f", &(scaler->min[i]), &(scaler->max[i]));
        }
        fclose(file);
        return (void*) scaler;
    }
}
void shuffle_index(int* index, int size, int random_state) {
	srand((random_state >> 2) ^ (time(NULL) << 2) ^ (clock() >> 3));
    size *= 2; size /= 3;
	for (int j = 0, t, a, b; j < size; j++) {
		a = rand() % size, b = rand() % size;
		t = index[a];
		index[a] = index[b];
		index[b] = t;
	}
}
void encoder_fit(char*** data, int num_samples, int col, void* encoder, char* encoder_type) {
    if (!strcmp(encoder_type, "One_hot_encoder")) {
        One_hot_encoder* encode = (One_hot_encoder*) encoder;
        int i, j = 0;
        encode->sample_types = new_set(17);
        for (i = 0; i < num_samples; i++) 
            if (!set_find(encode->sample_types, data[i][col])) set_add(encode->sample_types, data[i][col], j++);;
    } else if (!strcmp(encoder_type, "Label_encoder")) {
        Label_encoder* encode = (Label_encoder*) encoder;
        int i, j = 0;
        encode->sample_types = new_set(17);
        for (i = 0; i < num_samples; i++) 
            if (!set_find(encode->sample_types, data[i][col])) set_add(encode->sample_types, data[i][col], j++);;
    }
}
void* encoder_transform(char*** data, int num_samples, int col, void* encoder, char* encoder_type) {
    if (!strcmp(encoder_type, "One_hot_encoder")) {
        One_hot_encoder* encode = (One_hot_encoder*) encoder;
        float** encd = (float**)malloc(num_samples* sizeof(float*));
        for (int i = 0, j; i < num_samples; i++) {
            j = set_call(encode->sample_types, data[i][col]);
            encd[i] = (float*)calloc(encode->sample_types->size, sizeof(float));
            if (j != -1) encd[i][j] = 1;
        }
        return (void*)encd;
    } else if (!strcmp(encoder_type, "Label_encoder")) {
        Label_encoder* encode = (Label_encoder*) encoder;
        float* encd = (float*)malloc(num_samples* sizeof(float));
        for (int i = 0; i < num_samples; i++)
            encd[i] = (float) set_call(encode->sample_types, data[i][col]);
        return (void*)encd;
    }
}
float** label_to_one_hot_encode(float** data, int col, float* line_data, int samples, int* class_num) {
    int i, j = 0;
    if (!line_data) {
        if (!data) return NULL;
        line_data = (float*)calloc(samples, sizeof(float));
        for (i = 0; i < samples; i++) line_data[i] = data[i][col];
    }
    Set* st = new_set(17);
    char keys[10];
    for (i = 0; i < samples; i++) {
        ftostr(line_data[i], keys, 3);
        if (!set_find(st, keys)) set_add(st, keys, j++);
    }
    *class_num = st->size;
    float** ohe = (float**)malloc(samples* sizeof(float*));
    for (i = 0; i < samples; i++) {
        ohe[i] = (float*)calloc(*class_num, sizeof(float));
        ftostr(line_data[i], keys, 3);
        j = set_call(st, keys);
        if (j != -1) ohe[i][j] = 1.0;
    }
    free_set(st);
    if (data) free(line_data);
    return ohe;
}
void free_encoder(void* encoder, char* encoder_type) {
    if (!strcmp(encoder_type, "One_hot_encoder")) {
        One_hot_encoder* encode = (One_hot_encoder*) encoder;
        free_set(encode->sample_types);
        free(encode);
    } else if (!strcmp(encoder_type, "Label_encoder")) {
        Label_encoder* encode = (Label_encoder*) encoder;
        free_set(encode->sample_types);
        free(encode);
    }
}
Simple_Imputer* simple_impute(Data_Frame* df, char* strategy, float* fill_value, char** fill_str_value) {
    Simple_Imputer* news = (Simple_Imputer*)malloc(sizeof(Simple_Imputer));
    int i, j, k;
    news->digit_data = (float*)calloc(df->col, sizeof(float));
    if (!strcmp(strategy, "mean")) {
        for (j = 0; j < df->col; j++) {
            for (i = 0, k = 0; i < df->row; i++) 
                if (df->data[i][j] == df->data[i][j]) {
                    news->digit_data[j] += df->data[i][j];
                    k++;
                }
            news->digit_data[j] /= k;
        }
    } else if (!strcmp(strategy, "median")) {
        float* cur;
        for (j = 0; j < df->col; j++, k = 0) {
            for (i = 0; i < df->row; i++) if (df->data[i][j] == df->data[i][j]) k++;
            cur = (float*)malloc(k* sizeof(float));
            for (i = 0, k = 0; i < df->row; i++, k++)
                if (df->data[i][j] == df->data[i][j]) cur[k] = df->data[i][j];
            news->digit_data[j] = median(cur, k);
            free(cur);
        }
    } else if (!strcmp(strategy, "constant")) for (i = 0; i < df->col; i++) news->digit_data[i] = fill_value[i];
    if (df->str_cols[0] != 0 && fill_str_value) {
        news->str_data = (char**)malloc((df->str_cols[0] + 1)* sizeof(char*));
        if (!strcmp(fill_str_value[0], "most_frequent")) {
            Set** str_frequence = (Set**)malloc(df->str_cols[0]* sizeof(Set*));
            for (i = 0; i < df->str_cols[0]; i++) {
                str_frequence[i] = new_set(17);
                for (j = 0; j < df->row; j++) 
                    if (strcmp(df->str_data[j][i], "nan")) {
                        if (set_find(str_frequence[i], df->str_data[j][i]))
                            (*set_key_access(str_frequence[i], df->str_data[j][i]))++;
                        else set_add(str_frequence[i], df->str_data[j][i], 1);
                    }
            }
            Space* mid;
            for (k = 0; k < df->str_cols[0]; k++) {
                for (i = 0, j = 0; i < str_frequence[k]->max_size; i++) {
                    mid = str_frequence[k]->arr[i];
                    while (mid != NULL) {
                        if (mid->key > j) {
                            j = strlen(mid->keys) + 1;
                            news->str_data[k] = (char*)malloc(j* sizeof(char));
                            strcpy(news->str_data[k], mid->keys);
                            j = mid->key;
                        }
                        mid = mid->next;
                    }
                }
                free_set(str_frequence[k]);
            }
            free(str_frequence);
        } else {
            for (i = 0; i < df->str_cols[0]; i++) {
                if (fill_str_value[i]) {
                    news->str_data[i] = (char*)malloc((strlen(fill_str_value[i]) + 1)* sizeof(char));
                    strcpy(news->str_data[i], fill_str_value[i]);
                }
            }
        }
        news->str_data[df->str_cols[0]] = (char*)malloc(5* sizeof(char));
        strcpy(news->str_data[df->str_cols[0]], "null");
    }
    return news;
}
void simple_impute_transform(Data_Frame* df, Simple_Imputer* imputer) {
    int i, j;
    for (i = 0; i < df->col; i++) {
        for (j = 0; j < df->row; j++)
            if (df->data[j][i] != df->data[j][i]) df->data[j][i] = imputer->digit_data[i];
    }
    if (imputer->str_data) {
        for (i = 0; i < df->str_cols[0]; i++) {
            if (imputer->str_data[i]) {
                for (j = 0; j < df->row; j++)
                    if (!strcmp(df->str_data[j][i], "nan"))
                        strcpy(df->str_data[j][i], imputer->str_data[i]);
            }
        }
    }
}
void free_simple_imputer(Simple_Imputer* imputer) {
    if (imputer->digit_data) free(imputer->digit_data);
    if (imputer->str_data) {
        int i;
        for (i = 0; strcmp(imputer->str_data[i], "null"); i++) free(imputer->str_data[i]);
        free(imputer->str_data[i]);
        free(imputer->str_data);
    }
    free(imputer);
}
/*void train_test_split_ds(Dataset* data, Dataset* train, Dataset* test, float test_size, int random_state) {
    train->features = data->features;
    test->features = data->features;
    test->samples = round(test_size* data->samples);
    train->samples = data->samples - test->samples;
    test->x = (float**)malloc(test->samples* sizeof(float*));
    test->y = (float*)malloc(test->samples* sizeof(float));
    train->x = (float**)malloc(train->samples* sizeof(float*));
    train->y = (float*)malloc(train->samples* sizeof(float));

    int* random_i = (int*)malloc(data->samples* sizeof(int)), i;
    for (i = 0; i < data->samples; i++) random_i[i] = i;
    shuffle_index(random_i, data->samples, random_state);
    
    for (i = 0; i < test->samples; i++) {
        test->x[i] = (float*)malloc((test->features + 1)* sizeof(float));
        dataset_sample_copy(data, random_i[i], test, i);
    }
    for (int e = 0; i < data->samples; i++, e++) {
        train->x[e] = (float*)malloc((train->features + 1)* sizeof(float));
        dataset_sample_copy(data, random_i[i], train, e);
    }
    free(random_i);
}*/
