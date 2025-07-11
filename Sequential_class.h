#pragma once
#include "Keras_core.h"

const int Layer_Input_Dim[] = {0, 1, 3, 3, 3, 3};
const int Layer_Output_Dim[] = {0, 1, 1, 3, 3, 3};
void free_sequential(Sequential* model) {
    if (model->compiler) {
        if (model->compiler->optimize) free(model->compiler->optimize);
        free(model->compiler);
    }
    for (int i = 0; i < model->num_layers; i++) {
        switch (model->layer[i]->type) {
            case 1: free_dense_layer((Dense*) (model->layer[i]->layer)); break;
            case 2: free_flatten_layer((Flatten*) (model->layer[i]->layer)); break;
            case 3: free_conv_layer((Conv*) (model->layer[i]->layer)); break;
            case 4: free_max_pooling_layer((MaxPooling*) (model->layer[i]->layer)); break;
            case 5: free_average_pooling_layer((AveragePooling*) (model->layer[i]->layer)); break;
        }
        free(model->layer[i]);
    }
    free(model);
}
static FNode* check_dropout(Keras_layer** layer, int* cur_id) {
    FNode* drop = NULL;
    add_fnode(&drop, 0);
    for ((*cur_id)++; layer[*cur_id] && layer[*cur_id]->type == -1; (*cur_id)++) {
        add_fnode(&drop, ((IDropout*) (layer[*cur_id]->layer))->drop);
        drop->data += 1.0f;
    }
    return drop;
}
Sequential* tf_keras_sequential(Keras_layer** layer) {
    int i, j, *pre_size, end_drop = 0;
    FNode* drop;
    float* inp_size;
    for (i = 0; layer[i] && layer[i]->type; i++) ;
    if (!layer[i]) {
        printf("Warning: Model does not have input layer !!");
        return NULL;
    }
    inp_size = (float*) (layer[i]->layer);

    Sequential* model = (Sequential*)calloc(1, sizeof(Sequential));
    model->num_layers = 0;
    for (j = i + 1; layer[j]; j++) 
        if(layer[j]->type > 0) {
            if (!end_drop) {
                int num = Layer_Input_Dim[layer[j]->type];
                pre_size = (int*)calloc(5, sizeof(int));
                for ( ; end_drop < num; end_drop++) pre_size[end_drop] = (int) inp_size[end_drop];
            }
            model->num_layers++;
        }
    
    model->layer = (Keras_layer**)calloc(model->num_layers, sizeof(Keras_layer*));
    for (j = 0, i++; layer[i]; i = end_drop) {
        end_drop = i;
        drop = check_dropout(layer, &end_drop);
        if(!model->layer[j]) model->layer[j] = (Keras_layer*)malloc(sizeof(Keras_layer));
        model->layer[j]->type = layer[i]->type;
        switch (layer[i]->type) {
            case 1: model->layer[j++]->layer = new_dense_layer((IDense*) (layer[i]->layer), pre_size, drop); break;
            case 2: model->layer[j++]->layer = new_flatten_layer(pre_size); break;
            case 3: model->layer[j++]->layer = new_conv_layer((IConv*) (layer[i]->layer), pre_size, drop); break;
            case 4: model->layer[j++]->layer = new_max_pooling_layer((IMPool*) (layer[i]->layer), pre_size, drop); break;
            case 5: model->layer[j++]->layer = new_average_pooling_layer((IAPool*) (layer[i]->layer), pre_size, drop); break;
        }
        free_fllist(&drop);
    }
    free(pre_size);
    return model;
}
void sequential_forward(Sequential* model, void* input, int is_training, int batch, void* output) {
    Matrix* x1 = NULL, **y1;
    Tensor** x3 = NULL, ***y3;
    model->batch_size = batch;
    switch (Layer_Input_Dim[model->layer[0]->type]) {
        case 1: x1 = (Matrix*) input;
                model->batch_size = x1->row;
                break;
        case 3: x3 = (Tensor**) input; break;
    }

    int i;
    for (i = 0; i < model->num_layers; i++) {
        switch (model->layer[i]->type) {
            case 1: dense_forward((Dense*) model->layer[i]->layer, x1, is_training, &x1); break;
            case 2: flatten_forward((Flatten*) model->layer[i]->layer, x3, model->batch_size, &x1); break;
            case 3: conv_forward((Conv*) model->layer[i]->layer, x3, model->batch_size, is_training, &x3); break;
            case 4: max_pool_forward(x3, (MaxPooling*) model->layer[i]->layer, model->batch_size, is_training, &x3); break;
            case 5: average_pool_forward(x3, (AveragePooling*) model->layer[i]->layer, model->batch_size, is_training, &x3); break;
        }
    }
    
    i--;
    switch (Layer_Output_Dim[model->layer[i]->type]) {
        case 1: y1 = (Matrix**) output;
                *y1 = get_copy_matrix(x1);
                break;
        case 3: y3 = (Tensor***) output;
                *y3 = (Tensor**)malloc(model->batch_size* sizeof(Tensor*));
                for (i = 0; i < model->batch_size; i++) (*y3)[i] = get_copy_tensor(x3[i]);
                break;
    }
}
static void* get_activated_data(Sequential* model, int layer) {
    if (layer < 0) return NULL;
    switch (model->layer[layer]->type) {
        case 1: return ((Dense*) model->layer[layer]->layer)->a;
        case 2: return ((Flatten*) model->layer[layer]->layer)->flatted;
        case 3: return ((Conv*) model->layer[layer]->layer)->a;
        case 4: return ((MaxPooling*) model->layer[layer]->layer)->pool;
        case 5: return ((AveragePooling*) model->layer[layer]->layer)->pool;
    }
    return NULL;
}
static int get_output_dim(Sequential* model) {
    switch (model->layer[model->num_layers - 1]->type) {
        case 1: case 2: return 1;
        case 3: case 4: case 5: return 3;
    }
}
static void free_model_output(Sequential* model, void* output) {
    switch (Layer_Output_Dim[model->layer[model->num_layers - 1]->type]) {
        case 1: free_matrix((Matrix*) output); break;
        case 3: for (int i = 0; i < model->batch_size; i++) free_tensor(((Tensor**) output)[i]);
                free((Tensor**) output); break;
    }
}
void sequential_backprop(Sequential* model, void* input, void* output) {
    Matrix* x1, *y1 = NULL;
    Tensor** x3, **y3 = NULL;
    int i = model->num_layers - 1, j;
    switch (model->layer[i]->type) {
        case 1: case 2: y1 = get_copy_matrix((Matrix*) output); break;
        case 3: case 4: case 5:
            y3 = (Tensor**)malloc(model->batch_size* sizeof(Tensor*));
            for (j = 0; j < model->batch_size; j++) y3[j] = get_copy_tensor(((Tensor**) output)[i]);
            break;
    }

    for (j = i; i >= 0; i--) {
        switch(model->layer[i]->type) {
            case 1: x1 = (Matrix*) get_activated_data(model, i - 1);
                    dense_backprop(x1 ? x1 : (Matrix*) input, (Dense*) model->layer[i]->layer, !(i - j), &y1, model->compiler);
                    break;
            case 2: flatten_backprop((Flatten*) model->layer[i]->layer, &y1, &y3);
                    break;
            case 3: x3 = (Tensor**) get_activated_data(model, i - 1);
                    conv_backprop(x3 ? x3 : (Tensor**) input, (Conv*) model->layer[i]->layer, !(i - j), &y3, model->compiler);
                    break;
            case 4: x3 = (Tensor**) get_activated_data(model, i - 1);
                    max_pool_backprop(x3 ? x3 : (Tensor**) input, (MaxPooling*) model->layer[i]->layer, &y3);
                    break;
            case 5: x3 = (Tensor**) get_activated_data(model, i - 1);
                    average_pool_backprop(x3 ? x3 : (Tensor**) input, (AveragePooling*) model->layer[i]->layer, &y3);
                    break;
        }
    }
    free_matrix(y1);
    if (y3) for (j = 0; j < model->batch_size; j++) free_tensor(y3[j]);
}
static void update_parameters(Sequential* model, int* times) {
    Optimizer* opt = model->compiler->optimize;
    for (int i = 0; i < model->num_layers; i++) {
        switch (opt->type) {
            case 0: gradient_descent(model->layer[i]->type, model->layer[i]->layer, opt->learning_rate, opt->momentum); break;
            case 1: adaptive_gradient_descent(model->layer[i]->type, model->layer[i]->layer, opt->learning_rate, opt->epsilon); break;
            case 2: root_mean_square_propagation(model->layer[i]->type, model->layer[i]->layer, opt->learning_rate, opt->momentum, opt->rho, opt->epsilon); break;
            case 3: adaptive_delta_grad(model->layer[i]->type, model->layer[i]->layer, opt->learning_rate, opt->rho, opt->epsilon); break;
            case 4: adaptive_moment_estimation(model->layer[i]->type, model->layer[i]->layer, opt->learning_rate, opt->beta_1, opt->beta_2, opt->epsilon, times); break;
        }
    }
}
void model_fit(Sequential* model, Dataset* data, Dataset* val, int epochs, int batch_size, float validation_split, void* call_backs) {
    Dataset* train, *batch;
    void* y_pred;
    float loss_metrics[4];                         // loss, metrics, val_loss, val_metrics
    if (validation_split == 0) train = data;
    else if (val != NULL) train = data;
    else {
        train = (Dataset*)malloc(sizeof(Dataset));
        val = (Dataset*)malloc(sizeof(Dataset));
        if (validation_split < 0 || validation_split >= 1) validation_split = 0.2;
        train_test_split(data, train, val, validation_split, epochs);
        validation_split = -999;
    }

    Early_Stopping* estop = NULL;
    int best_epoch = 0;
    float* monitor = NULL;
    if (call_backs) {
        estop = (Early_Stopping*) call_backs;
        monitor = loss_metrics + estop->monitor;
    }

    int samples = get_ds_num_samples(train);
    int* random_i = (int*)malloc(samples* sizeof(int)), i, j, times;
    for (i = 0; i < samples; i++) random_i[i] = i;
    if (batch_size <= 0 || batch_size >= samples) batch_size = samples;
    shuffle_index(random_i, samples, time(NULL));

    clock_t start, end;
    for (i = 1, times = 2; i <= epochs; i++, times++) {
        start = clock();
        loss_metrics[0] = loss_metrics[1] = 0;
        shuffle_index(random_i, samples, i);
        for (j = 0; j < samples; j += batch_size) {
            model->batch_size = samples - j < batch_size ? samples - j : batch_size;
            batch = dataset_samples_order_copy(train, random_i, j, j + model->batch_size);
            sequential_forward(model, get_ds_x(batch), 1, model->batch_size, &y_pred);
            loss_metrics[0] += loss_func(model->compiler->loss_type, get_output_dim(model), y_pred, get_ds_y(batch), model->batch_size);
            loss_metrics[1] += metrics_func(model->compiler->metrics_type, get_output_dim(model), y_pred, get_ds_y(batch), model->batch_size);
            sequential_backprop(model, get_ds_x(batch), get_ds_y(batch));
            update_parameters(model, &times);
            free_model_output(model, y_pred);
            free_dataset(batch);
        }
        end = clock();

        j = (int) ceil((float) samples / batch_size), loss_metrics[0] /= j, loss_metrics[1] /= j;
        printf("Epoch %d/%d\n%d/%d [=============================] - %.2fs - loss: %.4f - %s: %.4f", 
            i, epochs, j, j, ((double) (end - start)) / CLOCKS_PER_SEC, loss_metrics[0], Metrics[model->compiler->metrics_type], loss_metrics[1]);
        if (validation_split) {
            sequential_forward(model, get_ds_x(val), 0, get_ds_num_samples(val), &y_pred);
            loss_metrics[2] = loss_func(model->compiler->loss_type, get_output_dim(model), y_pred, get_ds_y(val), model->batch_size);
            loss_metrics[3] = metrics_func(model->compiler->metrics_type, get_output_dim(model), y_pred, get_ds_y(val), model->batch_size);
            printf(" - val_loss: %.4f - val_%s: %.4f", loss_metrics[2], Metrics[model->compiler->metrics_type], loss_metrics[3]);
            free_model_output(model, y_pred);
        }
        if (call_backs) 
            if (early_stopping_check(estop, *monitor, i, &best_epoch, model)) break;
        printf("\n");
    }
    if (call_backs && estop->restore_best_weights) {
        printf("Restoring model weights from the end of the best epoch: %d\n", best_epoch);
        restore_best_weights(model);
    }
    free(random_i);
    if (validation_split == -999) {
        free_dataset(train);
        free_dataset(val);
    }
}
void model_evaluate(Sequential* model, Dataset* data) {
    void* y_pred;
    int samples = get_ds_num_samples(data);
    sequential_forward(model, get_ds_x(data), 0, samples, &y_pred);
    float loss = loss_func(model->compiler->loss_type, get_output_dim(model), y_pred, get_ds_y(data), samples);
    float metrics = metrics_func(model->compiler->metrics_type, get_output_dim(model), y_pred, get_ds_y(data), samples);
    printf("Evaluate on test data:\n%d/%d [==============================] - loss: %.4f - %s: %.4f\n", 
            samples, samples, loss, Metrics[model->compiler->metrics_type], metrics);
    free_model_output(model, y_pred);
}
void model_save(const char* file_name, Sequential* model) {
    FILE* f = fopen(file_name, "wb");
    if (!f) {
        printf("Error: Cannot open file %s !!", file_name);
        return ;
    }
    fwrite(MODEL_KEY, sizeof(char), 32, f);
    fwrite(&(model->num_layers), sizeof(int), 1, f);
    for (int i = 0; i < model->num_layers; i++) {
        fwrite(&(model->layer[i]->type), sizeof(int), 1, f);
        switch (model->layer[i]->type) {
            case 1: binary_write_dense(f, (Dense*) model->layer[i]->layer); break;
            case 3: binary_write_conv(f, (Conv*) model->layer[i]->layer); break;
            case 4: binary_write_max_pool(f, (MaxPooling*) model->layer[i]->layer); break;
            case 5: binary_write_average_pool(f, (AveragePooling*) model->layer[i]->layer); break;
        }
    }
    fclose(f);
    printf("Model saved to file %s !\n", file_name);
}
Sequential* load_model(const char* file_name) {
    FILE* f = fopen(file_name, "rb");
    if (!f) {
        printf("Error: Cannot open file %s !!", file_name);
        return NULL;
    }
    char* s = (char*)malloc(50* sizeof(char));
    fread(s, sizeof(char), 32, f);
    s[32] = '\0';
    if (strcmp(s, MODEL_KEY)) {
        printf("Warning: No permission to read this file !!");
        free(s);
        fclose(f);
        return NULL;
    }
    free(s);
    Sequential* model = (Sequential*)calloc(1, sizeof(Sequential));
    fread(&(model->num_layers), sizeof(int), 1, f);
    model->layer = (Keras_layer**)malloc(model->num_layers* sizeof(Keras_layer*));
    for (int i = 0; i < model->num_layers; i++) {
        model->layer[i] = (Keras_layer*)malloc(sizeof(Keras_layer));
        fread(&(model->layer[i]->type), sizeof(int), 1, f);
        switch (model->layer[i]->type) {
            case 1: model->layer[i]->layer = (Dense*)calloc(1, sizeof(Dense));
                    binary_read_dense(f, (Dense*) model->layer[i]->layer); break;
            case 2: model->layer[i]->layer = (Flatten*)calloc(1, sizeof(Flatten)); break;
            case 3: model->layer[i]->layer = (Conv*)calloc(1, sizeof(Conv));
                    binary_read_conv(f, (Conv*) model->layer[i]->layer); break;
            case 4: model->layer[i]->layer = (MaxPooling*)calloc(1, sizeof(MaxPooling));
                    binary_read_max_pool(f, (MaxPooling*) model->layer[i]->layer); break;
            case 5: model->layer[i]->layer = (AveragePooling*)calloc(1, sizeof(AveragePooling));
                    binary_read_average_pool(f, (AveragePooling*) model->layer[i]->layer); break;
        }
    }
    fclose(f);
    printf("Model loaded from file %s !\n", file_name);
    return model;
}
