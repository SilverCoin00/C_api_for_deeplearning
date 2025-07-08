#pragma once
#include "Keras_core.h"

const char* Monitors[] = {"loss", "accuracy", "val_loss", "val_accuracy"};
const char* Metrics[] = {"accuracy", "mse", "mae", "binary_accuracy", "categorical_accuracy"};
const char* Loss[] = {"mse", "binary_crossentropy", "categorical_crossentropy"};

int activation_encode(char* activation) {
    if (!strcmp(activation, "none") || !strcmp(activation, "linear")) return 0;
    else if (!strcmp(activation, "relu")) return 1;
    else if (!strcmp(activation, "sigmoid")) return 2;
    else if (!strcmp(activation, "tanh")) return 3;
    else if (!strcmp(activation, "softmax")) return 4;
    else if (!strcmp(activation, "softplus")) return 5;
    return -1;
}
Matrix* activation_func(Matrix* z, int activate_type) {  // a = f(z)
    int i, j;
    Matrix* a = new_matrix(z->row, z->col);
    if (activate_type == 0) {
        copy_matrix(a, z);
    } else if (activate_type == 1) {
        for (i = 0; i < z->row; i++) 
            for (j = 0; j < z->col; j++) if (z->val[i][j] > 0) a->val[i][j] = z->val[i][j];
    } else if (activate_type == 2) {
        for (i = 0; i < z->row; i++) 
            for (j = 0; j < z->col; j++) a->val[i][j] = 1 / (1 + exp(-z->val[i][j]));
    } else if (activate_type == 3) {
        for (i = 0; i < z->row; i++) 
            for (j = 0; j < z->col; j++) a->val[i][j] = 2 / (1 + exp(-2* z->val[i][j])) - 1;
    } else if (activate_type == 4) {
        float sum, max;
        for (i = 0; i < z->row; i++) {
            sum = 0;
            max = z->val[i][0];
            for (j = 1; j < z->col; j++) if (z->val[i][j] > max) max = z->val[i][j];
            for (j = 0; j < z->col; j++) {
                a->val[i][j] = exp(z->val[i][j] - max);
                sum += a->val[i][j];
            }
            for (j = 0; j < z->col; j++) a->val[i][j] /= sum;
        }
    } else if (activate_type == 5) {
        for (i = 0; i < z->row; i++) 
            for (j = 0; j < z->col; j++) a->val[i][j] = log(1 + exp(z->val[i][j]));
    }
    return a;
}
Tensor* activation_func_2D(Tensor* z, int activate_type) {
    Tensor* a = (Tensor*)malloc(sizeof(Tensor));
    a->depth = z->depth;
    a->mat = (Matrix**)malloc(z->depth* sizeof(Matrix*));
    for (int i = 0; i < z->depth; i++) a->mat[i] = activation_func(z->mat[i], activate_type);
    return a;
}
Matrix* activation_derivative(Matrix* a, Matrix* z, int activate_type) {  // da|dz = f'(z)
    int i, j;
    Matrix* deriv = new_matrix(a->row, a->col);
    if (activate_type == 0) {
        for (i = 0; i < a->row; i++)
            for (j = 0; j < a->col; j++) deriv->val[i][j] = 1;
    } else if (activate_type == 1) {
        for (i = 0; i < a->row; i++)
            for (j = 0; j < a->col; j++) deriv->val[i][j] = (z->val[i][j] > 0.0f) ? 1.0f : 0.0f;
    } else if (activate_type == 2) {
        for (i = 0; i < a->row; i++)
            for (j = 0; j < a->col; j++) deriv->val[i][j] = a->val[i][j]* (1 - a->val[i][j]);
    } else if (activate_type == 3) {
        for (i = 0; i < a->row; i++)
            for (j = 0; j < a->col; j++) deriv->val[i][j] = 1 - a->val[i][j]* a->val[i][j];
    } else if (activate_type == 5) {
        for (i = 0; i < a->row; i++)
            for (j = 0; j < a->col; j++) deriv->val[i][j] = 1 / (1 + exp(-z->val[i][j]));
    }
    return deriv;
}
Tensor* activation_derivative_2D(Tensor* a, Tensor* z, int activate_type) {
    Tensor* deriv = new_tensor(a->mat[0]->row, a->mat[0]->col, a->depth);
    for (int i = 0; i < a->depth; i++) 
        deriv->mat[i] = activation_derivative(a->mat[i], z->mat[i], activate_type);
    return deriv;
}

Optimizer* new_optimizer(char* optimizer, float learning_rate, float momentum, int nesterov, 
                float beta_1, float beta_2, float rho, float epsilon, float Grad_0) {
    Optimizer* newo = (Optimizer*)calloc(1, sizeof(Optimizer));
    if (!strcmp(optimizer, "SGD")) {
        newo->type = 0;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.01f;
        newo->momentum = momentum > 0 ? momentum : 0;
        newo->nesterov = !!nesterov;
    } else if (!strcmp(optimizer, "Adagrad")) {
        newo->type = 1;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->init_accumulator_grad = Grad_0 > 0 ? Grad_0 : 0.1f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
    } else if (!strcmp(optimizer, "RMSProp")) {
        newo->type = 2;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->rho = rho > 0 ? rho : 0.9f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
        newo->momentum = momentum > 0 ? momentum : 0;
    } else if (!strcmp(optimizer, "Adadelta")) {
        newo->type = 3;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->rho = rho > 0 ? rho : 0.95f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
    } else if (!strcmp(optimizer, "Adam")) {
        newo->type = 4;
        newo->learning_rate = learning_rate > 0 ? learning_rate : 0.001f;
        newo->epsilon = epsilon > 0 ? epsilon : 1e-7f;
        newo->beta_1 = beta_1 > 0 ? beta_1 : 0.9f;
        newo->beta_2 = beta_2 > 0 ? beta_2 : 0.999f;
    } else {
        free(newo);
        printf("Warning: Unreconized optimizer type !!");
        return NULL;
    }
    return newo;
}
Early_Stopping* new_earlystopping(char* monitor, float min_delta, int patience, float baseline, int restore_best_weights) {
    Early_Stopping* new_es = (Early_Stopping*)malloc(sizeof(Early_Stopping));
    int i;
    for (i = 0; i < 4; i++) 
        if (!strcmp(monitor, Monitors[i])) {
            new_es->monitor = i;
            if (i % 2 == 0) new_es->last_monitor_val = 1.0f / 0.0f;
            else new_es->last_monitor_val = 0.0f;
            new_es->best_monitor_val = new_es->last_monitor_val;
            break;
        }
    if (i == 4) {
        printf("Error: Unreconized monitor type !!");
        free(new_es);
        return NULL;
    }
    new_es->patience = patience > 0 ? patience : 0;
    new_es->min_delta = min_delta > 0 ? min_delta : 0;
    new_es->baseline = baseline;
    new_es->restore_best_weights = !!restore_best_weights;
    return new_es;
}
int early_stopping_check(Early_Stopping* es, float monitor_val) {
    if (es->monitor % 2 == 0) {
        if (monitor_val > es->last_monitor_val - es->min_delta) return 0;  // not improve
        else {
            if (monitor_val < es->best_monitor_val) {
                es->best_monitor_val = monitor_val;
                return 2;
            } else return 1;
        }
    } else {
        if (monitor_val < es->last_monitor_val + es->min_delta) return 0;
        else {
            if (monitor_val > es->best_monitor_val) {
                es->best_monitor_val = monitor_val;
                return 2;
            } else return 1;
        }
    }
}
void store_best_weights(Sequential* model) {
    FILE* f = fopen("Do_not_touch_while_training.txt", "wb");
    for (int i = 0; i < model->num_layers; i++) {
        switch (model->layer[i]->type) {
            case 1: binary_write_weights(f, ((Dense*) model->layer[i]->layer)->weights); break;
            case 3: binary_write_kernels(f, ((Conv*) model->layer[i]->layer)->filter); break;
        }
    }
    fclose(f);
}
void restore_best_weights(Sequential* model) {
    FILE* f = fopen("Do_not_touch_while_training.txt", "rb");
    for (int i = 0; i < model->num_layers; i++) {
        switch (model->layer[i]->type) {
            case 1: binary_read_weights(f, ((Dense*) model->layer[i]->layer)->weights); break;
            case 3: binary_read_kernels(f, ((Conv*) model->layer[i]->layer)->filter); break;
        }
    }
    fclose(f);
}
void model_compile(Sequential* model, Optimizer* optimize, char* loss, char* metrics) {
    model->compiler = (Model_Compiler*)malloc(sizeof(Model_Compiler));
    model->compiler->optimize = optimize;
    int i;
    Dense* dl; Conv* cl;
    if (model->compiler->optimize) {
        for (i = 0; i < model->num_layers; i++) {
            switch (model->layer[i]->type) {
                case 1: dl = (Dense*) model->layer[i]->layer;
                        dl->pre_velo = init_weights(dl->weights->w->row, dl->weights->w->col, 0);
                        dl->acc_grad = init_weights(dl->weights->w->row, dl->weights->w->col, 0);
                        fill_weights(dl->acc_grad, model->compiler->optimize->init_accumulator_grad); break;
                case 3: cl = (Conv*) model->layer[i]->layer;
                        cl->pre_velo = init_kernels(cl->filter->w[0]->mat[0]->row, cl->filter->w[0]->mat[0]->col, cl->filter->channel, cl->filter->w[0]->depth, 0);
                        cl->acc_grad = init_kernels(cl->filter->w[0]->mat[0]->row, cl->filter->w[0]->mat[0]->col, cl->filter->channel, cl->filter->w[0]->depth, 0);
                        fill_kernels(cl->acc_grad, model->compiler->optimize->init_accumulator_grad); break;
            }
        }
    }
    for (i = 0; i < 3; i++) 
        if (!strcmp(Loss[i], loss)) {
            model->compiler->loss_type = i + 1;
            break;
        }
    for (i = 0; i < 5; i++) 
        if (!strcmp(Metrics[i], metrics)) {
            model->compiler->metrics_type = i;
            break;
        }
}
static float mae_func_1D(Matrix* y_pred, Matrix* y_true) {
    int i, j;
    float loss = 0;
    for (i = 0; i < y_true->row; i++)
        for (j = 0; j < y_true->col; j++) 
            loss += fabs(y_pred->val[i][j] - y_true->val[i][j]);
    return loss / y_true->col / y_true->row;
}
static float mae_func_3D(int batch, Tensor** y_pred, Tensor** y_true) {
    int i, j, k, h;
    float loss = 0, dif;
    int depth =  y_true[0]->depth, row = y_true[0]->mat[0]->row, col = y_true[0]->mat[0]->col;
    for (i = 0; i < batch; i++) {
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) 
                    loss += fabs(y_pred[i]->mat[j]->val[k][h] - y_true[i]->mat[j]->val[k][h]);
    }
    return loss / batch / depth / row / col;
}
static float mse_func_1D(Matrix* y_pred, Matrix* y_true) {
    int i, j;
    float loss = 0, dif;
    for (i = 0; i < y_true->row; i++)
        for (j = 0; j < y_true->col; j++) {
            dif = y_pred->val[i][j] - y_true->val[i][j];
            loss += dif* dif;
        }
    return loss / y_true->col / y_true->row;
}
static float mse_func_3D(int batch, Tensor** y_pred, Tensor** y_true) {
    int i, j, k, h;
    float loss = 0, dif;
    int depth =  y_true[0]->depth, row = y_true[0]->mat[0]->row, col = y_true[0]->mat[0]->col;
    for (i = 0; i < batch; i++) {
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) {
                    dif = y_pred[i]->mat[j]->val[k][h] - y_true[i]->mat[j]->val[k][h];
                    loss += dif* dif;
                }
    }
    return loss / batch / depth / row / col;
}
static float binary_crossentropy_1D(Matrix* y_pred, Matrix* y_true) {
    float loss = 0, pred;
    for (int i = 0; i < y_true->row; i++) {
        pred = y_pred->val[i][0];
        pred = pred > 1e-8 ? pred : 1e-8;
        pred = pred < 1 - 1e-8 ? pred : 1 - 1e-8;
        loss += - (y_true->val[i][0]* logf(pred) + (1 - y_true->val[i][0])* logf(1 - pred));
    }
    return loss / y_true->row;
}
static float binary_crossentropy_3D(int batch, Tensor** y_pred, Tensor** y_true) {
    float loss = 0, pred, truee;
    int d = y_true[0]->depth, r = y_true[0]->mat[0]->row, c = y_true[0]->mat[0]->col, i, j, k, h;
    for (i = 0; i < batch; i++) 
        for (j = 0; j < d; j++) 
            for (k = 0; k < r; k++) 
                for (h = 0; h < c; h++) {
                    truee = y_true[i]->mat[j]->val[k][h];
                    pred = y_pred[i]->mat[j]->val[k][h];
                    pred = pred > 1e-8 ? pred : 1e-8;
                    pred = pred < 1 - 1e-8 ? pred : 1 - 1e-8;
                    loss += - (truee * logf(pred) + (1 - truee) * logf(1 - pred));
                }
    return loss / batch / d / r / c;
}
static float categorical_crossentropy_1D(Matrix* y_pred, Matrix* y_true) {
    int i, j;
    float loss = 0;
    for (i = 0; i < y_true->row; i++)
        for (j = 0; j < y_true->col; j++) if (y_true->val[i][j] == 1.0) loss -= logf(y_pred->val[i][j] + 1e-10);
    return loss / y_true->row;
}
static float categorical_crossentropy_3D(int batch, Tensor** y_pred, Tensor** y_true) {
    int i, j, k, h;
    float loss = 0, pred;
    return loss;
}
static float binary_accuracy_1D(Matrix* y_pred, Matrix* y_true) {
    float metric = 0;
    for (int i = 0; i < y_true->row; i++)
        metric += (y_pred->val[i][0] > 0.5 ? 1 : 0) == y_true->val[i][0];
    return metric / y_true->row;
}
static float binary_accuracy_3D(int batch, Tensor** y_pred, Tensor** y_true) {
    float metric = 0;
    int d = y_true[0]->depth, r = y_true[0]->mat[0]->row, c = y_true[0]->mat[0]->col, i, j, k, h;
    for (i = 0; i < batch; i++) 
        for (j = 0; j < d; j++) 
            for (k = 0; k < r; k++) 
                for (h = 0; h < c; h++) 
                    metric += (y_pred[i]->mat[j]->val[k][h] > 0.5 ? 1 : 0) == y_true[i]->mat[j]->val[k][h];
    return metric / batch / d / r / c;
}
static float categorical_accuracy_1D(Matrix* y_pred, Matrix* y_true) {
    float metric = 0;
    int max;
    for (int i = 0, j; i < y_true->row; i++) {
        max = 0;
        for (j = 1; j < y_true->col; j++) if (y_pred->val[i][j] > y_pred->val[i][max]) max = j;
        metric += y_true->val[i][max];
    }
    return metric / y_true->row;
}
static float categorical_accuracy_3D(int batch, Tensor** y_pred, Tensor** y_true) {
    int i, j, k, h;
    float metric = 0, pred;
    return metric;
}
static float accuracy(int output_dim, void* y_pred, void* y_true, int batch) {
    if (output_dim == 1) {
        if (((Matrix*) y_true)->col == 1) return binary_accuracy_1D((Matrix*) y_pred, (Matrix*) y_true);
        else return categorical_accuracy_1D((Matrix*) y_pred, (Matrix*) y_true);
    } else if (output_dim == 3) {
        return 0;
    }
}
float loss_func(int loss_type, int output_dim, void* y_pred, void* y_true, int batch) {
    int i, j;
    if (loss_type == 1) {
        if (output_dim == 1) return mse_func_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return mse_func_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    } else if (loss_type == 2) {
        if (output_dim == 1) return binary_crossentropy_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return binary_crossentropy_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    } else if (loss_type == 3) {
        if (output_dim == 1) return categorical_crossentropy_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return categorical_crossentropy_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    }
    return 0;
}
float metrics_func(int metrics_type, int output_dim, void* y_pred, void* y_true, int batch) {
    int i, j;
    if (metrics_type == 0) {
        return accuracy(output_dim, y_pred, y_true, batch);
    } else if (metrics_type == 1) {
        if (output_dim == 1) return mse_func_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return mse_func_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    } else if (metrics_type == 2) {
        if (output_dim == 1) return mae_func_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return mae_func_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    } else if (metrics_type == 3) {
        if (output_dim == 1) return binary_accuracy_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return binary_accuracy_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    } else if (metrics_type == 4) {
        if (output_dim == 1) return categorical_accuracy_1D((Matrix*) y_pred, (Matrix*) y_true);
        else if (output_dim == 3) return categorical_accuracy_3D(batch, (Tensor**) y_pred, (Tensor**) y_true);
    }
    return 0;
}
void check_nesterov(Weights* cur_w, Weights* pre_velo, Optimizer* opt) {
    float moment = opt->nesterov* opt->momentum;
    if (!moment) return ;
    for (int i = 0, j; i < cur_w->w->col; i++) {
        for (j = 0; j < cur_w->w->row; j++)
            cur_w->w->val[j][i] -= moment* pre_velo->w->val[j][i];
        cur_w->b[i] -= moment* pre_velo->b[i];
    }
}
void check_nestorov_2D(Kernel* cur_filter, Kernel* pre_velo, Optimizer* opt) {
    float moment = opt->nesterov* opt->momentum;
    if (!moment) return ;
    for (int i = 0, j, k, h; i < cur_filter->channel; i++) {
        for (j = 0; j < cur_filter->w[i]->depth; j++) 
            for (k = 0; k < cur_filter->w[i]->mat[j]->row; k++) 
                for (h = 0; h < cur_filter->w[i]->mat[j]->col; h++) 
                    cur_filter->w[i]->mat[j]->val[k][h] -= moment* pre_velo->w[i]->mat[j]->val[k][h];
        cur_filter->b[i] -= moment* pre_velo->b[i];
    }
}
void SGD_1D(Weights* w, Weights* grad, float lr, Weights* pre_velo, float moment) {
    int nodes = w->w->col;
    scalar_multiply(grad->w, lr);
    vector_scale(grad->b, lr, nodes);
    scalar_multiply(pre_velo->w, moment);
    vector_scale(pre_velo->b, moment, nodes);
    get_summ(pre_velo->w, grad->w);
    get_sumv(pre_velo->b, grad->b, nodes);
    get_minusm(w->w, pre_velo->w);
    get_minusv(w->b, pre_velo->b, nodes);
}
void SGD_3D(Kernel* w, Kernel* grad, float lr, Kernel* pre_velo, float moment) {
    int channel = w->channel;
    for (int i = 0; i < channel; i++) {
        tensor_scalar_multiply(grad->w[i], lr);
        tensor_scalar_multiply(pre_velo->w[i], moment);
        pre_velo->b[i] *= moment;
        get_sumt(pre_velo->w[i], grad->w[i]);
        pre_velo->b[i] += grad->b[i]* lr;
        get_minust(w->w[i], pre_velo->w[i]);
        w->b[i] -= pre_velo->b[i];
    }
}
void gradient_descent(int type, void* layer, float learning_rate, float moment) {
    // v(i) = moment.v(i-1) + lr.grad(i), w -= v(i)
    if (type == 1) {
        Dense* l = (Dense*) layer;
        SGD_1D(l->weights, l->deriv, learning_rate, l->pre_velo, moment);
    } else if (type == 3) {
        Conv* l = (Conv*) layer;
        SGD_3D(l->filter, l->deriv, learning_rate, l->pre_velo, moment);
    }
}
void Adagrad_1D(Weights* w, Weights* grad, float lr, Weights* acc_grad, float ep) {
    int row = w->w->row, col = w->w->col, i, j;
    for (i = 0; i < col; i++) {
        for (j = 0; j < row; j++) {
            acc_grad->w->val[j][i] += grad->w->val[j][i]* grad->w->val[j][i];
            w->w->val[j][i] -= lr* grad->w->val[j][i] / sqrtf(acc_grad->w->val[j][i] + ep);
        }
        acc_grad->b[i] += grad->b[i]* grad->b[i];
        w->b[i] -= lr* grad->b[i] / sqrtf(acc_grad->b[i] + ep);
    }
}
void Adagrad_3D(Kernel* w, Kernel* grad, float lr, Kernel* acc_grad, float ep) {
    int channel = w->channel, depth = w->w[0]->depth, row = w->w[0]->mat[0]->row, col = w->w[0]->mat[0]->col, i, j, k, h;
    for (i = 0; i < channel; i++) {
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) {
                    acc_grad->w[i]->mat[j]->val[k][h] += grad->w[i]->mat[j]->val[k][h]* grad->w[i]->mat[j]->val[k][h];
                    w->w[i]->mat[j]->val[k][h] -= lr* grad->w[i]->mat[j]->val[k][h] / sqrtf(acc_grad->w[i]->mat[j]->val[k][h] + ep);
                }
        acc_grad->b[i] += grad->b[i]* grad->b[i];
        w->b[i] -= lr* grad->b[i] / sqrtf(acc_grad->b[i] + ep);
    }
}
void adaptive_gradient_descent(int type, void* layer, float learning_rate, float epsilon) {
    // acc_grad(i) = sum[1->i](grad(i)^2), w -= lr.grad(i) / sqrt(acc_grad(i) + e)
    if (type == 1) {
        Dense* l = (Dense*) layer;
        Adagrad_1D(l->weights, l->deriv, learning_rate, l->acc_grad, epsilon);
    } else if (type == 3) {
        Conv* l = (Conv*) layer;
        Adagrad_3D(l->filter, l->deriv, learning_rate, l->acc_grad, epsilon);
    }
}
void RMSprop_1D(Weights* w, Weights* grad, float lr, Weights* pre_velo, float moment, Weights* acc_grad, float r, float ep) {
    int row = w->w->row, col = w->w->col, i, j;
    for (i = 0; i < col; i++) {
        for (j = 0; j < row; j++) {
            acc_grad->w->val[j][i] = r* acc_grad->w->val[j][i] + (1 - r)* grad->w->val[j][i]* grad->w->val[j][i];
            pre_velo->w->val[j][i] = moment* pre_velo->w->val[j][i] + lr* grad->w->val[j][i] / sqrtf(acc_grad->w->val[j][i] + ep);
            w->w->val[j][i] -= pre_velo->w->val[j][i];
        }
        acc_grad->b[i] = r* acc_grad->b[i] + (1 - r)* grad->b[i]* grad->b[i];
        pre_velo->b[i] = moment* pre_velo->b[i] + lr* grad->b[i] / sqrtf(acc_grad->b[i] + ep);
        w->b[i] -= pre_velo->b[i];
    }
}
void RMSprop_3D(Kernel* w, Kernel* grad, float lr, Kernel* pre_velo, float moment, Kernel* acc_grad, float r, float ep) {
    int channel = w->channel, depth =  w->w[0]->depth, row = w->w[0]->mat[0]->row, col = w->w[0]->mat[0]->col, i, j, k, h;
    for (i = 0; i < channel; i++) {
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) {
                    acc_grad->w[i]->mat[j]->val[k][h] *= r;
                    acc_grad->w[i]->mat[j]->val[k][h] += (1 - r)* grad->w[i]->mat[j]->val[k][h]* grad->w[i]->mat[j]->val[k][h];
                    pre_velo->w[i]->mat[j]->val[k][h] *= moment;
                    pre_velo->w[i]->mat[j]->val[k][h] += lr* grad->w[i]->mat[j]->val[k][h] / sqrtf(acc_grad->w[i]->mat[j]->val[k][h] + ep);
                    w->w[i]->mat[j]->val[k][h] -= pre_velo->w[i]->mat[j]->val[k][h];
                }
        acc_grad->b[i] *= r;
        acc_grad->b[i] += (1 - r)* grad->b[i]* grad->b[i];
        pre_velo->b[i] *= moment;
        pre_velo->b[i] += lr* grad->b[i] / sqrtf(acc_grad->b[i] + ep);
        w->b[i] -= pre_velo->b[i];
    }
}
void root_mean_square_propagation(int type, void* layer, float learning_rate, float moment, float rho, float epsilon) {
    // acc_grad(i) = (1 - rho). sum[1->i](rho^(i - j). grad(j))
    // v(i) = moment.v(i-1) + lr.grad(i) / sqrt(acc_grad(i) + e), w -= v(i)
    if (type == 1) {
        Dense* l = (Dense*) layer;
        RMSprop_1D(l->weights, l->deriv, learning_rate, l->pre_velo, moment, l->acc_grad, rho, epsilon);
    } else if (type == 3) {
        Conv* l = (Conv*) layer;
        RMSprop_3D(l->filter, l->deriv, learning_rate, l->pre_velo, moment, l->acc_grad, rho, epsilon);
    }
}
void Adadelta_1D(Weights* w, Weights* grad, float lr, Weights* acc_grad, float r, float ep, Weights* d) {
    int row = w->w->row, col = w->w->col, i, j;
    for (i = 0; i < col; i++) {
        for (j = 0; j < row; j++) {
            acc_grad->w->val[j][i] = r* acc_grad->w->val[j][i] + (1 - r)* grad->w->val[j][i]* grad->w->val[j][i];
            grad->w->val[j][i] *= lr* sqrtf((d->w->val[j][i] + ep) / (acc_grad->w->val[j][i] + ep));
            w->w->val[j][i] -= grad->w->val[j][i];
            d->w->val[j][i] = r* d->w->val[j][i] + (1 - r)* grad->w->val[j][i]* grad->w->val[j][i];
        }
        acc_grad->b[i] = r* acc_grad->b[i] + (1 - r)* grad->b[i]* grad->b[i];
        grad->b[i] *= lr* sqrtf((d->b[i] + ep) / (acc_grad->b[i] + ep));
        w->b[i] -= grad->b[i];
        d->b[i] = r* d->b[i] + (1 - r)* grad->b[i]* grad->b[i];
    }
}
void Adadelta_3D(Kernel* w, Kernel* grad, float lr, Kernel* acc_grad, float r, float ep, Kernel* d) {
    int channel = w->channel, depth =  w->w[0]->depth, row = w->w[0]->mat[0]->row, col = w->w[0]->mat[0]->col, i, j, k, h;
    for (i = 0; i < channel; i++) {
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) {
                    acc_grad->w[i]->mat[j]->val[k][h] *= r;
                    acc_grad->w[i]->mat[j]->val[k][h] += (1 - r)* grad->w[i]->mat[j]->val[k][h]* grad->w[i]->mat[j]->val[k][h];
                    grad->w[i]->mat[j]->val[k][h] *= lr* sqrtf((d->w[i]->mat[j]->val[k][h] + ep) / (acc_grad->w[i]->mat[j]->val[k][h] + ep));
                    w->w[i]->mat[j]->val[k][h] -= grad->w[i]->mat[j]->val[k][h];
                    d->w[i]->mat[j]->val[k][h] *= r;
                    d->w[i]->mat[j]->val[k][h] += (1 - r)* grad->w[i]->mat[j]->val[k][h]* grad->w[i]->mat[j]->val[k][h];
                }
        acc_grad->b[i] *= r;
        acc_grad->b[i] += (1 - r)* grad->b[i]* grad->b[i];
        grad->b[i] *= lr* sqrtf((d->b[i] + ep) / (acc_grad->b[i] + ep));
        w->b[i] -= grad->b[i];
        d->b[i] *= r;
        d->b[i] += (1 - r)* grad->b[i]* grad->b[i];
    }
}
void adaptive_delta_grad(int type, void* layer, float learning_rate, float rho, float epsilon) {
    // acc_grad(i) = (1 - rho). sum[1->i](rho^(i - j). grad(j))
    // grad(i) = lr. sqrt((delta(i-1) + e) / (acc_grad(i) + e)), w -= grad(i)
    // delta(i) = (1 - rho). sum[1->i](rho^(i - j). grad(j))
    if (type == 1) {
        Dense* l = (Dense*) layer;
        Adadelta_1D(l->weights, l->deriv, learning_rate, l->acc_grad, rho, epsilon, l->pre_velo);
    } else if (type == 3) {
        Conv* l = (Conv*) layer;
        Adadelta_3D(l->filter, l->deriv, learning_rate, l->acc_grad, rho, epsilon, l->pre_velo);
    }
}
void Adam_1D(Weights* w, Weights* grad, float lr, Weights* acc_grad, 
                float beta_1, float beta_2, float ep, Weights* pre_velo, int* is_new) {
    static float b1 = 1, b2 = 1;
    if (*is_new == 2) b1 = beta_1, b2 = beta_2, *is_new -= 2;            // start new training
    else if (*is_new == 1) b1 *= beta_1, b2 *= beta_2, (*is_new)--;      // next epochs
    float velo, accg;
    int row = w->w->row, col = w->w->col, i, j;
    for (i = 0; i < col; i++) {
        for (j = 0; j < row; j++) {
            pre_velo->w->val[j][i] = beta_1* pre_velo->w->val[j][i] + (1 - beta_1)* grad->w->val[j][i];
            acc_grad->w->val[j][i] = beta_2* acc_grad->w->val[j][i] + (1 - beta_2)* grad->w->val[j][i]* grad->w->val[j][i];
            velo = pre_velo->w->val[j][i] / (1 - b1), accg = acc_grad->w->val[j][i] / (1 - b2);
            w->w->val[j][i] -= lr* velo / (sqrtf(accg) + ep);
        }
        pre_velo->b[i] = beta_1* pre_velo->b[i] + (1 - beta_1)* grad->b[i];
        acc_grad->b[i] = beta_2* acc_grad->b[i] + (1 - beta_2)* grad->b[i]* grad->b[i];
        velo = pre_velo->b[i] / (1 - b1), accg = acc_grad->b[i] / (1 - b2);
        w->b[i] -= lr* velo / (sqrtf(accg) + ep);
    }
}
void Adam_3D(Kernel* w, Kernel* grad, float lr, Kernel* acc_grad, 
                float beta_1, float beta_2, float ep, Kernel* pre_velo, int* is_new) {
    static float b1 = 1, b2 = 1;
    if (*is_new == 2) b1 = beta_1, b2 = beta_2, *is_new -= 2;
    else if (*is_new == 1) b1 *= beta_1, b2 *= beta_2;
    float velo, accg;
    int channel = w->channel, depth =  w->w[0]->depth, row = w->w[0]->mat[0]->row, col = w->w[0]->mat[0]->col, i, j, k, h;
    for (i = 0; i < channel; i++) {
        for (j = 0; j < depth; j++) 
            for (k = 0; k < row; k++) 
                for (h = 0; h < col; h++) {
                    pre_velo->w[i]->mat[j]->val[k][h] *= beta_1;
                    acc_grad->w[i]->mat[j]->val[k][h] *= beta_2;
                    pre_velo->w[i]->mat[j]->val[k][h] += (1 - beta_1)* grad->w[i]->mat[j]->val[k][h];
                    acc_grad->w[i]->mat[j]->val[k][h] += (1 - beta_2)* grad->w[i]->mat[j]->val[k][h]* grad->w[i]->mat[j]->val[k][h];
                    velo = pre_velo->w[i]->mat[j]->val[k][h] / (1 - b1), accg = acc_grad->w[i]->mat[j]->val[k][h] / (1 - b2);
                    w->w[i]->mat[j]->val[k][h] -= lr* velo / (sqrtf(accg) + ep);
                }
        pre_velo->b[i] *= beta_1;
        acc_grad->b[i] *= beta_2;
        pre_velo->b[i] += (1 - beta_1)* grad->b[i];
        acc_grad->b[i] += (1 - beta_2)* grad->b[i]* grad->b[i];
        velo = pre_velo->b[i] / (1 - b1), accg = acc_grad->b[i] / (1 - b2);
        w->b[i] -= lr* velo / (sqrtf(accg) + ep);
    }
}
void adaptive_moment_estimation(int type, void* layer, float learning_rate, float beta_1, float beta_2, float epsilon, int* is_new) {
    // pre_velo(i) = (1 - beta_1). sum[1->i](beta_1^(i - j). grad(j)), pre_velo[t] /= (1 - beta_1^t)
    // acc_grad(i) = (1 - beta_2). sum[1->i](beta_2^(i - j). grad(j)), acc_grad[t] /= (1 - beta_2^t)
    // grad(i) = lr. pre_velo(i) / (sqrt(acc_grad(i)) + epsilon)
    if (type == 1) {
        Dense* l = (Dense*) layer;
        Adam_1D(l->weights, l->deriv, learning_rate, l->acc_grad, beta_1, beta_2, epsilon, l->pre_velo, is_new);
    } else if (type == 3) {
        Conv* l = (Conv*) layer;
        Adam_3D(l->filter, l->deriv, learning_rate, l->acc_grad, beta_1, beta_2, epsilon, l->pre_velo, is_new);
    }
}
