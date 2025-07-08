#include "Keras_core.h"

int main() {
    char file[] = "flag.csv";
    Data_Frame* df = read_csv(file, 1000, ",");
    Dataset* ds = trans_dframe_to_dset(2, df, NULL, "t");
    free_data_frame(df);
    
    Standard_scaler* scaler = (Standard_scaler*)new_scaler("Standard_scaler");
    scaler_fit(2, get_ds_x(ds), get_ds_num_samples(ds), scaler, "Standard_scaler");
    scaler_transform(2, get_ds_x(ds), get_ds_num_samples(ds), scaler, "Standard_scaler");
    free_scaler(scaler, "Standard_scaler");

    Sequential* model = tf_keras_sequential(
        tf_keras_layers_(
        Input_(28, 28, 1),
        Conv_(5, 5, 6, 1, 1, 1, "relu"),
        AveragePooling_(2, 2, 2, 2, 0),
        Conv_(5, 5, 16, 1, 1, 0, "relu"),
        AveragePooling_(2, 2, 2, 2, 0),
        Flatten_(),
        Dense_(120, "relu"),
        Dropout_(0.2),
        Dense_(84, "relu"),
        Dropout_(0.2),
        Dense_(10, "softmax"), NULL)
    );

    Optimizer* opt = new_optimizer("SGD", 0.01, 0.9, 1, Nan, Nan, Nan, Nan, Nan);
    //Optimizer* opt = new_optimizer("Adagrad", 0.008, Nan, Nan, Nan, Nan, Nan, 1e-8, 1e-2);
    //Optimizer* opt = new_optimizer("RMSProp", 0.006, Nan, Nan, Nan, Nan, 0.9, 1e-8, Nan);
    //Optimizer* opt = new_optimizer("Adadelta", 0.1, Nan, Nan, Nan, Nan, 0.9, 1e-8, Nan);
    //Optimizer* opt = new_optimizer("Adam", 0.006, Nan, Nan, 0.9, 0.999, Nan, 1e-8, Nan);
    Early_Stopping* estop = new_earlystopping("val_loss", 1e-8, 15, Nan, 1);
    model_compile(model, opt, "categorical_crossentropy", "categorical_accuracy");
    model_fit(model, ds, NULL, 200, 16, 0.2, estop);
    
    

    free_sequential(model);
    free(estop);
    free_dataset(ds);

    return 0;
}
