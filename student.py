import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense

def create_models(input_channels=1):
    img_inputs = Input(shape=(None, None, input_channels))

    seq_1 = Sequential([
        Conv2D(
            filters=32, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid'
        ),
        Conv2D(
            filters=64, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        Conv2D(
            filters=64, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid'
        ),
        Conv2D(
            filters=64, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        Conv2D(
            filters=64, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid'
        ),
    ])(img_inputs)

    Sf = Conv2D(
        filters=512, 
        kernel_size=5, 
        strides=1, 
        activation='relu',
        padding='same'
    )(seq_1)

    Sh = Conv2D(
        filters=1, 
        kernel_size=1, 
        strides=1, 
        padding='same'
    )(Sf)

    seg_model = Model(inputs=img_inputs, outputs=[Sf, Sh])

    feature_inputs = Input(shape = (None, None) + (Sf.shape[3] + Sh.shape[3],))

    Cf = Sequential([
        MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid'
        ),
        Conv2D(
            filters=8, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid'
        ),
        Conv2D(
            filters=16, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
        MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid'
        ),
        Conv2D(
            filters=32, 
            kernel_size=5, 
            strides=1, 
            activation='relu',
            padding='same'
        ),
    ])(feature_inputs)

    Ga_Cf = GlobalAveragePooling2D()(Cf)
    Gm_Cf = GlobalMaxPooling2D()(Cf)
    Ga_Sh = GlobalAveragePooling2D()(feature_inputs[:,:,:,-1:])
    Gm_Sh = GlobalMaxPooling2D()(feature_inputs[:,:,:,-1:])

    clf_features = tf.concat([Ga_Cf, Gm_Cf, Ga_Sh, Gm_Sh], axis=1)

    Cp = Dense(1)(clf_features)

    clf_model = Model(inputs=feature_inputs, outputs=Cp)

    return seg_model, clf_model
