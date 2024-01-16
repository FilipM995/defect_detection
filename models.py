import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense

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
        filters=1024, 
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


def _conv_block(num_layers, filter_nums, kernel_sizes):
    return [
        Conv2D(
            filters=filter_nums[i], 
            kernel_size=kernel_sizes[i], 
            strides=1, 
            activation='relu',
            padding='same'
        ) for i in range(num_layers)
    ]


def create_models_from_config(config, input_channels=3):
    img_inputs = Input(shape=(None, None, input_channels))

    seg_blocks = []

    for block_conf in config['seg_model']['seq_part']:
        seg_blocks += _conv_block(
            block_conf['num_layers'], 
            block_conf['filter_nums'], 
            block_conf['kernel_sizes']
        )
        seg_blocks.append(
            MaxPool2D(
                pool_size=2,
                strides=2,
                padding='valid'
            )
        )

    seq_1 = Sequential(seg_blocks)(img_inputs)

    Sf = Conv2D(
        filters=config['seg_model']['feature_layer']['filters'], 
        kernel_size=config['seg_model']['feature_layer']['kernel_size'], 
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

    clf_layers = []

    for layer_conf in config['clf_model']:
        clf_layers.append(
            MaxPool2D(
                pool_size=2,
                strides=2,
                padding='valid'
            )
        )
        clf_layers.append(
            Conv2D(
                filters=layer_conf['filters'], 
                kernel_size=layer_conf['kernel_size'], 
                strides=1, 
                activation='relu',
                padding='same'
            )
        )

    Cf = Sequential(clf_layers)(feature_inputs)

    Ga_Cf = GlobalAveragePooling2D()(Cf)
    Gm_Cf = GlobalMaxPooling2D()(Cf)
    Ga_Sh = GlobalAveragePooling2D()(feature_inputs[:,:,:,-1:])
    Gm_Sh = GlobalMaxPooling2D()(feature_inputs[:,:,:,-1:])

    clf_features = tf.concat([Ga_Cf, Gm_Cf, Ga_Sh, Gm_Sh], axis=1)

    Cp = Dense(1)(clf_features)

    clf_model = Model(inputs=feature_inputs, outputs=Cp)

    return seg_model, clf_model


def create_models_from_config_nomaxpool(config, input_channels=3):
    img_inputs = Input(shape=(640, 232, input_channels))

    seg_blocks = []

    for block_conf in config['seg_model']['seq_part']:
        seg_blocks += _conv_block(
            block_conf['num_layers'], 
            block_conf['filter_nums'], 
            block_conf['kernel_sizes']
        )
        seg_blocks.append(
            Conv2D(
                filters=block_conf['filter_nums'][-1], 
                kernel_size=2, 
                strides=2, 
                activation='relu',
                padding='valid'
            )
        )

    seq_1 = Sequential(seg_blocks)(img_inputs)

    Sf = Conv2D(
        filters=config['seg_model']['feature_layer']['filters'], 
        kernel_size=config['seg_model']['feature_layer']['kernel_size'], 
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

    feature_inputs = Input(shape = Sh.shape[1:3] + (Sf.shape[3] + Sh.shape[3],))

    clf_layers = []

    for layer_conf in config['clf_model']:
        clf_layers.append(
            Conv2D(
                filters=layer_conf['filters'], 
                kernel_size=2, 
                strides=2, 
                activation='relu',
                padding='valid'
            )
        )
        clf_layers.append(
            Conv2D(
                filters=layer_conf['filters'], 
                kernel_size=layer_conf['kernel_size'], 
                strides=1, 
                activation='relu',
                padding='same'
            )
        )

    Cf = Sequential(clf_layers)(feature_inputs)

    Ga_Cf = GlobalAveragePooling2D()(Cf)
    # Gm_Cf = GlobalMaxPooling2D()(Cf)
    Ga_Sh = GlobalAveragePooling2D()(feature_inputs[:,:,:,-1:])
    # Gm_Sh = GlobalMaxPooling2D()(feature_inputs[:,:,:,-1:])

    clf_features = tf.concat([Ga_Cf, Ga_Sh], axis=1)

    Cp = Dense(1)(clf_features)

    clf_model = Model(inputs=feature_inputs, outputs=Cp)

    return seg_model, clf_model