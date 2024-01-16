from random import choices
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from time import time
import argparse
import os
import json
import yaml
import tensorflow_model_optimization as tfmot
import numpy as np

from dataloaders import DATALOADER_PARAMS, set_loader_params, load_ksdd, load_ksdd2, load_ksdd_tl, load_ksdd2_tl
from models import create_models, create_models_from_config
from eval import compute_metrics

def get_optimizer(learning_rate):
    return Adam(learning_rate)

def compute_loss(seg_model, clf_model, img, mask, segw, lbl, gamma, delta, Lambda=1):
    Sf, Sh = seg_model(img)
    Cp = clf_model(tf.concat([Sf, Sh], axis=3))

    Lseg = tf.reduce_sum(
        segw * tf.nn.sigmoid_cross_entropy_with_logits(logits=Sh, labels=mask),
        axis=[1, 2]
    )
    Lclf = tf.nn.sigmoid_cross_entropy_with_logits(logits=Cp, labels=lbl)

    return tf.reduce_mean(Lambda*gamma*Lseg), tf.reduce_mean((1 - Lambda)*delta*Lclf)

@tf.function(experimental_relax_shapes=True)
def train_step(seg_model, clf_model, img, mask, segw, lbl, gamma, epoch, optimizer):
    with tf.GradientTape() as seg_tape, tf.GradientTape() as clf_tape:
        seg_loss, clf_loss = compute_loss(seg_model, clf_model, img, mask, segw, lbl, gamma, epoch)

    Gseg = seg_tape.gradient(seg_loss, seg_model.trainable_variables)
    Gclf = clf_tape.gradient(clf_loss, clf_model.trainable_variables)

    optimizer.apply_gradients(zip(Gseg, seg_model.trainable_variables))
    optimizer.apply_gradients(zip(Gclf, clf_model.trainable_variables))

def train_loop(train_pos, train_neg_iter, test, seg_model, clf_model, optimizer, epochs, delta, log_interval, test_on_cpu):
    for epoch in range(1, epochs + 1):
        start_time = time()
        Lambda = 1.0 - epoch/epochs
        for img, mask, segw, lbl, gamma in train_pos:
            train_step(seg_model, clf_model, img, mask, segw, lbl, gamma, epoch, optimizer)
            _img, _mask, _segw, _lbl, _gamma = train_neg_iter.get_next()
            train_step(seg_model, clf_model, _img, _mask, _segw, _lbl, _gamma, epoch, optimizer)
        end_time = time()
        print(f'Epoch: {epoch}, Training time: {end_time - start_time}')
        if epoch % log_interval == 0:
            metrics = compute_metrics(test, seg_model, clf_model)
            print(f'Metrics: {metrics}')
        print('-' * 50)
    if test_on_cpu:
        with tf.device("/CPU:0"):
            metrics = compute_metrics(test, seg_model, clf_model)
    return metrics

def cli():
    parser = argparse.ArgumentParser(description='Bonseyes Defect Detection train script')
    parser.add_argument(
        '--dataset-json-path', required=True, type=str,
        help='Path to dataset.json file generated by the Datatool'
    )
    parser.add_argument(
        '--shuffle-buf-size', required=True, type=int,
        help='Buffer size for dataset shuffling'
    )
    parser.add_argument(
        '--batch-size', required=True, type=int,
        help='Batch size'
    )
    parser.add_argument(
        '--optimizer', required=False, type=str,
        help='Optimizer to use for training'
    )
    parser.add_argument(
        '--learning-rate', required=True, type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--learning-decay', required=False, default=False, type=bool,
        help='Whether to use learning rate decay'
    )
    parser.add_argument(
        '--epochs', required=True, type=int,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--delta', required=True, type=float,
        help='Classification loss weight, relative to segmentation loss'
    )
    parser.add_argument(
        '--log-interval', required=True, type=int,
        help='Epoch interval for evaluating model performance during training and printing values of metrics'
    )
    parser.add_argument(
        '--metrics', required=False, nargs='*', type=str, default=[],
        help='List of metrics for model evaluation'
    )
    parser.add_argument(
        '--test-on-cpu', required=False, type=bool, default=False,
        help="Whether to test the model's performance on CPU after training"
    )
    parser.add_argument(
        '--base-path', required=True, type=str,
        help='Path to Datatool output directory'
    )
    parser.add_argument(
        '--tl-num-images', required=False, type=int,
        help='Number of images to use for transfer learning, if specified'
    )
    parser.add_argument(
        '--tl-pretrained-model', required=False, type=str,
        help='Path to a pretrained model when using transfer learning'
    )
    parser.add_argument(
        '--output-path', required=False, type=str,
        help='Path to a directory where the trained model will be saved'
    )
    return parser

def main():
    args = cli().parse_args()

    train_pos, train_neg_iter, test = load_ksdd2(
        args.base_path, args.dataset_json_path
        )

    seg_model = tf.keras.models.load_model(os.path.join(args.tl_pretrained_model, 'seg_model.h5'))
    clf_model = tf.keras.models.load_model(os.path.join(args.tl_pretrained_model, 'clf_model.h5'))
   
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    epochs = args.epochs
    delta = args.delta
    log_interval = args.log_interval
    lr = args.learning_rate
    test_on_cpu = args.test_on_cpu
    end_step = np.ceil(args.tl_num_images / args.batch_size).astype(np.int32) * epochs

    if args.learning_decay and args.optimizer != 'nadam':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_steps=10*246//args.batch_size,
            decay_rate=0.93,
            staircase=True
        )

    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif args.optimizer == 'nadam':
        if args.learning_decay:
            optimizer = tf.keras.optimizers.Nadam(lr, schedule_decay=lr/200.0)
        else:
            optimizer = tf.keras.optimizers.Nadam(lr)
    elif args.optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr)
    else:
        optimizer = get_optimizer(args.learning_rate)

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.99,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    seg_model = prune_low_magnitude(seg_model, **pruning_params)

    seg_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    metrics = train_loop(
        train_pos,
        train_neg_iter,
        test,
        seg_model,
        clf_model,
        optimizer,
        args.epochs,
        args.delta,
        args.log_interval,
        args.test_on_cpu
    )

    model_for_export = tfmot.sparsity.keras.strip_pruning(seg_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    with open(os.path.join(args.output_path, 'seg_model.tflite'), 'wb') as f:
        f.write(pruned_tflite_model)
    clf_model.save(os.path.join(args.output_path, 'clf_model.h5'))

if __name__ == '__main__':
    main()
