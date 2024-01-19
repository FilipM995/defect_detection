from random import choices
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from time import time
import argparse
import os
import json
import yaml
from datetime import datetime

from dataloaders import (
    DATALOADER_PARAMS,
    set_loader_params,
    load_ksdd2_custom,
    # load_ksdd,
    # load_ksdd2,
    # load_ksdd_tl,
    # load_ksdd2_tl,
    # load_ksdd2_aug,
)
from models import (
    create_models,
    create_models_from_config,
    create_models_from_config_nomaxpool,
)
from eval import compute_metrics


def get_optimizer(learning_rate):
    return tf.keras.optimizers.legacy.Adam(learning_rate)


def compute_loss(
    seg_model, clf_model, img, mask, segw, lbl, gamma, delta, Lambda
):
    Sf, Sh = seg_model(img)
    Cp = clf_model(tf.concat([Sf, Sh], axis=3))

    Lseg = tf.reduce_sum(
        segw * tf.nn.sigmoid_cross_entropy_with_logits(logits=Sh, labels=mask),
        axis=[1, 2],
    )
    Lclf = tf.nn.sigmoid_cross_entropy_with_logits(logits=Cp, labels=lbl)

    return tf.reduce_mean(Lambda * gamma * Lseg), tf.reduce_mean(
        (1 - Lambda) * delta * Lclf
    ), tf.reduce_mean(Lseg), tf.reduce_mean(Lclf)


@tf.function(experimental_relax_shapes=True)
def train_step(
    seg_model, clf_model, optimizer, img, mask, segw, lbl, gamma, delta, Lambda
):
    with tf.GradientTape() as seg_tape, tf.GradientTape() as clf_tape:
        seg_loss, clf_loss,Lseg,Lclf = compute_loss(
            seg_model, clf_model, img, mask, segw, lbl, gamma, delta, Lambda
        )

    seg_grad = seg_tape.gradient(seg_loss, seg_model.trainable_variables)
    clf_grad = clf_tape.gradient(clf_loss, clf_model.trainable_variables)

    optimizer.apply_gradients(zip(seg_grad, seg_model.trainable_variables))
    optimizer.apply_gradients(zip(clf_grad, clf_model.trainable_variables))

    return Lseg,Lclf


def train_loop(
    train_pos,
    train_neg_iter,
    test,
    seg_model,
    clf_model,
    optimizer,
    epochs,
    delta,
    log_interval,
    test_on_cpu,
):
    seg_losses_per_epoch = []
    clf_losses_per_epoch = []

    seg_AP_per_epoch = []
    clf_AP_per_epoch = []

    lseg_list = []
    lclf_list = []

    for epoch in range(1, epochs + 1):
        start_time = time()
        Lambda = 1.0 - epoch / epochs
        avg_clf_loss_epoch = tf.keras.metrics.Mean()
        avg_seg_loss_epoch = tf.keras.metrics.Mean()
        for img, mask, segw, lbl, gamma in train_pos:
            seg_loss, clf_loss = train_step(
                seg_model,
                clf_model,
                optimizer,
                img,
                mask,
                segw,
                lbl,
                gamma,
                delta,
                Lambda,
            )
            avg_seg_loss_epoch.update_state(seg_loss)
            avg_clf_loss_epoch.update_state(clf_loss)
            _img, _mask, _segw, _lbl, _gamma = train_neg_iter.get_next()
            seg_loss, clf_loss = train_step(
                seg_model,
                clf_model,
                optimizer,
                _img,
                _mask,
                _segw,
                _lbl,
                _gamma,
                delta,
                Lambda,
            )
            avg_seg_loss_epoch.update_state(seg_loss)
            avg_clf_loss_epoch.update_state(clf_loss)
        end_time = time()
        print(f'Epoch: {epoch}, Training time: {end_time - start_time}')
        avg_seg_loss_value = float(avg_seg_loss_epoch.result().numpy())
        avg_clf_loss_value = float(avg_clf_loss_epoch.result().numpy())

        seg_losses_per_epoch.append(avg_seg_loss_value)
        clf_losses_per_epoch.append(avg_clf_loss_value)
        
        print(f'Average Segmentation Loss: {avg_seg_loss_value:.2f}')
        print(f'Average Classification Loss: {avg_clf_loss_value:.2f}')
        print(f'Returning seg_loss_weight {Lambda:.2f} and dec_loss_weight {1-Lambda:.2f}')
        if epoch % log_interval == 0:
            metrics = compute_metrics(test, seg_model, clf_model)
            seg_AP_per_epoch.append(metrics['AP_seg'])
            clf_AP_per_epoch.append(metrics['AP_clf'])
            lseg_list.append(metrics['Lseg'])
            lclf_list.append(metrics['Lclf'])
            print(f'Metrics: {metrics}')
        print('-' * 50)
    if test_on_cpu:
        with tf.device("/CPU:0"):
            metrics = compute_metrics(test, seg_model, clf_model)
    else:
        with tf.device("gpu:0"):
            metrics = compute_metrics(test, seg_model, clf_model)
    return metrics, seg_losses_per_epoch, clf_losses_per_epoch, seg_AP_per_epoch, clf_AP_per_epoch, lseg_list, lclf_list


def cli():
    parser = argparse.ArgumentParser(
        description='Bonseyes Defect Detection train script'
    )
    parser.add_argument(
        '--input-channels',
        required=True,
        type=int,
        choices=[1, 3],
        help='Number of channels in the input images',
    )
    parser.add_argument(
        '--base-path',
        required=True,
        type=str,
        help='Path to Datatool output directory',
    )
    parser.add_argument(
        '--dataset-json-path',
        required=True,
        type=str,
        help='Path to dataset.json file generated by the Datatool',
    )
    parser.add_argument(
        '--train-percentage',
        required=True,
        type=float,
        help='Percentage of the dataset used for training',
    )
    parser.add_argument(
        '--height',
        required=True,
        type=int,
        help='Target height for input image resizing',
    )
    parser.add_argument(
        '--width',
        required=True,
        type=int,
        help='Target width for input image resizing',
    )
    parser.add_argument(
        '--dil-ksize',
        required=True,
        type=int,
        help='Kernel size for dilation of segmentation masks when calculating spatial weights for segmentation loss',
    )
    parser.add_argument(
        '--mixed-sup-N',
        required=True,
        type=int,
        help='Mixed supervision parameter - how many defective masks will be used for training',
    )
    parser.add_argument(
        '--dist-trans-w',
        required=True,
        type=float,
        help='Spatial weighting for the segmentation loss, formula w*x^p is applied to the output of the distance transform of segmentation masks',
    )
    parser.add_argument(
        '--dist-trans-p',
        required=True,
        type=float,
        help='Spatial weighting for the segmentation loss, formula w*x^p is applied to the output of the distance transform of segmentation masks',
    )
    parser.add_argument(
        '--shuffle-buf-size',
        required=True,
        type=int,
        help='Buffer size for dataset shuffling',
    )
    parser.add_argument(
        '--batch-size', required=True, type=int, help='Batch size'
    )
    parser.add_argument(
        '--optimizer',
        required=False,
        type=str,
        help='Optimizer to use for training',
    )
    parser.add_argument(
        '--learning-rate', required=True, type=float, help='Learning rate'
    )
    parser.add_argument(
        '--learning-decay',
        required=False,
        default=False,
        type=bool,
        help='Whether to use learning rate decay',
    )
    parser.add_argument(
        '--epochs',
        required=True,
        type=int,
        help='Number of epochs for training',
    )
    parser.add_argument(
        '--delta',
        required=True,
        type=float,
        help='Classification loss weight, relative to segmentation loss',
    )
    parser.add_argument(
        '--log-interval',
        required=True,
        type=int,
        help='Epoch interval for evaluating model performance during training and printing values of metrics',
    )
    parser.add_argument(
        '--metrics',
        required=False,
        nargs='*',
        type=str,
        default=[],
        help='List of metrics for model evaluation',
    )
    parser.add_argument(
        '--test-on-cpu',
        required=False,
        type=bool,
        default=False,
        help="Whether to test the model's performance on CPU after training",
    )
    parser.add_argument(
        '--tl-num-images',
        required=False,
        type=int,
        help='Number of images to use for transfer learning, if specified',
    )
    parser.add_argument(
        '--tl-pretrained-model',
        required=False,
        type=str,
        help='Path to a pretrained model when using transfer learning',
    )
    parser.add_argument(
        '--output-path',
        required=False,
        type=str,
        help='Path to a directory where the trained model will be saved',
    )
    parser.add_argument(
        '--networks-config-path',
        required=False,
        type=str,
        default='',
        help='Path to a config file specifying network architectures',
    )
    parser.add_argument(
        '--aug-dir',
        required=False,
        type=str,
        help='Path to a directory where the augmented images are stored',
    )
    parser.add_argument(
        '--aug-trained-model',
        required=False,
        type=str,
        help='Path to a directory where the trained model is saved',
    )
    return parser


def main():
    
    args = cli().parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_path}/output_{timestamp}_lr{args.learning_rate}_bs{args.batch_size}"
    print(f"Running training with LR={args.learning_rate}, Batch Size={args.batch_size}...")

    set_loader_params(
        height=args.height,
        width=args.width,
        dil_ksize=args.dil_ksize,
        mixed_sup_N=args.mixed_sup_N,
        dist_trans_w=args.dist_trans_w,
        dist_trans_p=args.dist_trans_p,
        shuffle_buf_size=args.shuffle_buf_size,
        batch_size=args.batch_size,
    )

    if args.tl_num_images is not None:
        # *** TRANSFER LEARNING ***
        print(f'- Using transfer learning on {args.tl_num_images} images')
        if 'kolektorsdd2' in args.base_path.lower():
            train_pos, train_neg_iter, test = load_ksdd2_tl(
                args.base_path, args.dataset_json_path, args.tl_num_images
            )
        else:
            train_pos, train_neg_iter, test = load_ksdd_tl(
                args.base_path, args.dataset_json_path, args.tl_num_images
            )
    elif args.aug_dir is not None:
        if 'kolektorsdd2' in args.base_path.lower():
            train_pos, train_neg_iter, test = load_ksdd2_aug(
                args.base_path, args.dataset_json_path, args.aug_dir
            )
    else:
        if 'ksdd2' in args.base_path.lower():
            train_pos, train_neg_iter, test = load_ksdd2_custom(
                args.base_path, args.dataset_json_path, args.train_percentage
            )
        else:
            train_pos, train_neg_iter, test = load_ksdd(
                args.base_path, args.dataset_json_path, args.train_percentage
            )

    if args.tl_pretrained_model is not None:
        # *** TRANSFER LEARNING ***
        print(
            f'- Using pretrained model {args.tl_pretrained_model} for transfer learning'
        )
        seg_model = tf.keras.models.load_model(
            os.path.join(args.tl_pretrained_model, 'seg_model.h5')
        )
        clf_model = tf.keras.models.load_model(
            os.path.join(args.tl_pretrained_model, 'clf_model.h5')
        )
    elif args.aug_trained_model is not None:
        seg_model = tf.keras.models.load_model(
            os.path.join(args.aug_trained_model, 'seg_model.h5')
        )
        clf_model = tf.keras.models.load_model(
            os.path.join(args.aug_trained_model, 'clf_model.h5')
        )
    elif args.networks_config_path != '':
        with open(args.networks_config_path) as f:
            networks_config = yaml.load(f, Loader=yaml.FullLoader)
        seg_model, clf_model = create_models_from_config(
            networks_config, input_channels=args.input_channels
        )
    else:
        seg_model, clf_model = create_models(
            input_channels=args.input_channels
        )

    seg_model.summary()
    clf_model.summary()

    lr = args.learning_rate
    if args.learning_decay and args.optimizer != 'nadam':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_steps=10 * 246 // args.batch_size,
            decay_rate=0.93,
            staircase=True,
        )

    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(lr)
    elif args.optimizer == 'nadam':
        if args.learning_decay:
            optimizer = tf.keras.optimizers.legacy.Nadam(
                lr, schedule_decay=lr / 200.0
            )
        else:
            optimizer = tf.keras.optimizers.legacy.Nadam(lr)
    elif args.optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.legacy.RMSprop(lr)
    else:
        optimizer = get_optimizer(args.learning_rate)

    metrics, seg_losses_per_epoch, clf_losses_per_epoch, seg_AP_per_epoch, clf_AP_per_epoch, lseg_list, lclf_list = train_loop(
        train_pos,
        train_neg_iter,
        test,
        seg_model,
        clf_model,
        optimizer,
        args.epochs,
        args.delta,
        args.log_interval,
        args.test_on_cpu,
    )

    seg_model.save(os.path.join(output_path, 'seg_model.h5'))
    clf_model.save(os.path.join(output_path, 'clf_model.h5'))

    data={"learning_rate": args.learning_rate, 
    "batch_size": args.batch_size,
    "N_mix_sup": args.mixed_sup_N,
    "seg_losses" :seg_losses_per_epoch, 
    "clf_losses" :clf_losses_per_epoch, 
    "seg_AP_list" :seg_AP_per_epoch, 
    "clf_AP_list" :clf_AP_per_epoch,
    "Lseg_list" :lseg_list, 
    "Lclf_list" :lclf_list,
    }
    metrics.update(data)

    if args.output_path is not None:
        with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
