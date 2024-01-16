import sys
import os

import tensorflow as tf

import tensorflow_addons as tfa

# sys.path.append('/app/bonseyes_dd_ai_asset/data/kolektorsdd')

# from custom_dataset_model import DTDatasetCustom

DATALOADER_PARAMS = {
    'height': 1408,
    'width': 512,
    'resize_factor': 8,
    'dil_ksize': 7,
    'mixed_sup_N': 100,
    'dist_trans_w': 1.0,
    'dist_trans_p': 2.0,
    'shuffle_buf_size': 500,
    'batch_size': 1,
    'to_grayscale': False,
}


def set_loader_params(
    height=1408,
    width=512,
    dil_ksize=7,
    mixed_sup_N=100,
    dist_trans_w=1.0,
    dist_trans_p=2.0,
    shuffle_buf_size=500,
    batch_size=1,
):
    DATALOADER_PARAMS['height'] = height
    DATALOADER_PARAMS['width'] = width
    DATALOADER_PARAMS['dil_ksize'] = dil_ksize
    DATALOADER_PARAMS['dist_trans_w'] = dist_trans_w
    DATALOADER_PARAMS['dist_trans_p'] = dist_trans_p
    DATALOADER_PARAMS['mixed_sup_N'] = mixed_sup_N
    DATALOADER_PARAMS['shuffle_buf_size'] = shuffle_buf_size
    DATALOADER_PARAMS['batch_size'] = batch_size


def parse_element(e):
    img_path = e[0]
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    # print(img.shape)
    if DATALOADER_PARAMS['to_grayscale']:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(
        img, (DATALOADER_PARAMS['height'], DATALOADER_PARAMS['width'])
    )

    mask_path = e[1]
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
    mask = tf.image.resize(
        mask, (DATALOADER_PARAMS['height'], DATALOADER_PARAMS['width'])
    )
    mask = tf.where(mask > 0.0, x=1.0, y=0.0)
    lbl = tf.reshape(tf.reduce_max(mask), shape=(1,))

    dil = tf.nn.dilation2d(
        tf.reshape(mask, (1, tf.shape(mask)[0], tf.shape(mask)[1], 1)),
        filters=tf.zeros(
            (DATALOADER_PARAMS['dil_ksize'], DATALOADER_PARAMS['dil_ksize'], 1)
        ),
        strides=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        padding='SAME',
        data_format='NHWC',
    )

    def segw_pos():
        edt = tfa.image.euclidean_dist_transform(
            tf.image.convert_image_dtype(dil[0, :, :, :], dtype=tf.uint8)
        )
        edt = edt / tf.reduce_max(edt)
        edt = edt ** DATALOADER_PARAMS['dist_trans_p']
        edt = edt * DATALOADER_PARAMS['dist_trans_w']
        edt = tf.where(edt == 0.0, x=1.0, y=edt)
        return edt

    def segw_neg():
        return mask + 1.0

    segw = tf.cond(
        tf.greater(tf.reduce_max(mask), 0.0),
        true_fn=segw_pos,
        false_fn=segw_neg,
    )

    mask = tf.image.resize(
        mask, (tf.shape(mask)[0] // 8, tf.shape(mask)[1] // 8)
    )
    segw = tf.image.resize(
        segw, (tf.shape(segw)[0] // 8, tf.shape(segw)[1] // 8)
    )

    # print(img.shape)
    return img, mask, segw, lbl


_n = tf.Variable(0, dtype=tf.int32)
_N = tf.Variable(DATALOADER_PARAMS['mixed_sup_N'], dtype=tf.int32)


def take_N(img, mask, segw, lbl, n, N):
    def _true_fn():
        n.assign_add(1)
        return 1.0

    def _false_fn():
        return 0.0

    gamma = tf.cond(tf.less(n, N), true_fn=_true_fn, false_fn=_false_fn)
    return img, mask, segw, lbl, gamma


# def load_ksdd(base_path, dataset_json_path, train_percentage=0.8):
#     dtd = DTDatasetCustom(name='input_dataset', operatingMode='memory')
#     dtd.load_from_json(dataset_json_path, element_list=['imagesWithMasks'])

#     dataset = tf.data.Dataset.from_tensor_slices(
#         [
#             (
#                 os.path.join(base_path, val.imgPath),
#                 os.path.join(base_path, val.maskPath),
#             )
#             for key, val in dtd.imagesWithMasks.items()
#         ]
#     )

#     train_len = int(len(dtd.imagesWithMasks) * train_percentage)
#     train = dataset.take(train_len)
#     train = train.map(parse_element)

#     train_neg = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 0.0
#     )
#     train_neg = train_neg.map(
#         lambda img, mask, segw, lbl: (img, mask, segw, lbl, 1.0)
#     )
#     train_neg = (
#         train_neg.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .repeat()
#         .prefetch(1)
#     )
#     train_neg_iter = iter(train_neg)

#     train_pos = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 1.0
#     )
#     train_pos = train_pos.map(
#         lambda img, mask, segw, lbl: take_N(img, mask, segw, lbl, _n, _N)
#     )
#     train_pos = (
#         train_pos.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .prefetch(1)
#     )

#     test = dataset.skip(train_len)
#     test = test.map(parse_element)
#     test = test.batch(DATALOADER_PARAMS['batch_size']).cache().prefetch(1)

#     return train_pos, train_neg_iter, test


# def load_ksdd2(base_path, dataset_json_path, train_percentage=1.0):
#     dtd = DTDatasetCustom(name='input_dataset', operatingMode='memory')
#     dtd.load_from_json(dataset_json_path, element_list=['imagesWithMasks'])

#     paths = [
#         (
#             os.path.join(base_path, val.imgPath),
#             os.path.join(base_path, val.maskPath),
#         )
#         for key, val in dtd.imagesWithMasks.items()
#     ]

#     train_paths = [
#         (img_path, mask_path)
#         for img_path, mask_path in paths
#         if 'train' in img_path.lower()
#     ]
#     test_paths = [
#         (img_path, mask_path)
#         for img_path, mask_path in paths
#         if 'test' in img_path.lower()
#     ]

#     train = tf.data.Dataset.from_tensor_slices(
#         train_paths[: int(len(train_paths) * train_percentage)]
#     )
#     test = tf.data.Dataset.from_tensor_slices(test_paths)

#     train = train.map(parse_element)
#     train_neg = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 0.0
#     )
#     train_neg = train_neg.map(
#         lambda img, mask, segw, lbl: (img, mask, segw, lbl, 1.0)
#     )
#     train_neg = (
#         train_neg.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .repeat()
#         .prefetch(1)
#     )
#     train_neg_iter = iter(train_neg)
#     train_pos = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 1.0
#     )
#     train_pos = train_pos.map(
#         lambda img, mask, segw, lbl: take_N(img, mask, segw, lbl, _n, _N)
#     )
#     train_pos = (
#         train_pos.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .prefetch(1)
#     )

#     test = test.map(parse_element)
#     test = test.batch(1).cache().prefetch(1)

#     return train_pos, train_neg_iter, test


# def load_ksdd_tl(base_path, dataset_json_path, num_images):
#     dtd = DTDatasetCustom(name='input_dataset', operatingMode='memory')
#     dtd.load_from_json(dataset_json_path, element_list=['imagesWithMasks'])

#     dataset = tf.data.Dataset.from_tensor_slices(
#         [
#             (
#                 os.path.join(base_path, val.imgPath),
#                 os.path.join(base_path, val.maskPath),
#             )
#             for key, val in dtd.imagesWithMasks.items()
#         ]
#     )

#     train = dataset.take(num_images)
#     train = train.map(parse_element)

#     train_neg = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 0.0
#     )
#     train_neg = train_neg.map(
#         lambda img, mask, segw, lbl: (img, mask, segw, lbl, 1.0)
#     )
#     train_neg = (
#         train_neg.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .repeat()
#         .prefetch(1)
#     )
#     train_neg_iter = iter(train_neg)

#     train_pos = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 1.0
#     )
#     train_pos = train_pos.map(
#         lambda img, mask, segw, lbl: take_N(img, mask, segw, lbl, _n, _N)
#     )
#     train_pos = (
#         train_pos.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .prefetch(1)
#     )

#     test = dataset.skip(num_images)
#     test = test.map(parse_element)
#     test = test.batch(DATALOADER_PARAMS['batch_size']).cache().prefetch(1)

#     return train_pos, train_neg_iter, test


# def load_ksdd2_tl(base_path, dataset_json_path, num_images):
#     dtd = DTDatasetCustom(name='input_dataset', operatingMode='memory')
#     dtd.load_from_json(dataset_json_path, element_list=['imagesWithMasks'])

#     paths = [
#         (
#             os.path.join(base_path, val.imgPath),
#             os.path.join(base_path, val.maskPath),
#         )
#         for key, val in dtd.imagesWithMasks.items()
#     ]

#     train_paths = [
#         (img_path, mask_path)
#         for img_path, mask_path in paths
#         if 'train' in img_path.lower()
#     ]
#     test_paths = [
#         (img_path, mask_path)
#         for img_path, mask_path in paths
#         if 'test' in img_path.lower()
#     ]

#     train = tf.data.Dataset.from_tensor_slices(train_paths[:num_images])
#     test = tf.data.Dataset.from_tensor_slices(test_paths)

#     DATALOADER_PARAMS['to_grayscale'] = True

#     train = train.map(parse_element)
#     train_neg = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 0.0
#     )
#     train_neg = train_neg.map(
#         lambda img, mask, segw, lbl: (img, mask, segw, lbl, 1.0)
#     )
#     train_neg = (
#         train_neg.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .repeat()
#         .prefetch(1)
#     )
#     train_neg_iter = iter(train_neg)
#     train_pos = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 1.0
#     )
#     train_pos = train_pos.map(
#         lambda img, mask, segw, lbl: take_N(img, mask, segw, lbl, _n, _N)
#     )
#     train_pos = (
#         train_pos.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .prefetch(1)
#     )

#     test = test.map(parse_element)
#     test = test.batch(DATALOADER_PARAMS['batch_size']).cache().prefetch(1)

#     return train_pos, train_neg_iter, test


# def load_ksdd2_aug(base_path, dataset_json_path, aug_dir):
#     dtd = DTDatasetCustom(name='input_dataset', operatingMode='memory')
#     dtd.load_from_json(dataset_json_path, element_list=['imagesWithMasks'])

#     paths = [
#         (
#             os.path.join(base_path, val.imgPath),
#             os.path.join(base_path, val.maskPath),
#         )
#         for key, val in dtd.imagesWithMasks.items()
#     ]

#     aug_files = os.listdir(aug_dir)
#     train_paths = [
#         (img_path, mask_path)
#         for img_path, mask_path in paths
#         if 'train' in img_path.lower()
#     ] + [
#         (
#             os.path.join(aug_dir, aug_files[i]),
#             os.path.join(aug_dir, aug_files[i + 1]),
#         )
#         for i in range(0, len(aug_files), 2)
#     ]

#     test_paths = [
#         (img_path, mask_path)
#         for img_path, mask_path in paths
#         if 'test' in img_path.lower()
#     ]

#     train = tf.data.Dataset.from_tensor_slices(train_paths)
#     test = tf.data.Dataset.from_tensor_slices(test_paths)

#     train = train.map(parse_element)
#     train_neg = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 0.0
#     )
#     train_neg = train_neg.map(
#         lambda img, mask, segw, lbl: (img, mask, segw, lbl, 1.0)
#     )
#     train_neg = (
#         train_neg.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .repeat()
#         .prefetch(1)
#     )
#     train_neg_iter = iter(train_neg)
#     train_pos = train.filter(
#         lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 1.0
#     )
#     train_pos = train_pos.map(
#         lambda img, mask, segw, lbl: take_N(img, mask, segw, lbl, _n, _N)
#     )
#     train_pos = (
#         train_pos.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
#         .batch(DATALOADER_PARAMS['batch_size'])
#         .cache()
#         .prefetch(1)
#     )

#     test = test.map(parse_element)
#     test = test.batch(DATALOADER_PARAMS['batch_size']).cache().prefetch(1)

#     return train_pos, train_neg_iter, test


def load_ksdd2_custom(train_folder, test_folder, train_percentage=1.0):
    # dtd = DTDatasetCustom(name='input_dataset', operatingMode='memory')
    # dtd.load_from_json(dataset_json_path, element_list=['imagesWithMasks'])

    # paths = [
    #     (
    #         os.path.join(base_path, val.imgPath),
    #         os.path.join(base_path, val.maskPath)
    #     )
    #     for key, val in dtd.imagesWithMasks.items()
    # ]

    train_files = sorted(os.listdir(train_folder))
    test_files = sorted(os.listdir(test_folder))


    train_paths = [
        (
            f"{train_folder}/{train_files[i]}",
            f"{train_folder}/{train_files[i + 1]}",
        )
        for i in range(0, len(train_files), 2)
    ]
    test_paths = [
        (f"{test_folder}/{test_files[i]}", f"{test_folder}/{test_files[i + 1]}")
        for i in range(0, len(test_files), 2)
    ]

    # paths = [(img_path, img_path.replace('.png', '_GT.png')) for img_path in os.listdir(dataset_path)]

    # train_paths = [
    #     (img_path, mask_path) for img_path, mask_path in paths if 'train' in img_path.lower()
    # ]
    # test_paths = [
    #     (img_path, mask_path) for img_path, mask_path in paths if 'test' in img_path.lower()
    # ]

    train = tf.data.Dataset.from_tensor_slices(
        train_paths[: int(len(train_paths) * train_percentage)]
    )
    test = tf.data.Dataset.from_tensor_slices(test_paths)

    train = train.map(parse_element)
    train_neg = train.filter(
        lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 0.0
    )
    train_neg = train_neg.map(
        lambda img, mask, segw, lbl: (img, mask, segw, lbl, 1.0)
    )
    train_neg = (
        train_neg.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
        .batch(DATALOADER_PARAMS['batch_size'])
        .cache()
        .repeat()
        .prefetch(1)
    )
    train_neg_iter = iter(train_neg)
    train_pos = train.filter(
        lambda img, mask, segw, lbl: tf.reduce_max(lbl) == 1.0
    )
    train_pos = train_pos.map(
        lambda img, mask, segw, lbl: take_N(img, mask, segw, lbl, _n, _N)
    )
    train_pos = (
        train_pos.shuffle(DATALOADER_PARAMS['shuffle_buf_size'])
        .batch(DATALOADER_PARAMS['batch_size'])
        .cache()
        .prefetch(1)
    )

    test = test.map(parse_element)
    test = test.batch(1).cache().prefetch(1)

    return train_pos, train_neg_iter, test
