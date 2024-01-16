import tensorflow as tf
import time


def compute_metrics(test_set, seg_model, clf_model, curve='PR'):
    pr_auc_clf = tf.keras.metrics.AUC(curve=curve)
    pr_auc_seg = tf.keras.metrics.AUC(curve=curve)
    img_num = 0

    infer_time = 0

    for img, mask, _, lbl in test_set:
        infer_t0 = time.time()
        Sf, Sh = seg_model(img, training=False)
        Cp = clf_model(tf.concat([Sf, Sh], axis=3), training=False)
        infer_time += time.time() - infer_t0

        # post proc
        Sh = tf.sigmoid(Sh)
        Cp = tf.sigmoid(Cp)

        pr_auc_clf.update_state(lbl, Cp)
        pr_auc_seg.update_state(mask, Sh)
        img_num += img.shape[0]

    return {
        'AP_clf': float(pr_auc_clf.result().numpy()),
        'AP_seg': float(pr_auc_seg.result().numpy()),
        'avg_infer_time': infer_time / img_num,
    }


