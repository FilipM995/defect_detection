import tensorflow as tf
import time

def compute_metrics(test_set, seg_model, clf_model, curve='PR'):
    pr_auc_clf = tf.keras.metrics.AUC(curve=curve)
    pr_auc_seg = tf.keras.metrics.AUC(curve=curve)
    lseg_avg=tf.keras.metrics.Mean()
    lclf_avg=tf.keras.metrics.Mean()
    img_num = 0

    infer_time = 0

    for img, mask, segw, lbl in test_set:
        infer_t0 = time.time()
        Sf, Sh = seg_model(img, training=False)
        Cp = clf_model(tf.concat([Sf, Sh], axis=3), training=False)
        infer_time += time.time() - infer_t0

        Lseg = tf.reduce_sum(
            segw * tf.nn.sigmoid_cross_entropy_with_logits(logits=Sh, labels=mask),
            axis=[1, 2],
        )
        Lseg = tf.reduce_mean(Lseg)
        Lclf = tf.nn.sigmoid_cross_entropy_with_logits(logits=Cp, labels=lbl)
        Lclf = tf.reduce_mean(Lclf)

        # post proc
        Sh = tf.sigmoid(Sh)
        Cp = tf.sigmoid(Cp)


        lseg_avg.update_state(Lseg)
        lclf_avg.update_state(Lclf)

        pr_auc_clf.update_state(lbl, Cp)
        pr_auc_seg.update_state(mask, Sh)
        img_num += img.shape[0]

    return {
        'AP_clf': float(pr_auc_clf.result().numpy()),
        'AP_seg': float(pr_auc_seg.result().numpy()),
        'Lseg': float(lseg_avg.result().numpy()),
        'Lclf': float(lclf_avg.result().numpy()),
        'avg_infer_time': infer_time / img_num,
    }


