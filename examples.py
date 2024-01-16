from dataloaders import load_ksdd2_custom as load_dataset
from models import create_models
from train import train_loop, get_optimizer
from eval import compute_metrics

train_pos, train_neg_iter, test = load_dataset(
    '..\\..\\..\\..\\shared\\out',
    '..\\..\\..\\..\\shared\\out\\default\\dataset.json',
)

# for img, mask, segw, lbl, gamma in train_pos:
#     print(img, mask, lbl)

seg_model, clf_model = create_models()

metrics = compute_metrics(test)
fpr, tpr, auc = metrics.roc_auc(seg_model, clf_model)
print(auc)

train_loop(
    train_pos,
    train_neg_iter,
    test,
    seg_model,
    clf_model,
    get_optimizer(0.0001),
    50,
    1.0,
    5,
)
