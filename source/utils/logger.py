import torch
import numpy as np


def get_confusion_matrix(y_pred, y_true, nr_classes):
    """Faster computation of confusion matrix.

    Args:
        y_pred: ground truth (1D)
        y_true: prediction (1D)
        nr_classes: number of classes
    """
    assert y_true.shape == y_pred.shape
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    y = nr_classes * y_true + y_pred

    y = torch.bincount(y)
    if len(y) < nr_classes * nr_classes:
        y = torch.cat((y, torch.zeros(nr_classes * nr_classes - len(y), dtype=torch.long)))
    y = y.reshape(nr_classes, nr_classes)

    return y.numpy()


class DiceLogger(object):
    """Dice score logger."""

    def __init__(self, n_classes, background_label):
        """Initialize dice score logger.

        Args:
            n_classes: number of classes
            background_label: label of background
        """
        super(DiceLogger, self).__init__()
        self.smooth = 1e-12
        self.n_classes = n_classes
        self.background_label = background_label
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def log(self, y_hat, y_true, mask=None):
        if mask is not None:
            y_true[~mask] = self.background_label
            mask_gt = y_true != self.background_label
            mask_pred = y_hat != self.background_label
            mask = mask_gt & mask_pred

            y_true = y_true[mask]
            y_pred = y_hat[mask]

        self.confusion_matrix += get_confusion_matrix(y_true=y_true, y_pred=y_pred, nr_classes=self.n_classes)

    def get_dice_score(self):
        assert np.sum(self.confusion_matrix) > 0, "Invalid confusion matrix"
        confusion_matrix = self.confusion_matrix.T
        scores = np.empty(self.n_classes)
        indices = np.arange(self.n_classes)
        for i in range(self.n_classes):
            TP = confusion_matrix[i, i]
            index = np.zeros_like(confusion_matrix, dtype=bool)
            index[indices == i, :] = True
            index[i, i] = False
            FP = confusion_matrix[index.astype(bool)].sum()
            index = np.zeros_like(confusion_matrix, dtype=bool)
            index[:, indices == i] = True
            index[i, i] = False
            FN = confusion_matrix[index.astype(bool)].sum()
            recall = TP / (FN + TP + self.smooth)
            precision = TP / (TP + FP + self.smooth)
            scores[i] = 2 * 1 / (1 / (recall + self.smooth) + 1 / (precision + self.smooth) + self.smooth)

        mean_dice_score = np.mean(scores[1:]) if self.n_classes > 2 else scores[1]

        return scores, mean_dice_score
