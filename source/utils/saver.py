import os
import torch
import numpy as np


class BaseSaver:
    """Base saver class."""

    def __init__(self, save_path, logger):
        """Initialize base saver.

        Args:
            save_path: path to save the model
            logger: logger instance
        """
        self.save_path = save_path
        self.logger = logger
        self.early_stop = False
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_ent = np.inf
        self.best_siment = 0.0
        self.best_im = 0.0

    def __call__(self, epoch, model, summary):
        if "val_loss" in summary.keys():
            if summary["val_loss"] < self.best_loss:
                self.logger.info(
                    f'Val. loss decreased ({self.best_loss:.6f} --> {summary["val_loss"]:.6f}). Saving model.'
                )
                torch.save(model.state_dict(), os.path.join(self.save_path, "model_best_loss.pt"))
                self.best_loss = summary["val_loss"]
        if "val_dice" in summary.keys():
            if summary["val_dice"] > self.best_dice:
                self.logger.info(
                    f'Val. dice score increased ({self.best_dice:.6f} --> {summary["val_dice"]:.6f}). Saving model.'
                )
                torch.save(model.state_dict(), os.path.join(self.save_path, "model_best_dice.pt"))
                self.best_dice = summary["val_dice"]


class EarlyStoppingSaver:
    """Early stopping."""

    def __init__(self, logger, patience=10, stop_epoch=50, verbose=True, metric="loss", save_path=None):
        """Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            logger: logger instance
            patience: how many epochs to wait after last validation loss improvement. Defaults to 10.
            stop_epoch: earliest epoch for stopping. Defaults to 50.
            verbose: verbose. Defaults to True.
            metric: validation metric. Defaults to 'loss'.
            save_path: where to save the checkpoint. Defaults to None.
        """
        self.logger = logger
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_type = metric
        self.save_path = save_path

    def __call__(self, epoch, model, summary):
        if self.metric_type == "loss":
            score = -summary["val_loss"]
        elif self.metric_type == "dice":
            score = summary["val_dice"]
        else:
            raise NotImplementedError

        if self.best_score is None:
            self.save_checkpoint(score, model, self.save_path)
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, self.save_path)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_score, model, save_path):
        """Saves model when validation score improves."""
        if self.verbose and self.best_score is not None:
            self.logger.info(
                f"Val. {self.metric_type} improved ({self.best_score:.6f} --> {val_score:.6f}). Saving model."
            )
        torch.save(model.state_dict(), os.path.join(save_path, f"model_best_{self.metric_type}.pt"))
