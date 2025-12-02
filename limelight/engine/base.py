
import os
import atexit

import torch.nn.functional as F
import torch
import numpy as np

from limelight.dataset.split_manager import TRAIN_MASK, VAL_MASK, TEST_MASK, LABEL
from limelight.engine.eval import EvalManager
from limelight.core import LoggerManager
from limelight.utils import import_rich, import_plotext

Live, Table, _, Layout, Text, RICH = import_rich()
plt, PLOTEXT = import_plotext()

logger = LoggerManager.get_logger(rich_stout=True if RICH else False)
COLUMN = ["Epoch", "Best Epoch",
          "Loss", "ValLoss",
          "Train", "Valid", "Test",
          "Best Valid", "Best Test"]

plt_size = (50, 13)
class LossCurve:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.colors = "green"
        self.theme = "clear"

    def update(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss.item())

        plt.title("Val Loss Curve")
        plt.xlabel("Epoch")
        #plt.ylabel("loss")
        plt.theme("clear")
        plt.ticks_color('green')

        plt.plot(self.epochs, self.losses, color=self.colors)
        plt.xfrequency(len(self.epochs) + 1)
        plt.xlim(min(self.epochs)-1,
                 max(self.epochs))
        min_loss, max_loss = min(self.losses), max(self.losses)
        best_epoch_idx = np.argmin(self.losses)
        plt.vline(self.epochs[best_epoch_idx], color="red")
        plt.ylim(min_loss * 0.9, max_loss * 1.1)
        plt.plotsize(*plt_size)

    def build(self):
        _text = Text(plt.build(), overflow="ignore", no_wrap=False)
        plt.clt()
        plt.clear_data()
        return _text


class RichTableManager:
    def __init__(self, title=""):
        self.title = title
        self.live = Live()
        self.live.start()
        #atexit.register(self._atexit())

    def generate_table(self, *args):
        table = Table(title=self.title)
        for idx, col in enumerate(COLUMN):
            if idx in [0, 1]:
                table.add_column(col, justify="right", style="cyan", no_wrap=True)
            else:
                table.add_column(col, justify="right")

        if len(args) > 0:
            table.add_row(str(args[0]), str(args[1]),
                               *[f"{arg:.3f}" if arg <= 10.0 else f"{arg:.2f}%" for arg in args[2:]])
        return table

    def update(self, *args):
        _table = self.generate_table(*args)
        self.live.update(_table, refresh=True)

    def __del__(self):
        self.live.stop()

    def stop(self):
        self.live.stop()

class BaseModule:
    def __init__(self, model, args, exp_dir):
        self.model = model
        self.exp_dir = exp_dir
        
        self.args = args
        self.num_classes = args["num_classes"]
        self.resumable = args["resumable"]
        self.max_patience = args["max_patience"]
        
        self.display_each_step = args["display_each_step"]
        self.eval_each_step = args["eval_each_step"]
        
        self.epochs = args["epochs"]
        self.amp = args["amp"]
        self.device = args["device"]
        self.metric = args["metric"].lower()

        self.optimizer_func = args["optimizer"]
        self.optimizer_args = {"lr": args["lr"], "weight_decay": args["weight_decay"]}
        self.optimizer = self.configure_optimizers()

        
        self.EvalManager = EvalManager(self.num_classes)
        self.eval_func = getattr(self.EvalManager, f'eval_{self.metric}')

        self.best_val_acc = 0
        self.best_test_acc = 0
        self.patience = 0
        self.start_epoch = 0
        self.best_epoch = 0
        self.loaded = False

    def fit(self, data, fit_epoch=float('inf')):
        data = data.to(self.device)
        self.model = self.model.to(self.device)
        if RICH:
            self.rich_table = RichTableManager()
            self.layout = Layout(name="root")
            self.layout.split_column(Layout(name="upper", size=5),
                                     Layout(name="lower", size=plt_size[1]))
            self.loss_mgr = LossCurve()
            atexit.register(self.rich_table.stop)

        for _epoch in range(self.start_epoch, self.epochs):
            self.patience += 1
            self.epoch = _epoch + 1

            train_loss = self._train_flow(data, self.epoch)
            if self.epoch % self.eval_each_step == 0:
                val_loss, train_eval, val_eval, test_eval = self._test_flow(data, self.epoch)
                if self.best_val_acc < val_eval:
                    self.best_val_acc = val_eval
                    self.best_test_acc = test_eval
                    self.best_epoch = self.epoch
                    self.patience = 0

                    self.best_model_state = self.model.state_dict()
                    self.best_optimizer_state = self.optimizer.state_dict()
                    if self.resumable:
                        self._save_checkpoint()

                if self.patience == 0 or self.epoch % self.display_each_step == 0:
                    log = f"Epoch: {self.epoch:02d}, " \
                          f"Loss: {train_loss:.3f}, " \
                          f"valLoss: {val_loss:.3f}, " \
                          f"Train: {100 * train_eval:.2f}%, " \
                          f"Valid: {100 * val_eval:.2f}%, " \
                          f"Test: {100 * test_eval:.2f}%, " \

                    logger.info(log)
                    self._save_checkpoint(last=True)

                if hasattr(self, "rich_table"):
                    self._rich_update(train_loss, val_loss, train_eval, val_eval, test_eval)

                if self._is_early_stop(fit_epoch):
                    logger.info(f"Early stop at epoch: {self.epoch}")
                    break

        if hasattr(self, "rich_table"):
            self.rich_table.stop()

    def _rich_update(self, train_loss, val_loss, train_eval, val_eval, test_eval,):
        self.loss_mgr.update(self.epoch, val_loss)

        table = self.rich_table.generate_table(self.epoch, self.best_epoch,
                                               train_loss, val_loss,
                                               train_eval, val_eval, test_eval,
                                               self.best_val_acc, self.best_test_acc)
        plot = self.loss_mgr.build()

        self.layout["upper"].update(table)
        self.layout["lower"].update(plot)
        self.rich_table.live.update(self.layout, refresh=True)

    def _is_early_stop(self, fit_epoch):
        return self.patience >= self.max_patience or self.epoch >= fit_epoch

    def _train_flow(self, data, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.amp):
            _pred = self.train_step(data)
        _loss = self._criterion(_pred, data, TRAIN_MASK)
        _loss.backward()
        self.optimizer.step()
        return _loss.item()

    def _test_flow(self, data, epoch):
        self.model.eval()
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.amp):
            _pred = self.val_step(data)
        return self.eval_check(_pred, data)

    def train_step(self, data):
        return self.model(data)

    def val_step(self, data):
        return self.model(data)

    @torch.no_grad()
    def eval_check(self, out, data):
        y = getattr(data, LABEL)
        train_mask = getattr(data, TRAIN_MASK)
        val_mask = getattr(data, VAL_MASK)
        test_mask = getattr(data, TEST_MASK)

        val_loss = self.criterion(out, y, val_mask)
        train_eval = self.eval_func(out[train_mask], y[train_mask])
        val_eval = self.eval_func(out[val_mask], y[val_mask])
        test_eval = self.eval_func(out[test_mask], y[test_mask])
        return val_loss, train_eval, val_eval, test_eval

    def reset_patience(self):
        self.patience = 0

    def _criterion(self, out, data, mask_name, label_smoothing=0.0):
        return self.criterion(out=out, y=getattr(data, LABEL),
                              mask=getattr(data, mask_name),
                              label_smoothing=label_smoothing)

    def criterion(self, out, y, mask, label_smoothing=0.0):
        if self.num_classes == 1:
            return self._criterion_binary(out, y, mask)
        else:
            return self._criterion_entropy(out, y, mask, label_smoothing)

    def _criterion_binary(self, out, y, mask):
        y = F.one_hot(y, y.max() + 1).squeeze(1) if y.shape[1] == 1 else y
        return F.binary_cross_entropy_with_logits(out[mask], y.squeeze(1)[mask].to(torch.float))

    def _criterion_entropy(self, out, y, mask, label_smoothing=0.0):
        return F.cross_entropy(out[mask], y[mask], label_smoothing=label_smoothing)

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_func)(self.model.parameters(), **self.optimizer_args)

    def _save_checkpoint(self, last=False):
        if not last:
            model_state = self.best_model_state
            optimizer_state = self.best_optimizer_state
        else:
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()

        checkpoint = {"model": model_state,
                      "optimizer": optimizer_state,
                      "best_val_acc": self.best_val_acc,
                      "best_test_acc": self.best_test_acc,
                      "patience": self.patience,
                      "start_epoch": self.epoch if last else self.epochs,
                      "epochs": self.epochs,
                      "best_epoch": self.best_epoch}

        path = os.path.join(self.exp_dir, "weight",
                            f"{'last' if last else 'best'}.pt")

        torch.save(checkpoint, path)
        logger.debug(f"Model saved at: {path}")

    def load(self, last=True):
        path = os.path.join(self.exp_dir, "weight",
                            f"{'last' if last else 'best'}.pt")

        logger.debug(f"Loading model from: {path}")
        checkpoint = torch.load(path, weights_only=True)
        self.best_model_state = checkpoint["model"]
        self.model.load_state_dict(self.best_model_state)
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.best_test_acc = checkpoint["best_test_acc"]
        self.patience = checkpoint["patience"]
        self.start_epoch = checkpoint["start_epoch"]
        self.epochs = checkpoint["epochs"]
        self.best_epoch = checkpoint["best_epoch"]
        self.loaded = True


if __name__ == "__main__":
    em = EvalManager(4)
    print(getattr(em, 'eval_f1')())
