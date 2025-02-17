import torch.nn.functional as F
import torch

from limelight.engine.eval import EvalManager

class BaseModule:
    def __init__(self, args, max_patience=None):
        self.args = args

        self.max_patience = max_patience if max_patience is not None else float('inf')
        self.device = args.device

        self.num_classes = args.num_classes
        self.metric = args.metric.lower()

        self.EvalManager = EvalManager(self.num_classes)
        self.eval_func = getattr(self.metric, f'eval_{self.metric}')

    def fit(self, data, missing_mask, run_id, epoch=None, best_val_acc=None, best_test_acc=None):
        self.data = data.to(self.device)
        self.model = self.model.to(self.device)

        self.optimizer = self.configure_optimizers()
        self.best_val_acc = best_val_acc
        self.best_test_acc = best_test_acc

        epochs = self.args.epochs if epoch is None else epoch
        for epoch in range(epochs):
            self.patience += 1

            result = self.fit_flow(epoch+1, missing_mask)
            if self.logger is not None:
                self.logger.add_result(run_id, result[:4])

            if self.patience >= self.max_patience:
                break

        if self.logger is not None:
            self.run_id = run_id

    def train_step(self):
        raise NotImplementedError

    def _train_flow(self):
        raise NotImplementedError

    def test_step(self, data, _print=True):
        raise NotImplementedError

    def _test_flow(self):
        raise NotImplementedError

    @torch.no_grad()
    def eval_check(self, out, batch, name):
        y = batch.y
        raise NotImplementedError

    def reset_patience(self):
        self.patience = 0

    def criterion(self, out, batch, split, name, label_smoothing=0.0):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError


if __name__ == "__main__":
    em = EvalManager(4)
    print(getattr(em, 'eval_f1')())
