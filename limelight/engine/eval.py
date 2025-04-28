import torch
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score, multiclass_auroc

class EvalManager:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def _argmax(self, input):
        if input.dim() == 1:
            return input
        elif input.dim() == 2:
            return input.argmax(dim=-1)
        else:
            raise ValueError("input must be 1D or 2D tensor")

    def eval_f1(self, y_pred, y_true, average='micro'):
        y_pred = self._argmax(y_pred)
        return multiclass_f1_score(y_pred, y_true,
                                   num_classes=self.num_classes, average=average).item()

    def eval_acc(self, y_pred, y_true, ):
        y_pred = self._argmax(y_pred)
        corr_count = torch.sum(y_pred == y_true)
        return (corr_count / len(y_pred)).item()

    def eval_rocauc(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=-1)
        return multiclass_auroc(y_pred, y_true.view(-1),
                                num_classes=self.num_classes)  # .item()


if __name__ == "__main__":
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4]])
    em = EvalManager(4)
    print(em.eval_f1(y_true, y_pred, average='micro'))
    print(em.eval_acc(y_true, y_pred))
    print(em.eval_rocauc(y_true, y_pred))
