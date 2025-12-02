
import torch
import torch.nn.functional as F

from new_rkd import ReliableProbModel

def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == label).sum().item()
    acc = correct / len(label)
    return acc

class baseModule:
    input_missing = False
    def __init__(self, num_classes, split_idx, device='cuda', logger=None, max_patience=float('inf')):
        super(baseModule, self).__init__()
        self.max_patience = max_patience
        self.device = device
        self.reset_patience()

        self.logger = logger

        self.split_idx = split_idx
        self.train_mask = split_idx['train_mask']
        self.val_mask = split_idx['val_mask']
        self.test_mask = split_idx['test_mask']

        self.num_classes = num_classes
        self.eval_func = accuracy


    def fit(self, data, missing_mask, run_id, epoch=None):
        self.data = data.to(self.device)
        self.model = self.model.to(self.device)

        self.optimizer = self.configure_optimizers()
        self.best_valid_acc = 0
        self.best_test_acc = 0

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



    @torch.no_grad()
    def test_step(self, data, _print=True):
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()

        logits = self.model(data.x, data.edge_index)
        result = self.eval_check(logits, data, data.name)
        if _print:
            print(f'\nTest step:\nvalLoss: {result[-1]:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, ')
        return logits

    def test(self, data, _print=True):
        return self.test_step(data, _print=_print)

    def _cls_loss(self, out, y, split, name, label_smoothing=0.0):
        if name in ("questions"):
            true_label = F.one_hot(y, y.max() + 1).squeeze(1) if y.shape[1] == 1 else y

            loss = F.binary_cross_entropy_with_logits(out[split],
                                                      true_label.squeeze(1)[split].to(torch.float))

        else:
            loss = F.cross_entropy(out[split], y.squeeze(1)[split],
                                   label_smoothing=label_smoothing)

        return loss

    @torch.no_grad()
    def eval_check(self, out, batch, name):
        y = batch.y
        train_acc = self.eval_func(y[self.train_mask], out[self.train_mask])

        valid_acc = self.eval_func(y[self.val_mask], out[self.val_mask])

        test_acc = self.eval_func(y[self.test_mask], out[self.test_mask])

        val_loss = self.criterion(out, batch, self.val_mask, name).item()
        return train_acc, valid_acc, test_acc, val_loss

    def criterion(self, out, batch, split, name, label_smoothing=0.0):
        raise NotImplementedError

    def reset_patience(self):
        self.patience = 0



class Stuednt_Trainer(baseModule):
    #input_missing = True
    def __init__(self, model, lr, split_idx, device, logger):
        super(Stuednt_Trainer, self).__init__(split_idx, device, logger, float("inf"))
        self.tau = 0.8
        self.lamb = 0.4

        self.lr = lr
        self.model = model
        self.RKD = ReliableProbModel(device)
        self.RKD_ready = False

        self.kd_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def initialization(self, teacher, data):
        self.RKD.initialization(teacher, data)
        self.logits_teacher = self.RKD.logits_teacher.to(self.device).detach()

    def updata_RKD(self, logits_student):
        self.RKD.updata(logits_student)
        self.RKD_ready = True

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def criterion(self, out, batch, split, name, label_smoothing=0.15, log=False):
        loss_cls = self._cls_loss(out, batch.y, split, name, label_smoothing)
        loss_kd = self._kd_loss(out, batch)
        loss = self.lamb * loss_cls  + (1 - self.lamb) * loss_kd
        if log:
            self.loss_cls = loss_cls.item()
            self.loss_kd = loss_kd.item()

        return loss

    def training_step(self, batch, updata=False, batch_idx=None):
        if updata:
            self.updata_RKD(self.out)
        out = self.model(batch.x, batch.edge_index)
        loss = self.criterion(out, batch, self.train_mask, batch.name, 0.0, True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        self.out = self.model(batch.x, batch.edge_index)
        result = self.eval_check(self.out, batch, batch.name)
        return result

    def RKD_sampler(self, data):
        reliable_prob = self.RKD.predict_prob()
        sampling_mask = torch.bernoulli(reliable_prob[data.edge_index[1].cpu()]).bool()
        data.check_list = data.edge_index[:, sampling_mask]
        return data

    def fit_flow(self, epoch, *args):
        self.model.train()
        self.optimizer.zero_grad()

        data = self.data
        data.edge_index = remove_self_loops(data.edge_index)[0]
        if self.RKD_ready:
            data = self.RKD_sampler(data)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            train_loss = self.training_step(data, epoch >= 50) #epoch >= 50

        train_loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            result = self.validation_step(data)

        if result[1] > self.best_valid_acc:
            self.best_valid_acc = result[1]
            self.best_test_acc = result[2]
            self.best_epoch = epoch
            self.best_state_dict = self.model.state_dict()
            self.reset_patience()

        if epoch % self.args.display_step == 0 or self.patience == 0:
            print(f'Epoch: {epoch:02d}, '
                  # f"power: {self.RKD.power:.3f}, "
                  f'Loss: {train_loss:.3f}, '
                  # f'clsLoss: {self.loss_cls:.3f}, '
                  f'kdLoss: {self.loss_kd:.3f}, '
                  f'valLoss: {result[-1]:.3f}, '
                  f'Train: {100 * result[0]:.1f}%, '
                  f'Val: {100 * result[1]:.1f}%, '
                  f'Test: {100 * result[2]:.1f}%, '
                  f'Best Val: {100 * self.best_valid_acc:.1f}%, '
                  f'Best Test: {100 * self.best_test_acc:.1f}%', end='\r')

        return result

    def fit(self, data, missing_mask, run_id):
        #from src.Dataset import filling
        #data = filling(data, missing_mask, "feature_propagation")
        epochs = int(self.args.epochs * 1.5)
        super().fit(data, missing_mask, run_id, epochs)

    def _kd_loss(self, out_student, data):
        check_list = add_self_loops(data.check_list)[0] if hasattr(data, "check_list") else data.edge_index

        loss = self.kd_loss(F.log_softmax(out_student[check_list[0]], dim=1),
                            F.log_softmax(self.logits_teacher[check_list[1]], dim=1))
        return loss