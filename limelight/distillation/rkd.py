
__all__ = ['ReliableProbModel']
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scipy.optimize import curve_fit

EPS = 1e-4
MAX_FEV = 1000
CURVE_P = [1.0]

class ReliableProbModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.power = 1.0    #args.init_power
        self.momentum = 0.995   #args.momentum
        self.bins_num = 50  #args.bins_num
        self.device = args.device

        self.noise_level = 1.0
        self.delta_entropy = None
        self.pred_teacher = None

    def predict_prob(self):
        return 1 - self.delta_entropy ** self.power

    def initialization(self, teacher, data):
        noise_data = data.clone()
        noise_data.x = noise_data.x + torch.randn_like(noise_data.x, device=noise_data.x.device) * self.noise_level

        logits_teacher = teacher.test(data, False)
        entropy_org = Categorical(logits=logits_teacher.softmax(dim=-1).detach()).entropy()
        entropy_noise = Categorical(logits=teacher.test(noise_data, False).softmax(dim=-1).detach()).entropy()

        weight = torch.abs(entropy_noise - entropy_org)
        self.delta_entropy = weight / torch.max(weight)
        self.pred_teacher = torch.argmax(logits_teacher, dim=1)
        self.logits_teacher = logits_teacher

    def prob_curve_fit(self, xdata, ydata):
        popt, pcov = curve_fit(self.curve_func, xdata, ydata, p0=CURVE_P, maxfev=MAX_FEV)
        return popt[0]

    def updata(self, logits_student):
        pred_teacher = self.pred_teacher.clone().detach()
        pred_student = torch.argmax(logits_student.detach(), dim=1)

        matched = pred_teacher == pred_student
        weight_true = self.delta_entropy[matched]
        weight_false = self.delta_entropy[~matched]

        bins = torch.linspace(0, 1, self.bins_num + 1, device='cpu').numpy()
        hist_true = torch.histc(weight_true, bins=self.bins_num, min=0, max=1)
        hist_false = torch.histc(weight_false, bins=self.bins_num, min=0, max=1)

        prob = hist_true / (hist_true + hist_false + EPS)
        prob = ((prob - prob.min()) / (prob.max() - prob.min() + EPS)).cpu().numpy()

        try:
            update_power = self.prob_curve_fit(bins[:-1] + 0.05, prob)
            self.power = self.momentum * self.power + (1 - self.momentum) * update_power
        except RuntimeError:
            pass

    @staticmethod
    def curve_func(x, a):
        return 1 - x ** a
