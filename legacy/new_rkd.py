
import warnings

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ReliableProbModel(nn.Module):
    def __init__(self, device, momentum=0.995, noise_level=1.0):

        super().__init__()

        self.power = 1.0
        self.momentum = momentum
        self.bins_num = 100
        self.device = device

        self.noise_level = noise_level

    def predict_prob(self, min=1e-8, max=1e2):
        return 1 - torch.clamp(self.delta_entropy, min=min, max=max) ** self.power if np.isfinite(self.power) else 0.0

    @staticmethod
    def curve_func(x, a):
        return 1 - x ** a

    def initialization(self, teacher, data):
        self.ready = False
        noise_data = data.clone()
        noise_data.x = noise_data.x + torch.randn_like(noise_data.x, device=noise_data.x.device) * self.noise_level

        logits_teacher = teacher(data.x, data.edge_index)
        entropy_org = Categorical(logits=logits_teacher.softmax(dim=-1).detach()).entropy()
        entropy_noise = Categorical(logits=teacher(noise_data.x, noise_data.edge_index).softmax(dim=-1).detach()).entropy()

        weight = torch.abs(entropy_noise - entropy_org)
        self.delta_entropy = weight / torch.max(weight)
        self.pred_teacher = torch.argmax(logits_teacher, dim=1)
        self.logits_teacher = logits_teacher

    def prob_curve_fit(self, xdata, ydata):
        popt, pcov = curve_fit(self.curve_func, xdata, ydata, p0=[1.0], maxfev=1000)
        return popt[0]

    def updata(self, logits_student):
        pred_teacher = self.pred_teacher.detach()
        try:
            pred_student = torch.argmax(logits_student.detach(), dim=1)
        except:
            return

        matched = pred_teacher == pred_student
        weight_true = self.delta_entropy[matched]
        weight_false = self.delta_entropy[~matched]

        bins = torch.linspace(0, 1, self.bins_num + 1, device='cpu').numpy()
        hist_true = torch.histc(weight_true, bins=self.bins_num, min=0, max=1)
        hist_false = torch.histc(weight_false, bins=self.bins_num, min=0, max=1)


        prob = hist_true / (hist_true + hist_false + 1e-4)
        prob = ((prob - prob.min()) / (prob.max() - prob.min())).cpu().numpy()

        try:
            update_power = self.prob_curve_fit(bins[:-1], prob)
            self.power = (self.momentum) * self.power + (1 - self.momentum) * update_power
            self.ready = True

        except (RuntimeError, ValueError) as e:
            pass