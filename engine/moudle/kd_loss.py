

import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
    """
    https://github.com/zju-SWJ/HWD/blob/main/losses/loss.py
    """

    def forward(self, hid_student, hid_teacher):
        """
        Compute the Contrastive Weight Distillation Loss.

        Args:
            hid_student: Hidden representation of the student model.
            hid_teacher: Hidden representation of the teacher model.

        Returns:
            loss: Computed loss value.
        """

        loss = F.mse_loss(hid_student, hid_teacher.detach())
        return loss


class CWDLoss(nn.Module):
    """
    https://github.com/zju-SWJ/HWD/blob/main/losses/loss.py
    """
    def __init__(self, temperature=1.5, **kwargs):
        super(CWDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, hid_student, hid_teacher):
        """
        Compute the Contrastive Weight Distillation Loss.

        Args:
            hid_student: Hidden representation of the student model.
            hid_teacher: Hidden representation of the teacher model.

        Returns:
            loss: Computed loss value.
        """
        n, dim = hid_student.shape
        hid_student = F.log_softmax(hid_student / self.temperature, dim=-1)
        hid_teacher = F.softmax(hid_teacher / self.temperature, dim=-1)

        loss = F.kl_div(hid_student, hid_teacher.detach(), reduction='sum') * (self.temperature ** 2)

        loss /= n
        return loss

class CCWDLoss(nn.Module):
    """
    https://github.com/zju-SWJ/HWD/blob/main/losses/loss.py
    """
    def __init__(self, temperature=1.5, **kwargs):
        super(CWDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, hid_student, hid_teacher):
        """

        Args:
            hid_student: Hidden representation of the student model.
            hid_teacher: Hidden representation of the teacher model.

        Returns:
            loss
        """
        n, dim = hid_student.shape
        hid_student = F.log_softmax(hid_student / self.temperature, dim=-1)
        hid_teacher = F.softmax(hid_teacher / self.temperature, dim=-1)

        hid_student_T = F.log_softmax(hid_student.T / self.temperature, dim=-1)
        hid_teacher_T = F.softmax(hid_teacher.T / self.temperature, dim=-1)

        loss = F.kl_div(hid_student, hid_teacher.detach(), reduction='sum') * (self.temperature ** 2)
        loss_T = F.kl_div(hid_student_T, hid_teacher_T.detach(), reduction='sum') * (self.temperature ** 2)
        loss = (loss + loss_T) / 2
        loss /= n
        return loss
