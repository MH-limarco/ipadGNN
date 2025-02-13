import torch

class BaseFilling:
    def forward(self, dataset, missing_mask):
        filled_x = self.fill(dataset)
        dataset.x = torch.where(missing_mask.to(dataset.x.device), dataset.x, filled_x)
        return dataset

    def __call__(self, dataset, missing_mask, **kwargs):
        return self.forward(dataset, missing_mask)

    @staticmethod
    def fill(dataset):
        """ return: feature matrix with filled missing values """
        raise NotImplementedError
