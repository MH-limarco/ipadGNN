import torch

class BaseFilling:
    def forward(self, dataset, **kwargs):
        filled_x = self.fill(dataset)
        dataset.x = torch.where(dataset.x.isnan(), filled_x, dataset.x)
        return dataset

    @staticmethod
    def fill(dataset):
        """ return: feature matrix with filled missing values """
        raise NotImplementedError

    def __call__(self, dataset, **kwargs):
        return self.forward(dataset, **kwargs)
