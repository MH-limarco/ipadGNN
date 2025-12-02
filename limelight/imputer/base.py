import torch

class BaseFilling:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, dataset, **kwargs):
        _org_device = dataset.x.device

        filled_x = self.fill(dataset).to(_org_device)
        dataset = dataset.to(_org_device)

        dataset.x = torch.where(dataset.x.isnan(), filled_x, dataset.x)
        #dataset.x = filled_x
        return dataset

    @staticmethod
    def fill(dataset):
        """ return: feature matrix with filled missing values """
        raise NotImplementedError

    def __call__(self, dataset, **kwargs):
        return self.forward(dataset, **kwargs)
