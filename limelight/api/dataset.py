
__all__ = ["load_dataset"]

from limelight.dataset import *

func_map = {
    "cora": "planetoid",
    "citeseer": "planetoid",
    "pubmed": "planetoid",
    "photo": "amazon",
    "computers": "amazon",
    "cs": "coauthor",
    "physics": "coauthor",
    "amazon-ratings": "hetero",
    "roman-empire": "hetero",
    "minesweeper": "hetero",
    "tolokers": "hetero",
    "questions": "hetero",
    "wikics": "wikics",
}

def load_dataset(dataset_name, split_method, missing_type="uniform", missing_rate=0.9, split=None, **kwargs):
    data_kwargs = {}
    name = dataset_name.lower()
    func_name = func_map[name]

    if split is not None:
        data_kwargs["split"] = split
    data = DatasetWrapper(data_loader(func_name, name, **data_kwargs))
    data = data.split(split_method, **kwargs)
    data = data.apply_missing(missing_rate, missing_type)
    if not hasattr(data.dataset, "num_classes"):
        data.dataset.num_classes = data.dataset.y.max().item() + 1
    return data.dataset, data.missing_mask

if __name__ == "__main__":
    dataset, mask = load_dataset("wikics", "fixed")
    print(dataset)
