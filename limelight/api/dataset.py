
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
    "ratings": "hetero",
    "roman-empire": "hetero",
    "minesweeper": "hetero",
    "tolokers": "hetero",
    "questions": "hetero",
}

def load_dataset(name, split_method, missing_type="uniform", missing_rate=0.9):
    func_name = func_map[name]
    dataset = DatasetWrapper(data_loader(func_name, name))
    dataset = dataset.split(split_method)
    dataset = dataset.apply_missing(missing_rate, missing_type)
    return dataset.dataset, dataset.missing_mask

if __name__ == "__main__":
    dataset, mask = load_dataset("cora", "random")
