
from limelight.imputer import *

func_map = {
    "fixed": FixValueFilling,
    "random": RandomFilling,
    "mean": MeanFilling,
    "nn-mean": NeighborhoodMeanFilling,
    "fp": FeaturePropagationFilling,
}

def filling_data(data, method:str):
    impute = func_map[method.lower()]()
    return impute(data)

if __name__ == "__main__":
    from limelight.api.dataset import load_dataset

    dataset, mask = load_dataset("pubmed", "random", missing_rate=0.5)
    print(dataset)
    print(filling_data(dataset, "fixed").x)
