
__all__ = ["filling_data"]

from limelight.imputer import *

func_map = {
    "fixed": FixValueFilling,
    "zero": ZeroFilling,
    "random": RandomFilling,
    "mean": MeanFilling,
    "nn-mean": NeighborhoodMeanFilling,
    "fp": FeaturePropagationFilling,
    "apcfi": APCFIFilling,
}

def filling_data(data, method:str, _print=True, **kwargs):
    if method.lower() not in func_map:
        raise ValueError(f"Method {method} not recognized. Available methods: {list(func_map.keys())}")

    if _print:
        print(f"Filling data with {method} method.")

    impute = func_map[method.lower()](**kwargs)
    return impute(data)




if __name__ == "__main__":
    from limelight.api.dataset import load_dataset

    dataset, mask = load_dataset("pubmed", "random", missing_rate=0.5)
    print(dataset)
    print(filling_data(dataset, "fp").x)
