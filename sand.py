
if __name__ == "__main__":
    from limelight.sandbox import sandbox

    args = {"model_type": "mpnn", "conv_type": "GCNConv",
            "hidden_channels": 512, "num_layers": 2, #6
            "heads": 4, "K": 1, "epochs": 1000,
            "res": False, "jk": False, "pre_linear": False,
            "norm_type": "identity", #identity, layer
            "act_type": "relu",
            "dropout": 0.5, #0.5
            "missing_type": "uniform", #uniform

            "max_patience": 1000,
            "lr": 0.0000,
            "weight_decay": 5e-4 ,

            "seed": 42,
            "dataset_name": "computers", "split_method": "fixed",

            "missing_rate": 0.9,
            "filling_method": "apcfi",

            "metric": "acc",
            }
    sandbox(args)