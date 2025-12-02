from limelight.core import parser_wrapper

# ✅ 對外提供 `get_args()`
def get_args(api_args=None, return_dict=True):
    return parser_wrapper.parse(api_args, return_dict)


if __name__ == "__main__":

    from limelight.core import ExperimentManager, LoggerManager
    from limelight.api.dataset import load_dataset
    from limelight.api.dataset import load_dataset
    from limelight.api.imputer import filling_data
    from limelight.api.model import build_model
    from limelight.utils import fix_seed

    SEED = 42
    fix_seed(SEED)

    args = {"model_type": "mpnn", "conv_type": "GCNConv",
                "hidden_channels": 512, "num_layers": 2,
                "heads": 4, "K": 1, "epochs": 2000,
                "res": True, "jk": False, "pre_linear": False,
                "norm_type": "layer",
                "act_type": "relu",
                "dropout": 0.3,
            "missing_type": "structural",

                "max_patience": 1000,

                "seed": 42,
                "dataset_name": "cs", "split_method": "fixed",

                "missing_rate": 0.0,
                "filling_method": "fp",

                "metric": "acc",
                }

    args = get_args(args)

    exp_manager = ExperimentManager(args)
    exp_dir = exp_manager.get_exp_dir()
    LoggerManager.set_save_dir(exp_dir)

    logger = LoggerManager.get_logger(rich_stout=False)
    logger.info("Test Logger Initialized!")
    logger.error("Test Error Message!")
    logger.warning("Test Warning Message!")
    logger.debug("Test Debug Message!")

    dataset, mask = load_dataset(**args)

    if args["missing_rate"] > 0:
        filled_dataset = filling_data(dataset, "fp")
    else:
        filled_dataset = dataset

    in_channels = filled_dataset.num_features
    num_classes = filled_dataset.num_classes
    args.update({"in_channels": in_channels, "num_classes": num_classes})

    from limelight.engine.base import BaseModule
    module = BaseModule(build_model(args), args, exp_dir=exp_dir)

    module.fit(filled_dataset, )
    #logger.info("模擬訓練中斷")
    #import time
    #time.sleep(2)
    #module.load()
    #module.fit(filled_dataset, 200)

