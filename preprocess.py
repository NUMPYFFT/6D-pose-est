import config
import utils

if __name__ == "__main__":
    args = config.get_config()
    utils.preprocess_dataset("train", args)
    utils.preprocess_dataset("val", args)