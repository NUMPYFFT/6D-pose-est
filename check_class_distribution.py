import config
import utils

if __name__ == "__main__":
    args = config.get_config()
    utils.check_class_distribution(args)
