import os

RANDOM_STATE = 42
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.25
TRAIN_PATH = "/Users/shakedcaspi/Documents/tau/survival_analysis_ml/project/datasets/train"
TEST_PATH = "/Users/shakedcaspi/Documents/tau/survival_analysis_ml/project/datasets/test"
FULL_DATASETS_PATH = "/Users/shakedcaspi/Documents/tau/survival_analysis_ml/project/datasets/full_datasets"
# add later logger using decorator !!
# LOGGER = logging.getLogger("SAExper")
# LOGGER.setLevel(logging.INFO)
TAB = "------- "


def log(x, num_tabs=0):
    print(TAB * num_tabs, x, sep='')
