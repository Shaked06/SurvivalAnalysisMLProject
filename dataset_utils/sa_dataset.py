from .sa_dataset_utils import PYCOX_DATASETS, ASAUR_DATASETS, DATASETS, PYCOX_DATASET_TO_FUNC, DATASET_TARGET_DECODER

from typing import Tuple
import numpy as np
import pandas as pd
from constants import RANDOM_STATE, TEST_SIZE, TRAIN_PATH, TEST_PATH, FULL_DATASETS_PATH, log
from sklearn.model_selection import train_test_split
import os
os.environ['R_HOME'] = '/Users/shakedcaspi/opt/anaconda3/envs/sa-proj/lib/R'


class SADataset(object):
    @classmethod
    def __init__(self, dataset_name: str) -> None:
        assert dataset_name in DATASETS, f"This dataset is not valid, choose one from {DATASETS}"

        self.dataset_name = dataset_name
        self.dataset = None
        self.X = None
        self.y = None

    @staticmethod
    def get_datasets() -> list:
        return DATASETS

    @staticmethod
    def get_downloaded_datasets() -> list:
        return [name.split(".csv")[0] for name in os.listdir(TRAIN_PATH)]

    @classmethod
    def load_dataset(self, train: bool = True) -> pd.DataFrame:

        full_dataset_path = os.path.join(
            FULL_DATASETS_PATH, f"{self.dataset_name}.csv")
        train_dataset_path = os.path.join(
            TRAIN_PATH, f"{self.dataset_name}.csv")
        test_dataset_path = os.path.join(TEST_PATH, f"{self.dataset_name}.csv")

        if self.dataset_name not in self.get_downloaded_datasets():  # if the dataset is not downloaded yet
            log("DOWNLOADING THE DATASET...", 1)

            if self.dataset_name in ASAUR_DATASETS:
                df = self._asaur_get_dataset()
            if self.dataset_name in PYCOX_DATASETS:
                df = self._pycox_get_dataset()

            # rename y columns to standart names
            duration, event = DATASET_TARGET_DECODER[self.dataset_name]
            df.rename(columns={duration: 'duration',
                      event: 'event'}, inplace=True)

            train_df, test_df = train_test_split(
                df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

            df.to_csv(full_dataset_path, index=False)

            train_df.to_csv(train_dataset_path, index=False)
            log(f"TRAIN Dataset {self.dataset_name}, {df.shape[0]} samples", 1)
            test_df.to_csv(test_dataset_path, index=False)
            log(f"TEST Dataset {self.dataset_name}, {df.shape[0]} samples", 1)

        if train:
            df = pd.read_csv(train_dataset_path)
        else:
            df = pd.read_csv(test_dataset_path)

        self.dataset = df
        return df

    @classmethod
    def _asaur_get_dataset(self):
        import rpy2.robjects.packages as rpackages
        import rpy2.robjects as robjects
        from rpy2.robjects.vectors import StrVector
        from rpy2.robjects import pandas2ri
        from rpy2 import rinterface_lib
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        # change the callbacks, all errors inside buf and can be printing if nessecary
        buf = []
        def f(x): buf.append(x)
        rinterface_lib.callbacks.consolewrite_print = f
        rinterface_lib.callbacks.consolewrite_warnerror = f

        packnames = ['asaur']
        names_to_install = [
            x for x in packnames if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))
            log(f"DOWNLOAD {names_to_install} (R PACKAGES)", 1)

        robjects.r("df <- asaur::prostateSurvival")

        with (robjects.default_converter + pandas2ri.converter).context():
            df = robjects.conversion.get_conversion().rpy2py(
                robjects.r["df"])

        # special preprocessing for prostateSurvival dataset
        # df = pd.get_dummies(df, drop_first=False)
        df["grade"] = df["grade"].map({"mode": 1, "poor": 0})
        df["stage"] = df["stage"].map({"T1ab": 1, "T1c": 2, "T2": 3})
        df["ageGroup"] = df["ageGroup"].map(
            {"66-69": 1, "70-74": 2, "75-79": 3, "80+": 4})
        df = df[df["status"] != 2]

        feature_cols = df.drop(["status", "survTime"], axis=1).columns
        df[feature_cols] = df[feature_cols].astype('float32')
        return df

    @classmethod
    def _pycox_get_dataset(self):

        func = PYCOX_DATASET_TO_FUNC[self.dataset_name]

        df = func.read_df()

        # special preprocessing for KKBOX dataset
        if self.dataset_name == "KKBOX":
            # remove olumns with missing values (there is a lot of missing values in each col)
            missing_vals_cols = list(df.columns[df.isna().sum(axis=0) > 0])
            # remove unnecessary columns
            missing_vals_cols += ["msno", "censor_duration"]
            df.drop(missing_vals_cols, axis=1, inplace=True)
            for col in list(df.select_dtypes(include=bool).columns) + list(df.select_dtypes(include=pd.CategoricalDtype).columns):
                df[col] = df[col].astype(np.int64)
        return df

    @classmethod
    def seperate_X_y(self, train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.dataset is None:
            self.load_dataset(train)

        X = self.dataset.drop(["event", "duration"], axis=1)
        y = self.dataset[["event", "duration"]]
        self.X, self.y = X, y

        return X, y
