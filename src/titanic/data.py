import pathlib
import re
import string

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin

set_config(transform_output="pandas")


class ColumnNotFoundError(Exception):
    def __init__(self, columns):
        super().__init__(f"One or more columns not found: {columns}")


class WithinClassImputer(TransformerMixin, BaseEstimator):
    def __init__(self, target_col, *, missing_values=np.nan, class_cols=None, strategy="mean"):
        self.target_col = target_col
        self.missing_values = missing_values
        self.class_cols = class_cols or []
        self.strategy = strategy

    @staticmethod
    def _impute(x, *, statistics, missing_values):
        if x.isnull().sum() == 0:
            return x

        try:
            result = x.replace(missing_values, statistics[x.name])
        except IndexError:
            result = x.replace(missing_values, statistics)
        return result

    @staticmethod
    def _group_by_class_cols(x, *, class_cols, target_col):
        if len(class_cols) == 0:
            return x[target_col]

        return x.groupby(class_cols)[target_col]

    def fit(self, X, y=None, **fit_params):  # noqa: ARG002
        grouped = self._group_by_class_cols(X, class_cols=self.class_cols, target_col=self.target_col)

        if self.strategy == "mean":
            statistics = grouped.mean()
        elif self.strategy == "median":
            statistics = grouped.median()
        else:
            msg = f"Invalid strategy: {self.strategy}"
            raise ValueError(msg)

        try:
            self.statistics_ = statistics.to_dict()
        except AttributeError:
            self.statistics_ = statistics
        return self

    def transform(self, X):
        grouped = self._group_by_class_cols(X, class_cols=self.class_cols, target_col=self.target_col)
        X = grouped.transform(self._impute, statistics=self.statistics_, missing_values=self.missing_values)
        return np.expand_dims(X, axis=1)

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        return [self.target_col]


def cabin_to_deck(x):
    out = (
        x.apply(lambda s: s[0] if pd.notnull(s) else "M")
            .replace(["A", "B", "C", "T"], "ABC")
            .replace(["D", "E"], "DE")
            .replace(["F", "G"], "FG")
    )
    return out.to_frame(name="Deck")


def calc_family_size(x):
    out = x["SibSp"] + x["Parch"] + 1
    out = pd.cut(out, bins=[0, 1, 4, 6, np.inf], labels=["Alone", "Small", "Medium", "Large"])
    return out.to_frame(name="FamilySizeGroup")


def create_family(x):
    out = (
        x.str.split(", ", expand=True)[0]
            .str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
    )
    return out.to_frame(name="Family")


def create_title(x):
    out = (
        x.str.split(", ", expand=True)[1]
            .str.split(".", expand=True)[0]
            .replace(["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"], "Mrs/Ms/Miss")
            .replace(["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"], "Dr/Military/Noble/Clergy")
    )
    return out.to_frame(name="Title")


def read_raw_data(data_dir: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(data_dir / "raw" / "train.csv")
    X_test = pd.read_csv(data_dir / "raw" / "test.csv")
    X_train = train.drop(columns="Survived")
    y_train = train["Survived"]
    return X_train, X_test, y_train


def fill_with_median(dat: pd.DataFrame, col: str, groups: list[str]) -> pd.DataFrame:
    if col not in dat.columns or any(g not in dat.columns for g in groups):
        raise ColumnNotFoundError([col, *groups]) from None
    medians = dat.loc["train"].groupby(groups)[col].median()
    median_col = dat.join(medians, on=groups, rsuffix="_med")[f"{col}_med"]
    return dat[col].combine_first(median_col)


def fill_missing_data(dat: pd.DataFrame) -> pd.DataFrame:
    dat["Age"] = fill_with_median(dat, "Age", ["Sex", "Pclass"])
    dat["Fare"] = fill_with_median(dat, "Fare", ["Pclass", "SibSp", "Parch"])
    dat["Embarked"] = dat["Embarked"].fillna(dat["Embarked"].value_counts().idxmax())
    dat["Deck"] = dat["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "M")
    dat["Deck"] = dat["Deck"].replace(["A", "B", "C", "T"], "ABC").replace(["D", "E"], "DE").replace(["F", "G"], "FG")
    return dat.drop(columns=["Cabin"])


def write_data_to_interim(dat: pd.DataFrame, data_dir: pathlib.Path) -> None:
    dat.to_csv(data_dir / "interim" / "all_clean.csv")
    dat.loc["train"].to_csv(data_dir / "interim" / "train_clean.csv")
    dat.loc["test"].to_csv(data_dir / "interim" / "test_clean.csv")


def prepare_data(data_dir: pathlib.Path) -> None:
    X_train, X_test, y_train = read_raw_data(data_dir)
    dat = fill_missing_data(dat)
    write_data_to_interim(dat, data_dir)
