import pathlib
import re
import string

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def read_interim_data(data_dir: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "interim" / "all_clean.csv", index_col=[0, 1])


def calculate_survival_rate(dat: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in ("Family", "Ticket"):
        msg = f'{col} must be either "Family" or "Ticket"'
        raise ValueError(msg) from None

    col2 = "FamilySize" if col == "Family" else "TicketFrequency"

    non_unique = pd.Series([x for x in dat.loc["train", col].unique() if x in dat.loc["test", col].unique()])
    survival_rates = dat.loc["train"].groupby(col)[[col2, "Survived"]].mean()
    rates = survival_rates.loc[survival_rates[col2] > 1, "Survived"].filter(non_unique, axis="index")
    dat2 = dat.join(rates.rename(f"{col}SurvivalRate"), on=col)
    dat2[f"{col}SurvivalRate"] = dat2[f"{col}SurvivalRate"].fillna(dat.loc["train", "Survived"].mean())
    dat2[f"{col}SurvivalRateNA"] = dat[col].isin(rates.index).astype(int)
    return dat2[[f"{col}SurvivalRate", f"{col}SurvivalRateNA"]]


def create_features(dat: pd.DataFrame) -> pd.DataFrame:
    dat["Fare"] = pd.qcut(dat["Fare"], 13)
    dat["Age"] = pd.qcut(dat["Age"], 10)
    dat["FamilySize"] = dat["SibSp"] + dat["Parch"] + 1

    dat["FamilySizeGroup"] = dat["FamilySize"].case_when(
        [
            (dat["FamilySize"].eq(1), "Alone"),
            (dat["FamilySize"].le(4), "Small"),
            (dat["FamilySize"].le(6), "Medium"),
            (dat["FamilySize"].gt(6), "Large"),
        ]
    )

    dat["TicketFrequency"] = dat.groupby("Ticket")["Ticket"].transform("count")

    dat["Title"] = dat["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dat["IsMarried"] = dat["Title"].eq("Mrs").astype(int)

    dat["Title"] = dat["Title"].replace(
        ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"], "Mrs/Ms/Miss"
    )
    dat["Title"] = dat["Title"].replace(
        ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"], "Dr/Military/Noble/Clergy"
    )

    names = dat["Name"].str.split(",", expand=True)[0]
    dat["Family"] = names.str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)

    dat[["FamilySurvivalRate", "FamilySurvivalRateNA"]] = calculate_survival_rate(dat, "Family")
    dat[["TicketSurvivalRate", "TicketSurvivalRateNA"]] = calculate_survival_rate(dat, "Ticket")
    dat["SurvivalRate"] = (dat["FamilySurvivalRate"] + dat["TicketSurvivalRate"]) / 2
    dat["SurvivalRateNA"] = (dat["FamilySurvivalRateNA"] + dat["TicketSurvivalRateNA"]) / 2

    return dat.drop(
        columns=[
            "Name",
            "SibSp",
            "Parch",
            "Ticket",
            "FamilySize",
            "Family",
            "FamilySurvivalRate",
            "TicketSurvivalRate",
            "FamilySurvivalRateNA",
            "TicketSurvivalRateNA",
        ]
    )


def encode_features(dat: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.Series):
    ord_features = ["Pclass", "Age", "Fare"]
    cat_features = ["Sex", "Deck", "Embarked", "Title", "FamilySizeGroup"]

    encoder = ColumnTransformer(
        [
            ("ordinal", OrdinalEncoder(), ord_features),
            ("onehot", OneHotEncoder(handle_unknown="warn"), cat_features),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(
        [
            ("encoding", encoder),
            ("scaling", StandardScaler()),
        ]
    )

    train_trans = pipe.fit_transform(dat.loc["train"].drop(columns="Survived"))
    test_trans = pipe.transform(dat.loc["test"])
    colnames = pipe.get_feature_names_out()

    X_train = pd.DataFrame(train_trans, columns=colnames, index=dat.loc["train"].index)
    X_test = pd.DataFrame(test_trans, columns=colnames, index=dat.loc["test"].index)
    y_train = dat.loc["train", "Survived"]

    return X_train, X_test, y_train


def write_data_to_processed(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, data_dir: pathlib.Path
) -> None:
    X_train.to_csv(data_dir / "processed" / "X_train.csv")
    X_test.to_csv(data_dir / "processed" / "X_test.csv")
    y_train.to_csv(data_dir / "processed" / "y_train.csv")


def engineer_features(data_dir: pathlib.Path) -> None:
    dat = read_interim_data(data_dir)
    dat = create_features(dat)
    X_train, X_test, y_train = encode_features(dat)
    write_data_to_processed(X_train, X_test, y_train, data_dir)
