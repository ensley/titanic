import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the data by imputing and creating new features that will be useful for
    prediction.

    Performs the following steps:
        1. Converts ``Survived`` to 0/1 rather than ``True``/``False``
        2. Creates an indicator for missing ``Cabin`` information
        3. Extracts titles from ``Name``
        4. Defines ``Relatives``, the sum of ``SibSp`` and ``Parch``
        5. Bins ``Relatives``, ``SibSp`` and ``Parch``
        6. Drops ``Name``, ``Ticket`` and ``Cabin``
        7. Imputes ``Embarked`` using the most frequent embarkation port
        8. Imputes ``Age`` and ``Fare`` based on 5-nearest neighbors
        9. Scales ``Age`` and ``Fare``
        10. Bins ``Age``
        11. One-hot encodes categorical variables

    Args:
        df: Cleaned and combined data output by :func:`~titanic.data.clean.make_dataset`

    Returns:
        pd.DataFrame: The processed data
    """
    # convert Survived to 0/1 rather than True/False
    df['Survived'] = df['Survived'].astype(float)
    # create indicator for missing cabin information
    df['has_cabin'] = (~df['Cabin'].isnull()).astype(float)
    # extract titles
    titles = df['Name'].str.extract('^.*, (.*?)\\.', expand=False)
    df['Title'] = titles.where(titles.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Other')
    # create "relatives" feature, the sum of SibSp and Parch
    df['Relatives'] = pd.cut(df['SibSp'] + df['Parch'], bins=[-1, 0, 3, np.Inf], labels=['0', '1-3', '4+'])
    # bin SibSp and Parch
    df['SibSp'] = pd.cut(df['SibSp'], bins=[-1, 0, 1, np.Inf], labels=['0', '1', '2+'])
    df['Parch'] = pd.cut(df['Parch'], bins=[-1, 0, np.Inf], labels=['0', '1+'])
    # drop unnecessary features
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # impute Embarked using the most frequently appearing port
    df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Embarked']])

    # do one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # impute age and fare using k nearest neighbors
    impute_age = KNNImputer(n_neighbors=5, weights='distance')
    df_num = df.select_dtypes(include='number')
    df_num = pd.DataFrame(impute_age.fit_transform(df_num), index=df_num.index, columns=df_num.columns)
    df[['Age', 'Fare']] = df_num[['Age', 'Fare']]

    # bin age now that it has been imputed
    df['AgeBin'] = pd.cut(df['Age'],
                          bins=[-1, 6, 14, 25, 45, np.Inf],
                          labels=['0-6', '7-14', '15-25', '26-45', '45+'])

    # do one-hot encoding again to incorporate age bins
    df = pd.get_dummies(df, drop_first=True)

    # scale age and fare
    scaler = StandardScaler().fit(df.loc['train', ['Age', 'Fare']])
    for source in ('train', 'test'):
        df.loc[source, ['Age', 'Fare']] = scaler.transform(df.loc[source, ['Age', 'Fare']])

    return df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """A wrapper around :func:`~titanic.features.transform.engineer_features`
    that can be called from the command line.

    Args:
        input_filepath: Path to the data created by :func:`~titanic.data.clean.make_dataset`
        output_filepath: Path where the processed data will be saved
    """
    logger = logging.getLogger(__file__)
    logger.info('making final data set from raw data')

    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    output_filepath.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(input_filepath / 'all.pkl.zip')
    df = engineer_features(df)

    df.loc['train'].to_csv(output_filepath / 'train.csv')
    df.loc['test'].drop('Survived', axis=1).to_csv(output_filepath / 'test.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
