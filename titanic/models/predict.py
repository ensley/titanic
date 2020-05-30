import logging
from pathlib import Path

import click
import joblib
import pandas as pd


def predict(model, data) -> pd.DataFrame:
    """Generate test set predictions in the correct format for submission to
    Kaggle.

    Args:
        model: A fitted RandomizedSearchCV object
        data: Test data

    Returns:
        A DataFrame with one column of predicted ``Survived`` values, indexed
        by passenger ID
    """
    preds = model.predict(data)
    return pd.DataFrame({'Survived': preds}, index=data.index)


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(data_filepath: str, model_filepath: str, output_filepath: str) -> None:
    """A wrapper around :func:`~titanic.models.predict.predict` that can be run
    from the command line.

    Args:
        data_filepath: Path to the processed data
        model_filepath: Path to the pickled models
        output_filepath: Path where the predictions will be saved
    """
    data_filepath = Path(data_filepath)
    model_filepath = Path(model_filepath)
    output_filepath = Path(output_filepath)
    output_filepath.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(data_filepath / 'test.csv')

    for model_path in model_filepath.glob('*.joblib'):
        model = joblib.load(model_path)
        predictions = predict(model, test)
        predictions.to_csv(output_filepath / f'{model_path.stem}.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
