import logging
from pathlib import Path
from typing import Dict

import click
import joblib
import pandas as pd
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

MODELS = {
    'logistic_regression': {
        'estimator': LogisticRegression(max_iter=500),
        'params': {
            'C': loguniform(1e-3, 1e-2)
        }
    },
    'support_vector_classifier': {
        'estimator': SVC(max_iter=100000, cache_size=1000),
        'params': {
            'C': loguniform(1e-3, 10),
            'kernel': ['linear', 'poly', 'rbf']
        }
    },
    'random_forest': {
        'estimator': RandomForestClassifier(),
        'params': {
            'n_estimators': list(range(10, 500, 10)),
            'max_depth': list(range(2, 11)),
            'min_samples_split': uniform(0, 0.5),
            'min_samples_leaf': uniform(0, 0.5)
        }
    }
}


def fit_model(x: pd.DataFrame, y: pd.Series, model: dict, seed: int = None) -> RandomizedSearchCV:
    """Fit a model to data.

    Performs a 5-fold cross-validated randomized search to find optimal
    model hyperparameters.

    Args:
        x: Training data covariates
        y: Training data response
        model: A dictionary where ``model['estimator']`` is a Scikit-learn estimator
            and ``model['params']`` is a set of parameter distributions that are
            passed to `RandomizedSearchCV`_. For example::

                {
                    'estimator': LogisticRegression(),
                    'params': {'C': loguniform(1e-3, 1e-2)}
                }

        seed: A random seed for the CV procedure.

    Returns:
        A fitted ``RandomizedSearchCV`` object.

    .. _RandomizedSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn-model-selection-randomizedsearchcv
    """
    search = RandomizedSearchCV(model['estimator'], model['params'],
                                n_iter=50, n_jobs=-1, verbose=0, random_state=seed)
    search.fit(x, y)
    return search


def print_cv_score(fitted_model: RandomizedSearchCV) -> str:
    """Pretty prints the best CV score attained by the randomized hyperparameter
    search.

    Args:
        fitted_model: The fitted CV search object

    Returns:
        A string stating the name of the estimator, the score, and the
        95% margin of error around the score.
    """
    estimator_name = fitted_model.estimator.__class__.__name__
    accuracy = fitted_model.best_score_
    stderr = fitted_model.cv_results_['std_test_score'][fitted_model.best_index_]
    return f'{estimator_name} accuracy: {accuracy:.2%} +/- {1.96 * stderr:.2%}'


def fit_models(x: pd.DataFrame, y: pd.Series, models: dict, seed: int = None) -> Dict[str, RandomizedSearchCV]:
    """Fit multiple models in succession.

    This is nothing more than an iteration through the models, calling :func:`~titanic.models.train.fit_model`
    on each one.

    Args:
        x: Training data covariates
        y: Training data response
        models: A dictionary of model specifications
        seed: A random seed to apply to each CV procedure

    Returns:
        A dictionary with keys matching the keys of ``models`` and values that
        are fitted RandomizedSearchCV objects.
    """
    return {k: fit_model(x, y, m, seed=seed) for k, m in models.items()}


def vote(x: pd.DataFrame, y: pd.Series, fitted_models: dict) -> VotingClassifier:
    """Train a voting classifier based on a set of fitted models

    Args:
        x: Training data covariates
        y: Training data response
        fitted_models: Fitted RandomizedSearchCV objects from :func:`~titanic.models.train.fit_models`

    Returns:
        A fitted voting classifier model
    """
    best_models = [(k, v.best_estimator_) for k, v in fitted_models.items()]
    ensemble = VotingClassifier(best_models, voting='hard')
    ensemble.fit(x, y)
    return ensemble


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """A wrapper around :func:`~titanic.models.train.fit_models` that can be
    called from the command line.

    Fits the models, pickles the fitted model objects, and writes them out to
    ``output_filepath``.

    Args:
        input_filepath: Path to the processed data
        output_filepath: Path where the fitted models will be saved
    """
    logger = logging.getLogger(__file__)
    logger.info('training models')

    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    output_filepath.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(input_filepath / 'train.csv')
    X = train.drop('Survived', axis=1)
    y = train['Survived']

    # perform CV parameter searches
    fitted_models = fit_models(X, y, MODELS, seed=10)

    # log best accuracy scores and save best models
    for name, fitted_model in fitted_models.items():
        logger.info(print_cv_score(fitted_model))
        joblib.dump(fitted_model.best_estimator_, output_filepath / f'{name}.joblib')

    # extract best models, use them to train a voting classifier
    ensemble = vote(X, y, fitted_models)
    joblib.dump(ensemble, output_filepath / 'ensemble.joblib')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
