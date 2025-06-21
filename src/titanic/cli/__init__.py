import logging
import pathlib
import subprocess
from zipfile import ZipFile

import click

from titanic import data, features
from titanic.__about__ import __version__

_DEFAULT_DATA_DIR = pathlib.Path(__file__).parents[3] / "data"
_DEFAULT_KAGGLE_EXECUTABLE = pathlib.Path.home() / ".local" / "bin" / "kaggle"
logging.basicConfig(level=logging.INFO)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "auto_envvar_prefix": "TITANIC"},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="titanic")
@click.option(
    "--data-dir",
    envvar="DATA_DIR",
    default=_DEFAULT_DATA_DIR,
    type=click.Path(file_okay=False, writable=True, exists=True, path_type=pathlib.Path),
)
@click.pass_context
def titanic(ctx: click.Context, data_dir: pathlib.Path) -> None:
    ctx.ensure_object(dict)
    ctx.obj["DATA_DIR"] = data_dir.resolve()
    click.echo(f"Data directory: {ctx.obj['DATA_DIR']}")


@titanic.command()
@click.option("--cleanup/--no-cleanup", is_flag=True, default=True)
@click.option(
    "--kaggle-path",
    type=click.Path(exists=True, executable=True),
    default=_DEFAULT_KAGGLE_EXECUTABLE,
    help="Path to kaggle CLI executable",
)
@click.pass_context
def fetch_data(ctx: click.Context, *, cleanup: bool, kaggle_path: pathlib.Path) -> None:
    raw_data_dir = ctx.obj["DATA_DIR"] / "raw"
    ctx.obj["KAGGLE_PATH"] = kaggle_path

    result = subprocess.run([kaggle_path, "competitions", "download", "-c", "titanic", "-p", raw_data_dir], check=False)
    try:
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        raise click.ClickException(str(e)) from e
    logging.info("Download complete")

    data_file = raw_data_dir / "titanic.zip"
    with ZipFile(data_file) as datazip:
        datazip.extractall(raw_data_dir)
    logging.info("Files extracted")

    if cleanup:
        data_file.unlink()
        logging.info("Archive removed")


@titanic.command()
@click.pass_context
def clean_data(ctx: click.Context) -> None:
    data.clean_data(ctx.obj["DATA_DIR"])


@titanic.command()
@click.pass_context
def engineer_features(ctx: click.Context) -> None:
    features.engineer_features(ctx.obj["DATA_DIR"])
