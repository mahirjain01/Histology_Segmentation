import csv
import os
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger

import src.framework
import src.util.arguments

DEFAULT_LOGS_DIR = {parameter.name: parameter.default for parameter in src.util.arguments._get_arguments(CSVLogger)}['name']


def load_model(version_number: Union[int, str], epoch: Union[int, str], step: int, logs_dir: str = os.path.join('logs', DEFAULT_LOGS_DIR)) -> pl.LightningModule:
    model = src.framework.Model.load_from_checkpoint(
        os.path.join(logs_dir, f'version_{version_number}', 'checkpoints', f'epoch={epoch}-step={step+1}.ckpt')
    )
    return model


def load_log(version_number: Union[int, str], logs_dir: str = os.path.join('logs', DEFAULT_LOGS_DIR)):
    with open(os.path.join(logs_dir, f'version_{version_number}', 'metrics.csv')) as csv_file:
        reader = csv.DictReader(csv_file)
        yield from reader
