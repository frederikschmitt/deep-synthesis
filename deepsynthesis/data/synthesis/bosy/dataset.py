"""Dataset of BoSy specifications"""

import argparse
import logging
import os

from deepsynthesis.data import ltl
from deepsynthesis.data.synthesis.bosy import bosy_input

logger = logging.getLogger(__name__)


def from_dir(dir, filter=None):
    """Constructs a dataset of BoSy specifications from a directory with BoSy specification files"""
    dataset = []
    for file in os.listdir(dir):
        if file.endswith('.json'):
            bosy_spec = bosy_input.from_file(os.path.join(dir, file))
            if not filter or filter(bosy_spec):
                dataset.append(bosy_spec)
    return ltl.specification.Dataset(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds analyzes a dataset of BoSy specifications')
    parser.add_argument('--data-dir', required=True, help='Data directory')
    args = parser.parse_args()
    dataset = from_dir(args.data_dir)
    dataset.print_stats()
