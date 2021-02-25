"""Synthesis dataset"""

import csv
import os
import re
from tqdm import tqdm

import pandas as pd

from deepsynthesis.data import aiger
from deepsynthesis.data import dataset
from deepsynthesis.data.utils import from_csv_str, to_csv_str
from deepsynthesis.data.statistics import stats_from_counts
from deepsynthesis.data import ltl

# circuit statistics keys
MAX_VAR_INDEX = 'MAX VARIABLE INDEX'
NUM_INPUTS = 'NUM INPUTS'
NUM_LATCHES = 'NUM LATCHES'
NUM_OUTPUTS = 'NUM OUTPUTS'
NUM_AND_GATES = 'NUM AND GATES'


def sample_to_csv_row(sample: dict):
    guarantees = ','.join(sample['guarantees'])
    inputs = ','.join(sample['inputs'])
    outputs = ','.join(sample['outputs'])
    realizable = sample['realizable']
    circuit = to_csv_str(sample['circuit'])
    return [guarantees, inputs, outputs, realizable, circuit]


def csv_row_to_sample(row: list):
    return {
        'guarantees': row[0].split(','),
        'inputs': row[1].split(','),
        'outputs': row[2].split(','),
        'realizable': int(row[3]),
        'circuit': from_csv_str(row[4])
    }


def from_dir(dataset_dir, splits=None):
    if not splits:
        splits = ['train', 'val', 'test', 'timeouts']
    split_dataset = SplitDataset()
    for split in splits:
        filepath = os.path.join(dataset_dir, split + '.csv')
        split_dataset[split] = from_file(filepath)
    return split_dataset


def from_file(filepath):
    data_frame = pd.read_csv(
        filepath,
        converters={'circuit': lambda c: str(c).replace('\\n', '\n')},
        dtype={
            'guarantees': str,
            'inputs': str,
            'outputs': str,
            'realizable': int
        },
        keep_default_na=False)
    return Dataset(data_frame)


class Dataset(dataset.SupervisedDataset):

    def sample_generator(self):
        for _, row in self.data_frame.iterrows():
            sample = {
                'guarantees':
                    row['guarantees'].split(',') if row['guarantees'] else [],
                'inputs':
                    row['inputs'].split(',') if row['inputs'] else [],
                'outputs':
                    row['outputs'].split(',') if row['outputs'] else [],
                'realizable':
                    row['realizable'],
                'circuit':
                    row['circuit']
            }
            yield sample

    def generator(self):
        for sample in self.sample_generator():
            yield ltl.specification.from_dict(sample), sample['circuit']

    def save(self, filepath):
        filepath = filepath + '.csv'
        circuit_series = self.data_frame['circuit']
        self.data_frame['circuit'] = circuit_series.str.replace('\n', '\\n')
        self.data_frame.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)
        self.data_frame['circuit'] = circuit_series


class SplitDataset(dataset.SplitSupervisedDataset):

    def circuit_stats(self, splits=None):
        counts = {
            MAX_VAR_INDEX: [],
            NUM_INPUTS: [],
            NUM_LATCHES: [],
            NUM_OUTPUTS: [],
            NUM_AND_GATES: []
        }
        for _, circuit in self.generator(splits):
            num_var_index, num_inputs, num_latches, num_outputs, num_and_gates = aiger.header_ints_from_str(
                circuit)
            counts[MAX_VAR_INDEX].append(num_var_index)
            counts[NUM_INPUTS].append(num_inputs)
            counts[NUM_LATCHES].append(num_latches)
            counts[NUM_OUTPUTS].append(num_outputs)
            counts[NUM_AND_GATES].append(num_and_gates)
        return stats_from_counts(counts)
