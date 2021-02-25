"""LTL Synthesis using the Transformer"""

import csv
import json
import logging
import os

from collections import Counter
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from deepsynthesis.data import aiger
from deepsynthesis.data import synthesis
from deepsynthesis.data.mc import nuxmv
from deepsynthesis.data.synthesis import bosy
from deepsynthesis.experiments.seq2seq_experiment import Seq2SeqExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisExperiment(Seq2SeqExperiment):

    def __init__(self,
                 aiger_order=None,
                 batch_size=128,
                 dataset=('/deep-synthesis/data/SC100'),
                 encode_realizable=False,
                 inputs=None,
                 max_input_length=128,
                 max_target_length=64,
                 outputs=None,
                 strix_path='/strix',
                 **kwargs):
        self.aiger_order = aiger_order if aiger_order else [
            'header', 'inputs', 'latches', 'outputs', 'ands'
        ]
        self.encode_realizable = encode_realizable
        self.inputs = inputs if inputs else ['i0', 'i1', 'i2', 'i3', 'i4']
        self.outputs = outputs if outputs else ['o0', 'o1', 'o2', 'o3', 'o4']
        self.strix_path = strix_path
        super().__init__(batch_size=batch_size,
                         dataset=dataset,
                         max_input_length=max_input_length,
                         max_target_length=max_target_length,
                         **kwargs)

    @property
    def abbr_name_args(self):
        result = super().abbr_name_args
        result.update({
            'aiger_order': 'ao',
            'encode_realizable': 'er'
        })
        return result

    @property
    def aiger_order_abbr_value(self):
        return '_' + ''.join([component[0] for component in self.aiger_order])

    @property
    def init_verifier(self):

        def verifier(problem, solution, solvable=True):
            return nuxmv.wrapper(problem,
                                 solution,
                                 realizable=solvable,
                                 strix_path=self.strix_path,
                                 temp_dir=self.temp_dir)

        return verifier

    def call(self, specification, training=False, verify=False):
        if not self.input_encoder.encode(specification):
            logger.info('Econding error: %s', self.input_encoder.error)
            return None
        formula_tensor, pos_enc_tensor = self.input_encoder.tensor
        #pylint: disable=E1102
        preds = self.eval_model((tf.expand_dims(
            formula_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
                                training=training)[0]
        results = []
        for beam in preds[0]:
            if not self.target_encoder.decode(np.array(beam)):
                logger.info('Decoding error: %s', self.target_encoder.error)
                return None
            beam_result = {}
            beam_result['circuit'] = self.target_encoder.circuit
            if verify:
                #pylint: disable=E1102
                beam_result['verification'] = self.verifier(
                    specification, beam_result['circuit'] + '\n')
            results.append(beam_result)
        return results

    def eval_syntcomp(self, syntcomp_dir, steps=None, training=False):

        def spec_filter(spec):
            return spec.num_inputs <= len(
                self.inputs) and spec.num_outputs <= len(self.outputs)

        spec_ds = bosy.dataset.from_dir(syntcomp_dir, spec_filter)
        spec_ds.rename_aps(self.inputs, self.outputs)
        for spec in spec_ds.dataset:
            spec.inputs = self.inputs
            spec.outputs = self.outputs
        self.eval_generator(spec_ds.dataset,
                            'syntcomp',
                            includes_target=False,
                            steps=steps,
                            training=training,
                            verify=True)

    def eval_timeouts(self, steps=None, training=False):
        timeouts = synthesis.dataset.from_file(
            os.path.join(self.data_dir, 'timeouts.csv'))
        timeouts = [sample for sample, _ in timeouts.generator()]
        self.eval_generator(timeouts,
                            'timeouts',
                            includes_target=False,
                            steps=steps,
                            training=training,
                            verify=True)

    def eval_generator(self,
                       generator,
                       name,
                       includes_target=False,
                       steps=None,
                       training=False,
                       verify=False):
        filename = f'{name}-a{self.alpha}-bs{self.beam_size}'
        log_filepath = os.path.join(self.model_dir, filename + '-log.csv')
        log_file = open(log_filepath, 'w')
        fieldnames = ['beam', 'status', 'problem', 'prediction', 'target']
        file_writer = csv.DictWriter(log_file,
                                     fieldnames=fieldnames,
                                     quoting=csv.QUOTE_ALL)
        file_writer.writeheader()
        counters = Counter()
        problem_batch, formula_batch, pos_enc_batch, target_batch = [], [], [], []
        with tqdm(desc='Evaluated samples', unit='sample') as pbar:
            for sample in generator:
                counters['samples'] += 1
                problem = sample[0] if includes_target else sample
                problem_name = problem.name if problem.name else problem.formula
                target = sample[1] if includes_target else None
                row = {
                    'beam': 0,
                    'problem': problem_name,
                    'prediction': '',
                    'target': target.replace('\n', '\\n') if target else ''
                }
                if not self.input_encoder.encode(problem):
                    row['status'] = f'Encoding Error {self.input_encoder.error}'
                    file_writer.writerow(row)
                    counters['encoding_error'] += 1
                    pbar.update()
                elif includes_target and not self.target_encoder.encode(target):
                    row['status'] = f'Target Error {self.target_encoder.error}'
                    file_writer.writerow(row)
                    counters['target_error'] += 1
                    pbar.update()
                else:
                    problem_batch.append(problem)
                    formula_tensor, pos_enc_tensor = self.input_encoder.tensor
                    formula_batch.append(formula_tensor)
                    pos_enc_batch.append(pos_enc_tensor)
                    if includes_target:
                        target_batch.append(sample[1])
                if counters['samples'] % self.batch_size == 0 and problem_batch:
                    batch_dataset = tf.data.Dataset.from_tensor_slices(
                        (formula_batch, pos_enc_batch))
                    batch = next(iter(batch_dataset.batch(self.batch_size)))
                    predictions, _ = self.eval_model(batch, training=training)  #pylint: disable=E1102
                    for i, pred in enumerate(predictions):
                        any_beam_satisfied = False
                        problem = problem_batch[i]
                        target = target_batch[i] if includes_target else ''
                        problem_name = problem.name if problem.name else problem.formula
                        for beam_id, beam in enumerate(pred):
                            row = {
                                'beam': beam_id,
                                'problem': problem_name,
                                'prediction': '',
                                'target': target.replace('\n', '\\n')
                            }
                            if not self.target_encoder.decode(np.array(beam)):
                                row['status'] = f'Decoding Error {self.target_encoder.error}'
                                row['prediction'] = np.array2string(
                                    np.array(beam),
                                    max_line_width=3 * self.max_target_length)
                                file_writer.writerow(row)
                                counters['decoding_error'] += 1
                                continue
                            realizable = self.target_encoder.realizable #True # 'i0 i0' in target
                            circuit = self.target_encoder.circuit
                            row['prediction'] = circuit.replace('\n', '\\n')
                            if includes_target:
                                if circuit == target:
                                    row['status'] = 'Match'
                                    counters['match'] += 1
                            result = self.verifier(problem, circuit + '\n', realizable)  #pylint: disable=E1102
                            counters[result.value] += 1
                            if result.value == 'satisfied':
                                any_beam_satisfied = True
                            if 'status' not in row:
                                row['status'] = result.value.capitalize()
                            else:
                                if row['status'] == 'Match' and result.value != 'satisfied':
                                    logger.warning('Match not satisfied')
                            file_writer.writerow(row)
                        if any_beam_satisfied:
                            counters['beam_search_satisfied'] += 1
                        pbar.update()
                        pbar.set_postfix(counters)
                    problem_batch, formula_batch, pos_enc_batch, target_batch = [], [], [], []
                    counters['steps'] += 1
                    if steps and counters['steps'] >= steps:
                        break
        log_file.close()
        stats_filepath = os.path.join(self.model_dir, filename + '-stats.json')
        with open(stats_filepath, 'w') as stats_file:
            json.dump(counters, stats_file, indent=2)

    @property
    def init_dataset(self):
        return synthesis.dataset.from_dir(self.data_dir)

    @property
    def init_target_encoder(self):
        return aiger.SequenceEncoder(
            start=True,
            eos=True,
            pad=self.max_target_length,
            components=self.aiger_order,
            encode_start=False,
            encode_realizable=self.encode_realizable,
            inputs=self.inputs,
            outputs=self.outputs)

    @classmethod
    def get_arg_parser(cls):
        parser = super().get_arg_parser()
        parser.description = 'Synthesis experiment'
        defaults = cls.get_default_args()
        parser.add_argument('--aiger-order',
                            nargs='*',
                            default=defaults['aiger_order'])
        parser.add_argument('--encode-realizable', action='store_true')
        parser.add_argument('--inputs', nargs='*', default=defaults['inputs'])
        parser.add_argument('--outputs', nargs='*', default=defaults['outputs'])
        parser.add_argument('--strix_path', default=defaults['strix_path'])
        return parser
