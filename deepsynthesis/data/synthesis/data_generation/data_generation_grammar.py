"""A generator that generates synthesis data from a file of guarantees"""

import argparse
import copy
import json
import logging
import os.path

import numpy as np
from numpy import random

import ray
from ray.util.queue import Queue

from deepsynthesis.data import aiger
from deepsynthesis.data import dataset
from deepsynthesis.data.utils import int_to_abbrev_str
from deepsynthesis.data.synthesis.common import Status
from deepsynthesis.data.synthesis.dataset import from_dir
from deepsynthesis.data.synthesis.data_generation import common
from deepsynthesis.data.synthesis.patterns.grammar.grammar import Grammar
from deepsynthesis.data.statistics import plot_stats, write_stats
from deepsynthesis.data.synthesis import strix

ray.init(dashboard_host='0.0.0.0')
#ray.init(address='auto')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote
class DataSetActor:

    def __init__(self, inputs, outputs, progress_actor, params, sample_queue,
                 timeouts_queue):
        self.inputs_no_neg = inputs
        self.outputs_no_neg = outputs
        self.inputs = inputs + [f'! {i}' for i in inputs]
        self.outputs = outputs + [f'! {o}' for o in outputs]
        self.progress_actor = progress_actor
        self.params = params
        self.sample_queue = sample_queue
        self.timeouts_queue = timeouts_queue

        logger.setLevel(logging.INFO)

        pattern_grammar = Grammar()
        self.in_out_patterns = pattern_grammar.input_output_patterns
        logger.info('Loaded %d input output patterns',
                    len(self.in_out_patterns))
        self.out_out_patterns = pattern_grammar.output_output_patterns
        logger.info('Loaded %d output output patterns',
                    len(self.out_out_patterns))

        self.counters = {
            'processing': 0,
            'valid': 0,
            'realizable': 0,
            'ands': {}
        }
        self.prob_realizable = self.params['realizable_frac']
        self.open = []
        self.sample_ids = []

    def register_worker(self):
        self.progress_actor.update.remote('worker')

    def has_unsolved_problems(self):
        return self.counters['valid'] + self.counters[
            'processing'] < self.params['num_samples']

    def get_problems(self):
        batch_size = self.params['batch_size']
        if self.counters['valid'] + self.counters[
                'processing'] + batch_size > self.params['num_samples']:
            batch_size = self.params['num_samples'] - self.counters[
                'valid'] - self.counters['processing']
        problems = []
        for _ in range(batch_size):
            if self.open:
                parent_problem = self.open.pop(0)
                parent_problem['parent'] = None
                problem = copy.deepcopy(parent_problem)
                problem['parent'] = parent_problem
                problem['circuit'] = ''
            else:
                problem = {
                    'guarantees': [],
                    'inputs': [],
                    'outputs': [],
                    'in_out_ids': [],
                    'out_out_ids': [],
                    'parent': None
                }
            out_pattern = random.choice(
                [True, False],
                p=[
                    self.params['out_pattern_prob'],
                    1.0 - self.params['out_pattern_prob']
                ])
            if out_pattern:
                pattern_id = random.choice(len(self.out_out_patterns))
                pattern = self.out_out_patterns[pattern_id]
                problem['out_out_ids'].append(pattern_id)
            else:
                if problem['in_out_ids'] and random.binomial(
                        1, self.params['in_pattern_repeat_prob']):
                    pattern_id = problem['in_out_ids'][-1]
                else:
                    pattern_id = random.choice(len(self.in_out_patterns))
                pattern = self.in_out_patterns[pattern_id]
                problem['in_out_ids'].append(pattern_id)
            pattern_inputs = random.choice(self.inputs,
                                           pattern.num_inputs,
                                           replace=False)
            pattern_outputs = random.choice(self.outputs,
                                            pattern.num_outputs,
                                            replace=False)
            guarantee = pattern.fill(pattern_inputs, pattern_outputs)
            problem['guarantees'].append(guarantee)
            logger.debug(problem['guarantees'])
            if self.params['all_aps']:
                problem['inputs'] = self.inputs_no_neg
                problem['outputs'] = self.outputs_no_neg
            else:
                pattern_inputs = [i.replace('! ', '') for i in pattern_inputs]
                pattern_outputs = [o.replace('! ', '') for o in pattern_outputs]
                problem['inputs'] = sorted(
                    list(set().union(pattern_inputs, problem['inputs'])))
                problem['outputs'] = sorted(
                    list(set().union(pattern_outputs, problem['outputs'])))
            logger.debug(problem['inputs'])
            logger.debug(problem['outputs'])
            problems.append(problem)
        self.counters['processing'] += batch_size
        return problems

    def post_solved_problems(self, problems):
        batch_size = len(problems)
        for problem in problems:
            status = problem['status']
            num_guarantees = len(problem['guarantees'])
            max_num_guarantees = self.params['num_guarantees'][1]
            if status == Status.REALIZABLE and (
                    not max_num_guarantees or
                    num_guarantees < max_num_guarantees):
                self.open.append(problem)
            elif status == Status.UNREALIZABLE or status == Status.TIMEOUT or (
                    status == Status.REALIZABLE and
                    num_guarantees == max_num_guarantees):
                if 0.0 < self.prob_realizable < 1.0:
                    max_num_realizable = int(self.params['num_samples'] *
                                             self.prob_realizable)
                    max_num_unrealizable = self.params[
                        'num_samples'] - max_num_realizable
                    if self.counters['realizable'] >= max_num_realizable:
                        self.prob_realizable = 0.0
                    if self.counters['valid'] - self.counters[
                            'realizable'] >= max_num_unrealizable:
                        self.prob_realizable = 1.0
                choose_realizable = np.random.choice(
                    [True, False],
                    p=[self.prob_realizable, 1.0 - self.prob_realizable])
                if status == Status.TIMEOUT:
                    problem['realizable'] = -1
                    self.timeouts_queue.put(problem)
                    self.progress_actor.update.remote('timeout')
                if not choose_realizable and status != Status.UNREALIZABLE:
                    continue
                if choose_realizable and status in (Status.UNREALIZABLE,
                                                    Status.TIMEOUT):
                    problem = problem['parent']
                if not problem:
                    continue
                problem['realizable'] = int(choose_realizable)
                try:
                    violated_bounds = common.check_upper_bounds(
                        problem, self.params, self.counters)
                except ValueError as error:
                    logger.error(
                        ('Checking upper bounds of the following sample failed:'
                         '\n%s\nwith error:\n%s'), problem, error)
                    continue
                if violated_bounds:
                    self.progress_actor.update_multi.remote(violated_bounds)
                    self.progress_actor.update.remote('invalid')
                    continue
                try:
                    violated_bounds = common.check_lower_bounds(
                        problem, self.params)
                except ValueError as error:
                    logger.error(
                        ('Checking lower bounds of the following sample failed:'
                         '\n%s\nwith error:\n%s'), problem, error)
                    continue
                if violated_bounds:
                    self.progress_actor.update_multi.remote(violated_bounds)
                    self.progress_actor.update.remote('invalid')
                    continue
                if self.params['unique_samples']:
                    ids = (tuple(sorted(problem['in_out_ids'])),
                           tuple(sorted(problem['out_out_ids'])))
                    if ids in self.sample_ids:
                        self.progress_actor.update.remote('duplicates')
                        continue
                    self.sample_ids.append(ids)
                self.sample_queue.put(problem)
                self.counters['valid'] += 1
                if self.params['max_frac_ands']:
                    _, _, _, _, num_ands = aiger.header_ints_from_str(
                        problem['circuit'])
                    self.counters['ands'][num_ands] = self.counters['ands'].get(
                        num_ands, 0) + 1
                if choose_realizable:
                    self.counters['realizable'] += 1
                    self.progress_actor.update.remote('realizable')
                self.progress_actor.update.remote('samples')
            elif status == Status.ERROR:
                self.progress_actor.update.remote('error')
                logger.warning('Error occured for problem %s', problem)
            else:
                logger.warning('Unknown status %s for problem %s', status,
                               problem)
        self.counters['processing'] -= batch_size
        return


def get_folder_name(args):
    folder_name = 'n' + int_to_abbrev_str(args.num_samples)
    if args.all_aps:
        folder_name += '-aap'
    if args.max_frac_ands:
        folder_name += '-af{0:g}'.format(args.max_frac_ands)
    if args.in_pattern_repeat_prob > 0:
        folder_name += '-ipr{0:g}'.format(args.in_pattern_repeat_prob)
    abbr_bound_flag_dict = {
        'na': args.num_ands,
        'nal': args.num_ands_plus_latches,
        'ng': (None if args.num_guarantees[0] == 1 else args.num_guarantees[0],
               args.num_guarantees[1]),
        'nl': args.num_latches,
        'nv': args.num_vars
    }
    for abbr in abbr_bound_flag_dict:
        min_value, max_value = abbr_bound_flag_dict[abbr]
        if min_value or max_value:
            min_str = f'{min_value}' if min_value else ''
            max_str = f'{max_value}' if max_value else ''
            folder_name += f'-{min_str}{abbr}{max_str}'
    folder_name += '-opp{0:g}'.format(args.out_pattern_prob)
    if args.realizable_frac < 1.0:
        folder_name += '-rf{0:g}'.format(args.realizable_frac)
    if not args.strix_auto:
        folder_name += '-nsa'
    if args.strix_timeout:
        folder_name += '-st{0:g}'.format(args.strix_timeout)
    if args.unique_samples:
        folder_name += '-us'
    return folder_name


def main(args):
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    folder_path = f'{args.data_dir}/{get_folder_name(args)}'
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        logger.info('Created folder %s', folder_path)
    data_gen_stats_file = os.path.join(folder_path, 'data_gen_stats.json')
    flag_filepath = os.path.join(folder_path, 'args.json')
    args_dict = vars(args)
    with open(flag_filepath, 'w') as flag_file:
        json.dump(args_dict, flag_file, indent=2)
    logger.info('Command line arguments written to %s', flag_filepath)
    progress_actor = common.ProgressActor.remote()  # pylint: disable=no-member
    samples_queue = Queue(maxsize=args.num_samples)
    # pylint: disable=no-member
    timeouts_queue = Queue(maxsize=args.num_samples)
    ds_actor = DataSetActor.remote(args.inputs, args.outputs, progress_actor,
                                   args_dict, samples_queue, timeouts_queue)
    dataset_writer_result = common.csv_dataset_writer.remote(
        samples_queue, folder_path, args.num_samples, args.train_frac,
        args.val_frac, args.test_frac)
    timeouts_file = os.path.join(folder_path, 'timeouts.csv')
    timeouts_writer_result = common.csv_file_writer.remote(
        timeouts_queue, timeouts_file)
    worker_results = [
        strix.wrapper.worker.remote(ds_actor,
                                    args.strix_bin,
                                    strix_auto=args.strix_auto,
                                    strix_timeout=args.strix_timeout,
                                    id=i) for i in range(args.num_worker)
    ]
    common.progress_bar(progress_actor, args.num_samples, data_gen_stats_file)
    ray.get(worker_results)
    ray.get(dataset_writer_result)
    timeouts_queue.put(None)
    ray.get(timeouts_writer_result)
    split_dataset = from_dir(folder_path)
    stats = split_dataset.circuit_stats(['train', 'val', 'test'])
    stats_file = os.path.join(folder_path, 'circuit-stats.json')
    write_stats(stats, stats_file)
    plot_file = os.path.join(folder_path, 'circuit-stats.png')
    plot_stats(stats, plot_file)
    split_dataset.shuffle()
    split_dataset.save(folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates a synthesis dataset using grammar patterns')
    common.add_parser_args(parser)
    dataset.add_parser_args(parser)
    strix.wrapper.add_parser_args(parser)
    parser.add_argument('-d',
                        '--data-dir',
                        type=str,
                        default='/deep-synthesis/data',
                        metavar='dir',
                        help=('directory to save datasets'))
    parser.add_argument('--in-pattern-repeat-prob', type=float, default=0.0)
    parser.add_argument('--inputs',
                        nargs='*',
                        default=['i0', 'i1', 'i2', 'i3', 'i4'],
                        help='list of input atomic propositions')
    parser.add_argument('--no-unique-samples',
                        action='store_false',
                        dest='unique_samples',
                        help='samples in dataset are not necessarily unique')
    parser.add_argument('--out-pattern-prob',
                        type=float,
                        default=0.2,
                        help=('probability of choosing an '
                              'output pattern when sampling patterns'))
    parser.add_argument('--outputs',
                        nargs='*',
                        default=['o0', 'o1', 'o2', 'o3', 'o4'],
                        help='list of output atomic propositions')
    main(parser.parse_args())
