"""A generator that generates synthesis data from a file of guarantees"""

import argparse
import copy
import json
import logging
import os.path
import sys

import numpy as np

import ray
from ray.util.queue import Queue

from deepsynthesis.data import aiger
from deepsynthesis.data import dataset
from deepsynthesis.data.utils import int_to_abbrev_str
from deepsynthesis.data.synthesis.common import Status
from deepsynthesis.data.synthesis.dataset import from_dir
from deepsynthesis.data.synthesis.data_generation import common
from deepsynthesis.data.statistics import plot_stats, write_stats
from deepsynthesis.data.synthesis import strix

ray.init(dashboard_host='0.0.0.0')
#ray.init(address='auto')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote
class DataSetActor:

    def __init__(self, guarantees, progress_actor, params, sample_queue,
                 timeouts_queue):
        self.guarantees = guarantees
        self.progress_actor = progress_actor
        self.params = params
        self.sample_queue = sample_queue
        self.timeouts_queue = timeouts_queue

        self.guarantees_ids = list(range(len(self.guarantees)))
        self.counters = {
            'processing': 0,
            'valid': 0,
            'realizable': 0,
            'ands': {}
        }
        self.prob_realizable = self.params['realizable_frac']
        self.open = []
        self.sample_ids = []

        if self.params['all_aps'] or self.params['resample_aps']:
            self.inputs = set()
            self.outputs = set()
            for guarantee in self.guarantees:
                self.inputs = self.inputs.union(guarantee['inputs'])
                self.outputs = self.outputs.union(guarantee['outputs'])
            self.inputs = sorted(self.inputs)
            self.outputs = sorted(self.outputs)

        logger.setLevel(logging.INFO)

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
                    'guarantees_ids': [],
                    'parent': None
                }
            choices = self.guarantees_ids
            if self.params['unique_guarantees']:
                #list provides a deep copy
                choices = list(choices)
                for idx in problem['guarantees_ids']:
                    choices.remove(idx)
            idx = np.random.choice(choices)
            #dict provides a deep copy
            guarantee = dict(self.guarantees[idx])
            if self.params['resample_aps']:
                resampled_inputs = np.random.choice(self.inputs,
                                                    len(guarantee['inputs']),
                                                    replace=False).tolist()
                resampled_outputs = np.random.choice(self.outputs,
                                                     len(guarantee['outputs']),
                                                     replace=False).tolist()
                for ap, resampled_ap in zip(
                        guarantee['inputs'] + guarantee['outputs'],
                        resampled_inputs + resampled_outputs):
                    guarantee['pattern'] = guarantee['pattern'].replace(
                        ap, resampled_ap)
                guarantee['inputs'] = resampled_inputs
                guarantee['outputs'] = resampled_outputs
            problem['guarantees'].append(guarantee['pattern'])
            logger.debug(problem['guarantees'])
            if self.params['all_aps']:
                problem['inputs'] = self.inputs
                problem['outputs'] = self.outputs
            else:
                problem['inputs'] = sorted(
                    list(set().union(problem['inputs'], guarantee['inputs'])))
                problem['outputs'] = sorted(
                    list(set().union(problem['outputs'], guarantee['outputs'])))
            logger.debug(problem['inputs'])
            logger.debug(problem['outputs'])
            problem['guarantees_ids'].append(idx)
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
                guarantees_ids = problem['guarantees_ids']
                if self.params['unique_samples']:
                    guarantees_ids_tuple = tuple(sorted(guarantees_ids))
                    if guarantees_ids_tuple in self.sample_ids:
                        self.progress_actor.update.remote('duplicates')
                        continue
                    else:
                        self.sample_ids.append(guarantees_ids_tuple)
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
    if args.resample_aps:
        folder_name += '-rap'
    if args.realizable_frac < 1.0:
        folder_name += '-rf{0:g}'.format(args.realizable_frac)
    if not args.strix_auto:
        folder_name += '-nsa'
    if args.strix_timeout:
        folder_name += '-st{0:g}'.format(args.strix_timeout)
    if args.unique_guarantees:
        folder_name += '-ug'
    if args.unique_samples:
        folder_name += '-us'
    return folder_name


def main(args):
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    guarantees_path = args.guarantees
    if not os.path.isfile(guarantees_path):
        sys.exit(f'{guarantees_path} is not a file')
    with open(guarantees_path, 'r') as guarantees_file:
        guarantees = json.load(guarantees_file)['patterns']
        logger.info('Read in %d guarantees', len(guarantees))
    #create folder and files
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
    ds_actor = DataSetActor.remote(guarantees, progress_actor, args_dict,
                                   samples_queue, timeouts_queue)
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
        description='Generates a synthesis dataset from a set of guarantees')
    common.add_parser_args(parser)
    dataset.add_parser_args(parser)
    strix.wrapper.add_parser_args(parser)
    parser.add_argument('-d',
                        '--data-dir',
                        type=str,
                        default='/deep-synthesis/data',
                        metavar='dir',
                        help=('directory to save datasets'))
    parser.add_argument('-g',
                        '--guarantees',
                        type=str,
                        default='/deep-synthesis/data/syntcomp-patterns.json',
                        metavar='path',
                        help=('path to a file with guarantees'))
    parser.add_argument('--no-unique-guarantees',
                        action='store_false',
                        dest='unique_guarantees',
                        help=('guarantees in a single sample are not '
                              'necessarily unique'))
    parser.add_argument('--no-unique-samples',
                        action='store_false',
                        dest='unique_samples',
                        help='samples in dataset are not necessarily unique')
    parser.add_argument('--resample-aps',
                        action='store_true',
                        help='resample atomic propositions in guarantees')
    main(parser.parse_args())
