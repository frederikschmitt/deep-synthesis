"""Wrapper for calling Strix"""

import argparse
import logging
import os
import ray
import subprocess

from deepsynthesis.data.synthesis.common import Status

log = logging.getLogger(__name__)


@ray.remote
def worker(server,
           strix_bin,
           strix_auto=False,
           strix_timeout=None,
           log_level='info',
           id=0):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    server.register_worker.remote()
    while ray.get(server.has_unsolved_problems.remote()):
        problems = ray.get(server.get_problems.remote())
        for problem in problems:
            solution = wrapper_dict(problem, strix_bin, strix_auto,
                                    strix_timeout)
            problem.update(solution)
        ray.get(server.post_solved_problems.remote(problems))


def wrapper_dict(problem: dict,
                 strix_bin,
                 strix_auto=False,
                 strix_timeout=None):
    formula_str, ins_str, outs_str = input_from_dict(problem)
    return wrapper_str(formula_str, ins_str, outs_str, strix_bin, strix_auto,
                       strix_timeout)


def wrapper_str(formula_str,
                ins_str,
                outs_str,
                strix_bin,
                strix_auto=False,
                strix_timeout=None):
    try:
        args = [strix_bin, '-f', formula_str]
        if ins_str:
            args.append(f'--ins={ins_str}')
        if outs_str:
            args.append(f'--outs={outs_str}')
        if strix_auto:
            args.append('--auto')
        log.debug(f'subprocess args: {args}')
        out = subprocess.run(args, capture_output=True, timeout=strix_timeout)
    except subprocess.TimeoutExpired:
        log.debug('Strix timeout')
        return {'status': Status.TIMEOUT, 'circuit': ''}
    except subprocess.CalledProcessError:
        log.error('subprocess called process error')
        return {'status': Status.ERROR}
    except Exception as error:
        log.critical(error)
    log.debug(f'Strix returncode: {out.returncode}')
    strix_stdout = out.stdout.decode('utf-8')
    log.debug(f'Strix stdout: {strix_stdout}')
    strix_stdout_lines = strix_stdout.splitlines()
    if out.returncode == 0 and strix_stdout_lines[0] == 'REALIZABLE':
        log.debug('realizable')
        aiger_circuit = '\n'.join(strix_stdout_lines[1:])
        log.debug(f'AIGER circuit:{aiger_circuit}')
        return {'status': Status.REALIZABLE, 'circuit': aiger_circuit}
    if out.returncode == 0 and strix_stdout_lines[0] == 'UNREALIZABLE':
        log.debug('unrealizable')
        aiger_circuit = '\n'.join(strix_stdout_lines[1:])
        log.debug(f'AIGER circuit:{aiger_circuit}')
        return {'status': Status.UNREALIZABLE, 'circuit': aiger_circuit}
    log.debug('Strix error')
    return {'status': Status.ERROR, 'message': strix_stdout}


def wrapper_file(strix_path, file, timeout=None):
    raise NotImplementedError()


def input_from_dict(problem: dict):
    parenthesized_guarantees = [
        f'({guarantee})' for guarantee in problem['guarantees']
    ]
    formula_str = '&'.join(parenthesized_guarantees)
    ins_str = ','.join(problem['inputs'])
    outs_str = ','.join(problem['outputs'])
    return formula_str, ins_str, outs_str


def add_parser_args(parser):
    parser.add_argument('--strix-bin',
                        type=strix_bin_path,
                        default='/strix/bin/strix',
                        metavar='path',
                        help='path to Strix binary')
    parser.add_argument('--strix-no-auto',
                        action='store_false',
                        dest='strix_auto',
                        help='no additional minimization of Mealy machine')
    parser.add_argument('--strix-timeout',
                        type=float,
                        default=10.0,
                        metavar='timeout',
                        help='Strix timeout')


def strix_bin_path(path: str):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f'{path} is not a Strix binary')
    return path
