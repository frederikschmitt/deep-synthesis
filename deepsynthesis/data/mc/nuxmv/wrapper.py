"""Wrapper for model checking script from Strix that uses the nuXmv model checker"""

import enum
import logging
import os
import subprocess

import ray

from deepsynthesis.data import ltl


class Status(enum.Enum):
    SATISFIED = 'satisfied'
    VIOLATED = 'violated'
    INVALID = 'invalid'
    TIMEOUT = 'timeout'
    ERROR = 'error'


@ray.remote
def worker(spec: dict, circuit: str, realizable: bool, strix_path, temp_dir,
           timeout):
    return wrapper_dict(spec, circuit, realizable, strix_path, temp_dir,
                        timeout)


def wrapper_dict(speficiation: dict,
                 circuit: str,
                 realizable: bool,
                 strix_path,
                 temp_dir,
                 timeout=10):
    spec_obj = ltl.specification.from_dict(speficiation)
    return wrapper(spec_obj, circuit, realizable, strix_path, temp_dir, timeout)


def wrapper(specification,
            circuit: str,
            realizable: bool,
            strix_path,
            temp_dir,
            timeout=10):
    temp_dir = os.path.join(temp_dir, 'nuxmv')
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    specification.to_file(temp_dir, 'specification.tlsf', format='tlsf')
    circuit_filepath = os.path.join(temp_dir, 'circuit.aag')
    with open(circuit_filepath, 'w') as aiger_file:
        aiger_file.write(circuit)
    verfiy_script_path = os.path.join(strix_path, 'scripts/verify.sh')
    try:
        args = [
            verfiy_script_path, circuit_filepath,
            os.path.join(temp_dir, 'specification.tlsf')
        ]
        if realizable:
            args.append('REALIZABLE')
        else:
            args.append('UNREALIZABLE')
        args.append(str(timeout))
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.debug('subprocess timeout')
        return Status.TIMEOUT
    except subprocess.CalledProcessError:
        logging.error('subprocess called process error')
        return Status.ERROR
    except Exception as error:
        logging.critical(error)
    out = result.stdout.decode('utf-8')
    err = result.stderr.decode('utf-8')
    if out == 'SUCCESS\n':
        return Status.SATISFIED
    if out == 'FAILURE\n':
        return Status.VIOLATED
    if err.startswith('error: cannot read implementation file'):
        return Status.INVALID
    print(out)
    print(err)
    return Status.ERROR
