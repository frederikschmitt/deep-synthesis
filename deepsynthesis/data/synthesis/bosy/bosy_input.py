"""BoSy LTL specification"""

import json
import ntpath

import deepsynthesis.data.ltl as ltl


def from_file(filepath):
    with open(filepath, 'r') as spec_file:
        spec_dict = json.loads(spec_file.read())
    spec_dict['name'] = ntpath.basename(filepath)
    return ltl.specification.from_dict(spec_dict)


BOSY_INPUT_TEMPL = """{{
    "semantics": "{semantics}",
    "inputs": [{inputs}],
    "outputs": [{outputs}],
    "assumptions": [{assumptions}],
    "guarantees": [{guarantees}]
}}"""


def format_bosy_input(inputs, outputs, guarantees, bosy_semantics="mealy"):
    if inputs == []:
        # BoSy requires at least one input
        # TODO report BoSy bug
        inputs = ['i_default']
    inputs_str = ",".join([f'"{i}"' for i in inputs])
    outputs_str = ",".join([f'"{i}"' for i in outputs])
    guarantees_str = ',\n'.join([f'"{guarantee}"' for guarantee in guarantees])
    return BOSY_INPUT_TEMPL.format(semantics=bosy_semantics,
                                   inputs=inputs_str,
                                   outputs=outputs_str,
                                   assumptions='',
                                   guarantees=f'{guarantees_str}')
