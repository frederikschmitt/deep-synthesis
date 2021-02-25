"""Dataset of LTL Specifications"""

import logging
from statistics import mean
from statistics import median

from deepsynthesis.data.ltl.lexer import lex

logger = logging.getLogger(__name__)


class Dataset():

    def __init__(self, dataset):
        self.dataset = dataset
        logger.info('Successfully constructed dataset of %d LTL specifications',
                    len(self.dataset))

    def rename_aps(self, input_aps, output_aps, random=True, renaming=None):
        for specification in self.dataset:
            specification.rename_aps(input_aps, output_aps, random, renaming)
        logger.info(('Renamed input atomic propositions to %s and renamed '
                     'output atomic propositions to %s'), input_aps, output_aps)

    def print_stats(self):
        """Computes statistics of the dataset"""
        num_inputs = [spec.num_inputs for spec in self.dataset]
        num_outputs = [spec.num_outputs for spec in self.dataset]
        num_assumptions = [spec.num_assumptions for spec in self.dataset]
        num_guarantees = [spec.num_guarantees for spec in self.dataset]
        print(f'Computed statistics of {len(self.dataset)} specifications')
        for key, values in [('inputs', num_inputs), ('outputs', num_outputs),
                            ('assumptions', num_assumptions),
                            ('guarantees', num_guarantees)]:
            print(f'Number of {key}')
            print((f'minimum: {min(values)} maximum: {max(values)} '
                   f'mean: {mean(values)} median: {median(values)} '
                   f'total: {sum(values)}'))

    def guarantees(self, unique=False):
        """Returns a list of all (syntactically unique) guarantees including inputs and outputs that appear in the dataset"""
        result = []
        guarantees = set()
        for spec in self.dataset:
            for guarantee in spec.guarantees:
                if unique:
                    if guarantee in guarantees:
                        continue
                    guarantees.add(guarantee)
                #TODO move functionalty into specification class
                tokens = lex(guarantee)
                result.append({
                    'inputs': [i for i in spec.inputs if i in tokens],
                    'outputs': [o for o in spec.outputs if o in tokens],
                    'guarantee': guarantee
                })
        logger.info('Bundled %d %s guarantees', len(result),
                    "unique" if unique else "non-unique")
        return result

    def save(self, dir):
        for spec in self.dataset:
            spec.to_file(dir)
        logger.info('Saved %d files to %s', len(self.dataset), dir)
