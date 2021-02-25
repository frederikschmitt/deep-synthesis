"""AIGER circuit encoder"""

import tensorflow as tf

from deepsynthesis.data import encoder
from deepsynthesis.data.aiger.aiger import parse, parse_no_header, Symbol

NEWLINE_TOKEN = '<n>'
COMPLEMENT_TOKEN = '<c>'
LATCH_TOKEN = '<l>'
REALIZABLE_TOKEN = '<r>'
UNREALIZABLE_TOKEN = '<u>'


class SequenceEncoder(encoder.sequence.Encoder):

    def __init__(self,
                 start,
                 eos,
                 pad,
                 components=None,
                 encode_start=True,
                 encode_realizable=False,
                 inputs=None,
                 outputs=None,
                 vocabulary=None,
                 tf_dtype=tf.int32):
        """
            inputs, outputs: only required if components does not contain header or symbols
        """
        self.components = components if components else [
            'inputs', 'latches', 'outputs', 'ands'
        ]
        self.encode_realizable = encode_realizable
        self.inputs = inputs
        self.outputs = outputs
        self.realizable = True
        super().__init__(start, eos, pad, encode_start, vocabulary, tf_dtype)

    @property
    def circuit(self):
        return self.sequence

    def encode(self, sequence):
        self.realizable = 'i0 i0' in sequence
        return super().encode(sequence)

    def lex(self):
        self.tokens = []

        if self.encode_realizable:
            if self.realizable:
                self.tokens.append(REALIZABLE_TOKEN)
            else:
                self.tokens.append(UNREALIZABLE_TOKEN)

        try:
            aiger = parse(self.circuit)
        except ValueError as err:
            self.error = err
            return False

        for component in self.components:
            if component == 'header':
                header_ints = str(aiger.header).split(' ')[1:]
                self.tokens.extend(header_ints)
                self.tokens.append(NEWLINE_TOKEN)
            else:
                for elem in getattr(aiger, component):
                    str_lits = str(elem).split(' ')
                    self.tokens.extend(str_lits)
                    self.tokens.append(NEWLINE_TOKEN)
        #remove last newline token
        self.tokens = self.tokens[:-1]

        return True

    def decode(self, ids, realizable=True):
        success = super().decode(ids)
        components = list(self.components)

        if self.encode_realizable:
            realizable_token = self.sequence[:3]
            self.sequence = self.sequence[4:]
            if realizable_token not in (REALIZABLE_TOKEN, UNREALIZABLE_TOKEN):
                self.error = 'First token not realizable token'
                return False
            else:
                if realizable_token == REALIZABLE_TOKEN:
                    self.realizable = True
                else:
                    self.realizable = False

        if 'header' not in self.components or 'symbols' not in self.components:
            num_inputs = len(self.inputs)
            num_outputs = len(self.outputs)

        self.sequence = self.sequence.replace(NEWLINE_TOKEN, '\n')
        self.sequence = self.sequence.replace(' \n ', '\n')
        if 'header' not in self.components:
            try:
                aiger = parse_no_header(self.sequence, num_inputs, num_outputs,
                                        components)
            except ValueError as error:
                self.error = error
                return False
        else:
            self.sequence = 'aag ' + self.sequence
            try:
                aiger = parse(self.sequence, components)
            except ValueError as error:
                self.error = error
                return False
        if 'symbols' not in self.components:
            symbols = [
                Symbol('i', i, self.inputs[i] if self.realizable else self.outputs[i]) for i in range(num_inputs)
            ]
            symbols.extend(
                [Symbol('l', i, f'l{i}') for i in range(aiger.num_latches)])
            symbols.extend(
                [Symbol('o', i, self.outputs[i] if self.realizable else self.inputs[i]) for i in range(num_outputs)])
            aiger.symbols = symbols
        self.sequence = str(aiger)
        return success

    def sort_tokens(self, tokens):
        tokens.sort()
        tokens.sort(key=len)

    def vocabulary_filename(self):
        return 'aiger-vocab' + super().vocabulary_filename()
