"""LTL encoder"""

import logging

from deepsynthesis.data import encoder
from deepsynthesis.data.expression import Notation
from deepsynthesis.data.ltl.lexer import lex
from deepsynthesis.data.ltl.parser import parse_prefix, parse_infix


class SequenceEncoder(encoder.sequence.Encoder):

    @property
    def formula(self):
        return self.sequence

    def lex(self):
        self.tokens = lex(self.formula)
        success = self.tokens is not None
        if not success:
            self.error = 'Lex formula'
        return success

    def vocabulary_filename(self):
        return 'ltl-vocab' + super().vocabulary_filename()


class TreeEncoder(encoder.expression.Encoder):

    @property
    def formula(self):
        return self.expression

    def lex(self):
        self.tokens = lex(self.formula)
        success = self.tokens is not None
        if not success:
            self.error = 'Lex formula'
        return success

    def parse(self):
        if self.notation == Notation.PREFIX:
            self.ast = parse_prefix(self.formula)
        elif self.notation == Notation.INFIX:
            self.ast = parse_infix(self.formula)
        else:
            logging.critical('Unsupported notation %s', self.notation)
        success = self.ast is not None
        if not success:
            self.error = 'Parse formula'
        return success

    def vocabulary_filename(self):
        return 'ltl-vocab' + super().vocabulary_filename()
