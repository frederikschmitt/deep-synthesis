"""LTL specification encoder"""

from deepsynthesis.data.ltl import encoder


class TreeEncoder(encoder.TreeEncoder):

    def encode(self, spec):
        return super().encode(spec.formula)