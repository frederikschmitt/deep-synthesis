from deepsynthesis.data.tree import Tree
from deepsynthesis.data.ltl.lexer import lex


def from_str(formula_str, format='infix'):
    tokens = Lexer().tokenize(formula_str)
    if format == 'infix':
        from deepsynthesis.data.ltl.parser import InfixParser
        return InfixParser().parse(tokens)
    elif format == 'prefix':
        from deepsynthesis.data.ltl.parser import PrefixParser
        return PrefixParser().parse(tokens)
    else:
        raise ValueError("Invalid format")


def tokenize(formula_str):
    tokens = []
    for token in Lexer().tokenize(formula_str):
        tokens.append(token.value)
    return tokens

