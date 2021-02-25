"""LTL parser"""

import argparse
import sly

from deepsynthesis.data.ast import BinaryAST
from deepsynthesis.data.ltl.lexer import PrefixLexer, InfixLexer


class PrefixParser(sly.Parser):

    tokens = PrefixLexer.tokens
    precedence = (
        ('right', EQUIV, IMPL),
        ('left', XOR),
        ('left', OR),
        ('left', AND),
        ('right', UNTIL, WUNTIL, RELEASE),
        ('right', EVEN, GLOB),
        ('right', NEXT),
        ('right', NOT),
    )

    @_('EQUIV expr expr',
       'IMPL expr expr',
       'XOR expr expr',
       'OR expr expr',
       'AND expr expr',
       'UNTIL expr expr',
       'WUNTIL expr expr',
       'RELEASE expr expr')
    def expr(self, p):
        return BinaryAST(p[0], p.expr0, p.expr1)

    @_('EVEN expr',
       'GLOB expr',
       'NEXT expr',
       'NOT expr')
    def expr(self, p):
        return BinaryAST(p[0], p.expr)

    @_('CONST')
    def expr(self, p):
        return BinaryAST(p.CONST)

    @_('AP')
    def expr(self, p):
        return BinaryAST(p.AP)

    def error(self, p):
        pass


class InfixParser(sly.Parser):

    tokens = InfixLexer.tokens
    precedence = (
        ('right', EQUIV, IMPL),
        ('left', XOR),
        ('left', OR),
        ('left', AND),
        ('right', UNTIL, WUNTIL, RELEASE),
        ('right', EVEN, GLOB),
        ('right', NEXT),
        ('right', NOT),
    )

    @_('expr EQUIV expr',
       'expr IMPL expr',
       'expr XOR expr',
       'expr OR expr',
       'expr AND expr',
       'expr UNTIL expr',
       'expr WUNTIL expr',
       'expr RELEASE expr')
    def expr(self, p):
        return BinaryAST(p[1], p.expr0, p.expr1)

    @_('EVEN expr',
       'GLOB expr',
       'NEXT expr',
       'NOT expr')
    def expr(self, p):
        return BinaryAST(p[0], p.expr)

    @_('LPAR expr RPAR')
    def expr(self, p):
        return p.expr

    @_('CONST')
    def expr(self, p):
        return BinaryAST(p.CONST)

    @_('AP')
    def expr(self, p):
        return BinaryAST(p.AP)

    def error(self, p):
        pass


INFIX_LEXER = None
INFIX_PARSER = None
PREFIX_LEXER = None
PREFIX_PARSER = None


def parse_infix(formula: str):
    global INFIX_LEXER
    if INFIX_LEXER is None:
        INFIX_LEXER = InfixLexer()
    global INFIX_PARSER
    if INFIX_PARSER is None:
        INFIX_PARSER = InfixParser()
    return INFIX_PARSER.parse(INFIX_LEXER.tokenize(formula))


def parse_prefix(formula: str):
    global PREFIX_LEXER
    if PREFIX_LEXER is None:
        PREFIX_LEXER = PrefixLexer()
    global PREFIX_PARSER
    if PREFIX_PARSER is None:
        PREFIX_PARSER = PrefixParser()
    return PREFIX_PARSER.parse(PREFIX_LEXER.tokenize(formula))


def parse(formula: str):
    """
    Parses LTL formula
    Args:
        formula: string, in infix or prefix notation
    Returns:
        abstract syntax tree or None if formula can not be parsed
    """
    ast = parse_infix(formula)
    if ast:
        return ast
    return parse_prefix(formula)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description=
        'Parses LTL formulas in infix or prefix notation and prints out parse tree'
    )
    while True:
        formula = input('Formula: ')
        ast = parse(formula)
        print(ast)
