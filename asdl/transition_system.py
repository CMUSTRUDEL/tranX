# coding=utf-8
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Type, cast

from .asdl import ASDLGrammar, ASDLProduction, Field
from .asdl_ast import AbstractSyntaxTree, CompressedAST, RealizedField

if TYPE_CHECKING:
    from .hypothesis import Hypothesis

__all__ = [
    "Action",
    "ApplyRuleAction",
    "GenTokenAction",
    "ReduceAction",
    "TransitionSystem",
]


class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production: ASDLProduction):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()


class GenTokenAction(Action):
    STOP_SIGNAL = "</primitive>"

    def __init__(self, token: str):
        self.token = token

    def is_stop_signal(self):
        return self.token == self.STOP_SIGNAL

    def __repr__(self):
        return 'GenToken[%s]' % self.token


class ReduceAction(Action):
    def __repr__(self):
        return 'Reduce'


@lru_cache(maxsize=None)
def _field_id_map(prod: ASDLProduction) -> Dict[Field, int]:
    fields = {}
    for idx, field in enumerate(prod.fields):
        fields[field] = idx
    return fields


class TransitionSystem(object):
    UNFILLED = "@unfilled@"

    def __init__(self, grammar: ASDLGrammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast: AbstractSyntaxTree) -> List[Action]:
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions: List[Action] = []

        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            # is a composite field
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions = self.get_actions(cast(AbstractSyntaxTree, field.value))
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions = self.get_actions(cast(AbstractSyntaxTree, val))
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions = self.get_actions(cast(AbstractSyntaxTree, field.value))

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        field_actions.append(ReduceAction())
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)

                # if an optional field is filled, then do not need Reduce action
                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    # reduce action
                    field_actions.append(ReduceAction())

            actions.extend(field_actions)

        return actions

    def tokenize_code(self, code: str, mode) -> List[str]:
        raise NotImplementedError

    def compare_ast(self, hyp_ast: AbstractSyntaxTree, ref_ast: AbstractSyntaxTree) -> bool:
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast: AbstractSyntaxTree) -> str:
        raise NotImplementedError

    def surface_code_to_ast(self, code: str) -> AbstractSyntaxTree:
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field: RealizedField) -> List[Action]:
        raise NotImplementedError

    def compress_ast(self, ast: AbstractSyntaxTree) -> CompressedAST:
        if ast is None or isinstance(ast, str):
            return ast
        field_map = _field_id_map(ast.production)
        fields: List[Any] = [None] * len(field_map)
        for field in ast.fields:
            if isinstance(field.value, list):
                value = [self.compress_ast(value) for value in field.value]
            else:
                value = self.compress_ast(field.value)
            fields[field_map[field.field]] = value
        comp_ast = (self.grammar.prod2id[ast.production], fields)
        return comp_ast

    def decompress_ast(self, ast: CompressedAST) -> AbstractSyntaxTree:
        if ast is None or ast == self.UNFILLED:
            return ast
        prod_id, fields = ast
        node = AbstractSyntaxTree(self.grammar.id2prod[prod_id])
        for field, value in zip(node.fields, fields):
            if value is not None:
                value = value if isinstance(value, list) else [value]
                if self.grammar.is_composite_type(field.type):
                    for val in value:
                        field.add_value(self.decompress_ast(val))
                else:
                    for val in value:
                        field.add_value(val)
        return node

    def get_valid_continuation_types(self, hyp: 'Hypothesis') -> Sequence[Type[Action]]:
        if hyp.tree:
            assert hyp.frontier_field is not None
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp: 'Hypothesis') -> List[ASDLProduction]:
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]

    @staticmethod
    def get_class_by_lang(lang: str) -> Type['TransitionSystem']:
        if lang == 'python':
            from .lang.py.py_transition_system import PythonTransitionSystem
            return PythonTransitionSystem
        elif lang == 'python3':
            from .lang.py3.py3_transition_system import Python3TransitionSystem
            return Python3TransitionSystem
        elif lang == 'lambda_dcs':
            from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem
            return LambdaCalculusTransitionSystem
        elif lang == 'prolog':
            from .lang.prolog.prolog_transition_system import PrologTransitionSystem
            return PrologTransitionSystem
        elif lang == 'wikisql':
            from .lang.sql.sql_transition_system import SqlTransitionSystem
            return SqlTransitionSystem
        elif lang == 'c':
            from .lang.c.c_transition_system import CTransitionSystem
            return CTransitionSystem

        raise ValueError('unknown language %s' % lang)
