import copy
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import flutes
from pycparser.c_generator import CGenerator

from asdl.asdl import ASDLGrammar, ASDLProduction, Field
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField
from asdl.transition_system import Action, ApplyRuleAction, GenTokenAction, TransitionSystem
from common.registerable import Registrable
from components.action_info import ActionInfo
from .c_utils import CLexer, asdl_ast_to_c_ast

__all__ = [
    "CTransitionSystem",
]


@Registrable.register('c')
class CTransitionSystem(TransitionSystem):
    grammar: ASDLGrammar

    def __init__(self, grammar: ASDLGrammar):
        super().__init__(grammar)
        self.lexer = CLexer()
        self.generator = CGenerator()

    def tokenize_code(self, code: str, mode=None) -> List[str]:
        return self.lexer.lex(code)

    def surface_code_to_ast(self, code: str):
        raise TypeError("Cannot convert from surface code to AST for C code")

    def ast_to_surface_code(self, asdl_ast: AbstractSyntaxTree) -> str:
        c_ast = asdl_ast_to_c_ast(asdl_ast, self.grammar)
        code = self.generator.visit(c_ast)
        return code

    def compare_ast(self, hyp_ast: AbstractSyntaxTree, ref_ast: AbstractSyntaxTree) -> bool:
        hyp_code = self.ast_to_surface_code(hyp_ast)
        ref_reformatted_code = self.ast_to_surface_code(ref_ast)

        ref_code_tokens = self.tokenize_code(ref_reformatted_code)
        hyp_code_tokens = self.tokenize_code(hyp_code)

        return ref_code_tokens == hyp_code_tokens

    def get_primitive_field_actions(self, realized_field: RealizedField) -> List[Action]:
        actions: List[Action] = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':  # expr -> Global(identifier* names)
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            tokens = []
            if realized_field.type.name == 'string':
                for field_val in field_values:
                    tokens.extend(field_val.split(' ') + ['</primitive>'])
            else:
                for field_val in field_values:
                    tokens.append(field_val)

            for tok in tokens:
                actions.append(GenTokenAction(tok))

        return actions

    def is_valid_hypothesis(self, hyp, **kwargs):
        # We don't know whether it's valid; just assume it is.
        return True

    def compress_actions(self, action_infos: List[ActionInfo]) -> List[ActionInfo]:
        compressed_infos = []
        for action_info in action_infos:
            compressed_info = copy.copy(action_info)
            if action_info.frontier_prod is not None:
                compressed_info.frontier_prod = self.grammar.prod2id[action_info.frontier_prod]
            if action_info.frontier_field is not None:
                compressed_info.frontier_field = self.grammar.field2id[action_info.frontier_field]
            if isinstance(action_info.action, ApplyRuleAction):
                compressed_info.action = ApplyRuleAction(self.grammar.prod2id[action_info.action.production])
            compressed_infos.append(compressed_info)
        return compressed_infos

    @lru_cache(maxsize=None)
    def field_map(self, prod: ASDLProduction) -> Dict[Field, int]:
        fields = {}
        for idx, field in enumerate(prod.fields):
            fields[field] = idx
        return fields

    CompressedAST = Tuple[int, List[Any]]  # (prod_id, (fields...))

    def compress_ast(self, ast: AbstractSyntaxTree) -> CompressedAST:
        field_map = self.field_map(ast.production)
        fields: List[Any] = [None] * len(field_map)
        for field in ast.fields:
            value = flutes.map_structure(
                lambda x: self.compress_ast(x) if isinstance(x, AbstractSyntaxTree) else x,
                field.value)
            fields[field_map[field.field]] = value
        comp_ast = (self.grammar.prod2id[ast.production], fields)
        return comp_ast
