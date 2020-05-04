import copy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import flutes
import sentencepiece as spm
from pycparser.c_generator import CGenerator

from asdl.asdl import ASDLGrammar, ASDLProduction, ASDLType, Field
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField
from asdl.transition_system import Action, ApplyRuleAction, GenTokenAction, TransitionSystem
from common.registerable import Registrable
from components.action_info import ActionInfo
from .c_utils import CLexer, asdl_ast_to_c_ast

__all__ = [
    "CTransitionSystem",
    "CHypothesis",
]

from ...hypothesis import Hypothesis


@Registrable.register('c')
class CTransitionSystem(TransitionSystem):
    grammar: ASDLGrammar

    def __init__(self, grammar: ASDLGrammar, spm_model: Optional[spm.SentencePieceProcessor] = None):
        super().__init__(grammar)
        self.lexer = CLexer()
        self.generator = CGenerator()
        self.sp = spm_model

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

    def _tokenize(self, value: str):
        if self.sp is not None:
            return self.sp.EncodeAsPieces(value)
        return value.split(' ')

    def get_primitive_field_actions(self, realized_field: RealizedField) -> List[Action]:
        actions: List[Action] = []
        for field_val in realized_field.as_value_list:
            for token in self._tokenize(field_val):
                actions.append(CGenTokenAction(token))
            actions.append(CGenTokenAction(CGenTokenAction.STOP_SIGNAL))
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


class CHypothesis(Hypothesis):
    SPM_SPACE = "▁"

    def __init__(self, use_subword: bool = True):
        super().__init__()
        self._use_subword = use_subword

    def is_multiword_primitive_type(self, type: ASDLType) -> bool:
        # All primitive types (IDENT & LITERAL) are multi-word.
        return True

    def detokenize(self, tokens: List[str]) -> str:
        if self._use_subword:
            return "".join(tokens).lstrip(self.SPM_SPACE).replace(self.SPM_SPACE, " ")
        return " ".join(tokens)


class CGenTokenAction(GenTokenAction):
    @property
    def stop_signal(self) -> str:
        return "@@end@@"
