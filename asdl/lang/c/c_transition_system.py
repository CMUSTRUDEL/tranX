from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import flutes
import sentencepiece as spm
from pycparser.c_ast import Node as ASTNode
from pycparser.c_generator import CGenerator

from asdl.asdl import ASDLGrammar, ASDLProduction, ASDLType, Field
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField
from asdl.hypothesis import Hypothesis
from asdl.lang.c.c_utils import CLexer, SPM_SPACE, asdl_ast_to_c_ast
from asdl.transition_system import Action, ApplyRuleAction, GenTokenAction, ReduceAction, TransitionSystem
from common.registerable import Registrable

__all__ = [
    "RobustCGenerator",
    "CompressedAST",
    "CTransitionSystem",
    "CHypothesis",
    "CGenTokenAction",
]

CompressedAST = Tuple[int, List[Any]]  # (prod_id, (fields...))

T = TypeVar('T')


def _(a: Optional[T], b: T) -> T:
    return a if a is not None else b


class RobustCGenerator(CGenerator):
    @classmethod
    def _replace_none(cls, node: ASTNode) -> None:
        for child in node:
            if child is not None and child != CTransitionSystem.UNFILLED:
                cls._replace_none(child)
        for key in node.__slots__:
            if key.startswith("_"): continue
            value = getattr(node, key)
            if value == CTransitionSystem.UNFILLED:
                setattr(node, key, f"<{node.__class__.__name__}.{key}>")
            elif isinstance(value, list):
                for idx, val in enumerate(value):
                    if val == CTransitionSystem.UNFILLED:
                        value[idx] = f"<{node.__class__.__name__}.{key}[{idx}]>"

    def generate_code(self, node: ASTNode) -> str:
        self._replace_none(node)
        return super().visit(node)

    def visit(self, node):
        if node is None:
            return "<None>"
        if isinstance(node, str):
            return node
        return super().visit(node)

    def visit_IdentifierType(self, n):
        return ' '.join(_(name, "<ID>") for name in n.names)


@Registrable.register('c')
class CTransitionSystem(TransitionSystem):
    grammar: ASDLGrammar

    UNFILLED = "@unfilled@"

    def __init__(self, grammar: ASDLGrammar, spm_model: Optional[spm.SentencePieceProcessor] = None):
        super().__init__(grammar)
        self.lexer = CLexer()
        self.generator = RobustCGenerator()
        self.sp = spm_model

    def tokenize_code(self, code: str, mode=None) -> List[str]:
        return self.lexer.lex(code)

    def surface_code_to_ast(self, code: str):
        raise TypeError("Cannot convert from surface code to AST for C code")

    def ast_to_surface_code(self, asdl_ast: AbstractSyntaxTree) -> str:
        c_ast = asdl_ast_to_c_ast(asdl_ast, self.grammar, ignore_error=True)
        code = self.generator.generate_code(c_ast)
        return code

    def compare_ast(self, hyp_ast: AbstractSyntaxTree, ref_ast: AbstractSyntaxTree) -> bool:
        hyp_code = self.ast_to_surface_code(hyp_ast)
        ref_reformatted_code = self.ast_to_surface_code(ref_ast)

        ref_code_tokens = self.tokenize_code(ref_reformatted_code)
        hyp_code_tokens = self.tokenize_code(hyp_code)

        return ref_code_tokens == hyp_code_tokens

    def _tokenize(self, value: str) -> List[str]:
        if self.sp is not None:
            return self.sp.EncodeAsPieces(value)
        return value.split(' ')

    def _get_primitive_field_actions(self, values: List[str]) -> List[Action]:
        actions: List[Action] = []
        for field_val in values:
            for token in self._tokenize(field_val):
                actions.append(CGenTokenAction(token))
            actions.append(CGenTokenAction(CGenTokenAction.STOP_SIGNAL))
        return actions

    def get_primitive_field_actions(self, realized_field: RealizedField) -> List[Action]:
        return self._get_primitive_field_actions(realized_field.as_value_list)

    def is_valid_hypothesis(self, hyp, **kwargs):
        # We don't know whether it's valid; just assume it is.
        return True

    @lru_cache(maxsize=None)
    def field_id_map(self, prod: ASDLProduction) -> Dict[Field, int]:
        fields = {}
        for idx, field in enumerate(prod.fields):
            fields[field] = idx
        return fields

    def compress_ast(self, ast: AbstractSyntaxTree) -> CompressedAST:
        field_map = self.field_id_map(ast.production)
        fields: List[Any] = [None] * len(field_map)
        for field in ast.fields:
            value = flutes.map_structure(
                lambda x: self.compress_ast(x) if isinstance(x, AbstractSyntaxTree) else x,
                field.value)
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

    def _get_actions_from_compressed(self, asdl_ast: CompressedAST, actions: List[Action]) -> None:
        r"""Generate action sequence given the compressed ASDL Syntax Tree."""
        prod_id, fields = asdl_ast
        production = self.grammar.id2prod[prod_id]
        parent_action = ApplyRuleAction(production)
        actions.append(parent_action)
        assert len(fields) == len(production.fields)

        for field, field_value in zip(production.fields, fields):
            if field_value is not None:
                field_value = field_value if field.cardinality == 'multiple' else [field_value]
                if self.grammar.is_composite_type(field.type):  # is a composite field
                    for val in field_value:
                        self._get_actions_from_compressed(val, actions)
                else:  # is a primitive field
                    actions.extend(self._get_primitive_field_actions(field_value))

            # if an optional field is filled, then do not need Reduce action
            if field.cardinality == 'multiple' or (field.cardinality == 'optional' and field_value is None):
                actions.append(ReduceAction())

    def get_actions_from_compressed(self, asdl_ast: CompressedAST) -> List[Action]:
        r"""Generate action sequence given the compressed ASDL Syntax Tree."""
        actions = []
        self._get_actions_from_compressed(asdl_ast, actions)
        return actions


class CHypothesis(Hypothesis):
    def __init__(self, use_subword: bool = True):
        super().__init__()
        self._use_subword = use_subword

    def is_multiword_primitive_type(self, type: ASDLType) -> bool:
        # All primitive types (IDENT & LITERAL) are multi-word.
        return True

    def detokenize(self, tokens: List[str]) -> str:
        if self._use_subword:
            return "".join(tokens).lstrip(SPM_SPACE).replace(SPM_SPACE, " ")
        return " ".join(tokens)


class CGenTokenAction(GenTokenAction):
    STOP_SIGNAL = "@@end@@"
