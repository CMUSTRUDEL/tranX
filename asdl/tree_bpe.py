import copy
import functools
import pickle
import sys
from collections import Counter
from typing import Callable, Counter as CounterT, Dict, List, NamedTuple, Optional, Union

from typing_extensions import Literal

from .asdl import ASDLConstructor, ASDLGrammar, ASDLProduction, Field as ASDLField
from .asdl_ast import CompressedAST

__all__ = [
    "AST",

    "TreeIndex",
    "Value",
    "Size",

    "Terminal",
    "NonTerminal",
    "SingleValue",
    "MultipleValue",
    "Subtree",
    "Field",
    "Idiom",

    "TreeBPEMixin",
    "TreeBPE",
]


class AST:
    # Dummy class for type annotation & type checking purposes
    Leaf = str
    NonLeaf = tuple

    MultipleField = list

    @staticmethod
    def is_non_leaf(node: CompressedAST) -> bool:
        return isinstance(node, AST.NonLeaf)

    @staticmethod
    def is_leaf(node: CompressedAST) -> bool:
        return isinstance(node, AST.Leaf)

    @staticmethod
    def is_multiple_field(node: CompressedAST) -> bool:
        return isinstance(node, AST.MultipleField)


# Represents a parent--child pair. Possible values for child node:
class TreeIndex:
    # 1. child_value=None,    value_index=None
    #      An optional field with no value.
    # 2. child_value=<str>,   value_index=None
    #      A single field with literal value `child_value`.
    # 3. child_value=<str>,   value_index=<int>
    #      A multiple field, value at `value_index` is a literal value `child_value`.
    # 4. child_value=<int>,   value_index=None
    #      A single field with a subtree of production ID `child_value`.
    # 5. child_value=<int>,   value_index=<int>
    #      A multiple field, value at `value_index` is a subtree of production ID `child_value`.
    class Value(NamedTuple):  # pair of (parent production, child literal value or None)
        prod_id: int  # ID of the production rule (parent node)
        field_index: int  # index of the field within the production rule
        child_value: Union[int, str, None]  # ID of the production rule/literal value in the child node
        value_index: Optional[int] = None  # index within a multi-value field

    # 6. field_size=<int>
    #      A multiple field with number of children fixed to `field_size`.
    class Size(NamedTuple):  # pair of (parent production, specific field size)
        prod_id: int  # ID of the production rule (parent node)
        field_index: int  # index of the field within the production rule
        field_size: int  # size of the chosen multiple field

    t = Union[Value, Size]


# For some reason `pickle` decides that it only finds top-level classes in the module.
Value = TreeIndex.Value
Size = TreeIndex.Size


class Terminal(NamedTuple):
    value: str


class NonTerminal(NamedTuple):
    prod_id: int
    children: Dict[int, 'FieldValue']  # (field_idx) -> value


class SingleValue(NamedTuple):  # for single & optional fields
    value: 'Subtree'


class MultipleValue(NamedTuple):  # for multiple fields
    value: Dict[int, 'Subtree']
    size: Optional[int] = None  # the number of elements in the field, or None if not determined

    @property
    def fully_filled(self) -> bool:
        return self.size == len(self.value)


FieldValue = Union[SingleValue, MultipleValue]
Subtree = Union[Terminal, NonTerminal]


class Field(NamedTuple):
    class FieldIndex(int): pass

    class ValueIndex(int): pass

    name: str  # concat'ed name from production-field paths
    cardinality: Literal['single', 'optional', 'multiple']
    type: int  # type ID in ASDL grammar

    # Path leading to the field in the subtree consisting only of original production rules.
    # This is used to generate the idiom `subtree`, for a clearer representation of the idiom.
    original_index: List[Union[FieldIndex, ValueIndex]]


class Idiom(NamedTuple):
    id: int
    name: str
    subtree: NonTerminal  # subtree containing only built-in idioms (productions)
    tree_index: Optional[TreeIndex.t]
    fields: List[Field]
    child_field_range: slice  # range of fields that belonged to children before applying the idiom


class TreeBPEMixin:
    @staticmethod
    def _get_ast_value(ast: Optional[CompressedAST]) -> Union[str, int, None]:
        if ast is None or AST.is_leaf(ast):
            return ast
        return ast[0]

    def _update_subtree(self, ast: CompressedAST, delta: int = 1) -> None:
        raise ValueError

    _current_idiom: 'Idiom'

    def _replace_idiom(self, ast: CompressedAST) -> int:
        # Only the new production ID is returned; the fields are modified in-place.
        prod_id, fields = ast
        index = self._current_idiom.tree_index
        # The production ID doesn't match; return.
        if prod_id != index.prod_id:
            return prod_id
        # Check if required fields match children values.
        child_field = fields[index.field_index]
        if isinstance(index, TreeIndex.Value):
            if index.value_index is not None:
                # This must be a multiple field.
                if 0 <= index.value_index < len(child_field) or -len(child_field) <= index.value_index < 0:
                    child_field = child_field[index.value_index]
                else:
                    child_field = None
            subtree_match = (self._get_ast_value(child_field) == index.child_value)
        else:  # TreeIndex.Size
            subtree_match = (len(child_field) == index.field_size)
        if not subtree_match:
            return prod_id

        # If the current AST matches the replacement subtree...
        # Decrement counters for all parent-child pairs.
        self._update_subtree(ast, -1)
        if isinstance(index, TreeIndex.Value):
            # Collapse the child node (if exists) and move its fields to be fields of the current node.
            if isinstance(index.child_value, int):
                child_fields = child_field[1]
                # Decrement counters for the child node to collapse.
                self._update_subtree(child_field, -1)
            else:
                child_fields = []
            if index.value_index is None:
                fields[index.field_index:(index.field_index + 1)] = child_fields
            else:
                # Remove element in the original field with cardinality='multiple'.
                del fields[index.field_index][index.value_index]
                fields[(index.field_index + 1):(index.field_index + 1)] = child_fields
        else:  # TreeValue.Size
            # Expand "multiple field" into multiple direct fields.
            fields[index.field_index:(index.field_index + 1)] = child_field
        # Increment counters for new parent-child pairs.
        new_prod_id = self._current_idiom.id
        self._update_subtree((new_prod_id, fields))
        return new_prod_id

    def _revert_idiom(self, ast: CompressedAST) -> int:
        # Only the new production ID is returned; the fields are modified in-place.
        prod_id, fields = ast
        if prod_id != self._current_idiom.id:
            return prod_id

        # Decrement counters for all parent-child pairs.
        self._update_subtree(ast, -1)
        index = self._current_idiom.tree_index
        child_field_slice = self._current_idiom.child_field_range
        child_fields = fields[child_field_slice]
        assert len(child_fields) == child_field_slice.stop - child_field_slice.start
        if isinstance(index, TreeIndex.Value):
            # The children fields are combined into a subtree...
            if isinstance(index.child_value, int):
                child_ast = (index.child_value, child_fields)
                self._update_subtree(child_ast)
            else:
                child_ast = index.child_value
            if index.value_index is None:
                # ...and is replaced with the subtree.
                fields[child_field_slice] = [child_ast]
            else:
                # ...and is removed. The subtree is inserted into a multiple field.
                if index.value_index >= 0:
                    value_index = index.value_index
                else:
                    # `list.insert` inserts the value *before* the position.
                    value_index = len(fields[index.field_index]) + 1 + index.value_index
                fields[index.field_index].insert(value_index, child_ast)
                del fields[child_field_slice]
        else:  # TreeIndex.Size
            # The children fields are combined into a multiple field.
            assert len(child_fields) == index.field_size
            fields[child_field_slice] = [child_fields]
        # Increment counters for new parent-child pairs.
        new_prod_id = index.prod_id
        self._update_subtree((new_prod_id, fields))
        return new_prod_id

    @functools.lru_cache()
    def _traverse(self, func: Callable[[CompressedAST], int]) -> Callable[[CompressedAST], int]:
        def traverse_fn(ast: CompressedAST) -> int:
            new_prod_id = func(ast)
            fields = ast[1]
            # Recurse on each child.
            for field_idx, field in enumerate(fields):
                if AST.is_multiple_field(field):
                    for value_idx, value in enumerate(field):
                        if AST.is_non_leaf(value):
                            value_prod_id = traverse_fn(value)
                            if value_prod_id != value[0]:
                                field[value_idx] = (value_prod_id, value[1])
                else:
                    if AST.is_non_leaf(field):
                        field_prod_id = traverse_fn(field)
                        if field_prod_id != field[0]:
                            fields[field_idx] = (field_prod_id, field[1])
            return new_prod_id

        return traverse_fn


class CustomUnpickler(pickle.Unpickler):
    # Everything dumped should be classes within this module.

    def __init__(self, file):
        super().__init__(file)
        self._module = sys.modules[__name__]

    def find_class(self, module: str, name: str):
        if hasattr(self._module, name):
            return getattr(self._module, name)
        return super().find_class(module, name)


class TreeBPE(TreeBPEMixin):
    r"""The user-facing class for TreeBPE. A :class:`TreeBPE` instance represents a trained TreeBPE model."""

    count: CounterT[int]

    def __init__(self, idioms: List['Idiom'], revert_ids: List[int]):
        self.idioms = idioms.copy()
        self.revert_ids = sorted(revert_ids, reverse=True)
        start_idx = next(idx for idx, idiom in enumerate(idioms) if idiom.tree_index is not None)
        revert_id_set = set(revert_ids)
        end_idx = next(idx for idx in range(len(self.idioms), start_idx, -1) if idx - 1 not in revert_id_set)

        self._apply_idioms = [idioms[idx] for idx in range(start_idx, end_idx)]
        self._revert_idioms = [idioms[idx] for idx in self.revert_ids if start_idx <= idx < end_idx]

        self.apply_idiom = self._traverse(self._replace_idiom)
        self.revert_idiom = self._traverse(self._revert_idiom)

    def __getstate__(self):
        # Don't store the functions; they're not pickle-able.
        return self.idioms, self.revert_ids

    def __setstate__(self, state) -> None:
        idioms, revert_ids = state
        self.__init__(idioms, revert_ids)

    @staticmethod
    def load(path: str) -> 'TreeBPE':
        r"""Load a trained TreeBPE model from path."""
        with open(path, "rb") as f:
            return CustomUnpickler(f).load()

    def save(self, path: str) -> None:
        r"""Save the trained TreeBPE model to path."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _update_subtree(self, ast: CompressedAST, delta: int = 1) -> None:
        self.count[ast[0]] += delta

    def _count_nodes(self, ast: CompressedAST) -> None:
        self.count[ast[0]] += 1
        for field in ast[1]:
            values = field if AST.is_multiple_field(field) else [field]
            for value in values:
                if AST.is_non_leaf(value):
                    self._count_nodes(value)

    def encode(self, ast: CompressedAST) -> CompressedAST:
        r"""Encode a compressed AST using rules for this TreeBPE model."""
        self.count = Counter()
        self._count_nodes(ast)
        prod_id, fields = copy.deepcopy(ast)
        for idiom in self._apply_idioms:
            if self.count[idiom.tree_index.prod_id] > 0:
                self._current_idiom = idiom
                prod_id = self.apply_idiom((prod_id, fields))
        for idiom in self._revert_idioms:
            if self.count[idiom.id] > 0:
                self._current_idiom = idiom
                prod_id = self.revert_idiom((prod_id, fields))
        return prod_id, fields

    def decode(self, ast: CompressedAST) -> CompressedAST:
        r"""Decoded an encoded AST using rules for this TreeBPE model."""
        self.count = Counter()
        self._count_nodes(ast)
        prod_id, fields = copy.deepcopy(ast)
        for idiom in reversed(self._apply_idioms):
            if self.count[idiom.id] > 0:
                self._current_idiom = idiom
                prod_id = self.revert_idiom((prod_id, fields))
        return prod_id, fields

    def patch_grammar(self, grammar: ASDLGrammar) -> ASDLGrammar:
        r"""Add learned rules to an existing ASDL grammar."""
        productions = grammar.productions.copy()
        for idiom in self._apply_idioms:
            # Reverted idioms still need to be added, otherwise the production IDs will be incorrect.
            fields = [ASDLField(field.name, grammar.id2type[field.type], field.cardinality)
                      for field in idiom.fields]
            constructor = ASDLConstructor(idiom.name, fields)
            assert len(productions) == idiom.id
            productions.append(ASDLProduction(productions[idiom.tree_index.prod_id].type, constructor))
        new_grammar = ASDLGrammar(productions, preserve_order=True)
        new_grammar.root_type = grammar.root_type
        return new_grammar
