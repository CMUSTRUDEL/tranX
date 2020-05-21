import copy
import itertools
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Counter as CounterT, Dict, Iterator, List, NamedTuple, Optional, Tuple, TypeVar, Union

import flutes
from argtyped import Arguments
from pycparser.c_ast import Node as ASTNode
from termcolor import colored
from tqdm import tqdm
from typing_extensions import Literal

from asdl.asdl import ASDLGrammar
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CTransitionSystem, CompressedAST, RobustCGenerator
from datasets.c.build_dataset import RawExample

T = TypeVar('T')
MTuple = Tuple[T, ...]
MaybeDict = Union[T, Dict[int, T]]
NodeValue = Union[CompressedAST, str, List[CompressedAST]]


class Args(Arguments):
    data_dir: str = "tranx_data_new"  # path to `create_c_dataset.py` outputs
    output_dir: str = "tranx_data_new"  # path to output folder where generated data will be stored
    n_procs: int = 0  # number of worker processes to spawn
    num_idioms: int = 100  # number of idioms to extract

    max_examples: Optional[int]


def read_data(data_dir: str, verbose: bool = True) -> Iterator[CompressedAST]:
    files = [file for file in sorted(Path(data_dir).iterdir())
             if file.name.startswith("data_") and file.suffix == ".pkl"]
    for file in tqdm(files, desc="Reading file", disable=not verbose):
        with file.open("rb") as f:
            data: List[RawExample] = pickle.load(f)
        for example in data:
            yield example.ast


# Represents a parent--child pair. Possible values for child node:
# 1. child_value=None,    value_index=None
#      An optional field with no value.
# 2. child_value=<int>,   value_index=None
#      A single field with a subtree of production ID `child_value`.
# 3. child_value=<str>,   value_index=None
#      A single field with literal value `child_value`.
# 4. child_value=<tuple>, value_index=None
#      A multiple field fully-filled with values from `child_value`.
# 5. child_value=<int>,   value_index=<int>
#      A multiple field, value at `value_index` is a subtree of production ID `child_value`.
# 6. child_value=<str>,   value_index=<int>
#      A multiple field, value at `value_index` is a literal value `child_value`.
class SubtreeIndex(NamedTuple):
    prod_id: int  # ID of the production rule
    # ID of the production rule/literal value in the child node(s)
    child_value: Union[int, str, MTuple[int], MTuple[str], None]
    field_index: int  # index of the field within the production rule
    value_index: Optional[int] = None  # index within a multi-value field


class FindSubtreesState(flutes.PoolState):
    def __init__(self):
        sys.setrecursionlimit(32768)
        self.count: CounterT[SubtreeIndex] = Counter()

    @staticmethod
    def _get_ast_value(ast: Optional[CompressedAST]) -> Union[str, int, None]:
        if ast is None or isinstance(ast, str):
            return ast
        return ast[0]

    @classmethod
    def _get_subtree_indexes(cls, ast: CompressedAST) -> List[SubtreeIndex]:
        # Return subtree indexes for only the current layer.
        prod_id, fields = ast
        indexes = []
        for field_idx, field in enumerate(fields):
            if isinstance(field, list):
                child_values = tuple(cls._get_ast_value(value) for value in field)
                assert all(isinstance(x, int) for x in child_values) or all(isinstance(x, str) for x in child_values)
                # Fully-filled field.
                indexes.append(SubtreeIndex(prod_id=prod_id, child_value=child_values, field_index=field_idx))
                for value_idx, child_value in enumerate(child_values):
                    # Allow counting indices from both front and back.
                    indexes.append(SubtreeIndex(prod_id=prod_id, child_value=child_value,
                                                field_index=field_idx, value_index=value_idx))
                    indexes.append(SubtreeIndex(prod_id=prod_id, child_value=child_value,
                                                field_index=field_idx, value_index=-(len(field) - value_idx)))
            else:
                child_value = cls._get_ast_value(field)
                indexes.append(SubtreeIndex(prod_id=prod_id, child_value=child_value, field_index=field_idx))
        return indexes

    @flutes.exception_wrapper()
    def count_subtrees(self, ast: CompressedAST) -> None:
        prod_id, fields = ast
        indexes = self._get_subtree_indexes(ast)
        self.count.update(indexes)

        for field in fields:
            values = field if isinstance(field, list) else [field]
            for value in values:
                if isinstance(value, tuple):
                    self.count_subtrees(value)

    @flutes.exception_wrapper()
    def replace_subtree_index(self, ast: CompressedAST, replace_index: Tuple[int, SubtreeIndex]) -> CompressedAST:
        prod_id, fields = ast

        index = replace_index[1]
        new_prod_id = prod_id
        new_fields = fields.copy()
        if prod_id == index.prod_id:
            child_field = fields[index.field_index]
            if index.value_index is not None:
                if 0 <= index.value_index < len(child_field) or -len(child_field) <= index.value_index < 0:
                    child_field = child_field[index.value_index]
                else:
                    child_field = None
            if isinstance(index.child_value, tuple) and isinstance(child_field, list):
                child_value = tuple(self._get_ast_value(value) for value in child_field)
            else:
                child_value = self._get_ast_value(child_field)
            # If the current AST matches the replacement subtree...
            if child_value == index.child_value:
                # Decrement counters for all parent-child pairs.
                indexes = self._get_subtree_indexes((prod_id, fields))
                for idx in indexes:
                    self.count[idx] -= 1
                child_fields = []
                if isinstance(index.child_value, int):
                    collapse_nodes: List[CompressedAST] = [child_field]
                elif (isinstance(index.child_value, tuple) and
                      len(index.child_value) > 0 and isinstance(index.child_value[0], int)):
                    collapse_nodes = child_field
                else:
                    collapse_nodes = []
                for node in collapse_nodes:
                    child_fields.extend(node[1])
                    # Decrement counters for the child node to collapse.
                    indexes = self._get_subtree_indexes(node)
                    for idx in indexes:
                        self.count[idx] -= 1
                if index.value_index is None:
                    new_fields[index.field_index:(index.field_index + 1)] = child_fields
                else:
                    # Remove element in the original field with cardinality='multiple'.
                    new_fields[index.field_index] = new_fields[index.field_index].copy()
                    del new_fields[index.field_index][index.value_index]
                    new_fields[(index.field_index + 1):(index.field_index + 1)] = child_fields
                new_prod_id = replace_index[0]
                # Increment counters for new parent-child pairs.
                indexes = self._get_subtree_indexes((new_prod_id, new_fields))
                self.count.update(indexes)

        # Recurse on each child, update counters where necessary.
        for field_idx, field in enumerate(new_fields):
            if isinstance(field, list):
                new_field = field.copy()
                child_changed = False
                for value_idx, value in enumerate(field):
                    if isinstance(value, tuple):
                        new_value = self.replace_subtree_index(value, replace_index)
                        if new_value[0] != value[0]:
                            child_changed = True
                            for val_idx in [value_idx, -(len(field) - value_idx)]:
                                self.count[SubtreeIndex(prod_id=new_prod_id, child_value=value[0],
                                                        field_index=field_idx, value_index=val_idx)] -= 1
                                self.count[SubtreeIndex(prod_id=new_prod_id, child_value=new_value[0],
                                                        field_index=field_idx, value_index=val_idx)] += 1
                        new_field[value_idx] = new_value
                if child_changed:
                    self.count[SubtreeIndex(prod_id=new_prod_id, child_value=tuple(val[0] for val in field),
                                            field_index=field_idx)] -= 1
                    self.count[SubtreeIndex(prod_id=new_prod_id, child_value=tuple(val[0] for val in new_field),
                                            field_index=field_idx)] += 1
                new_fields[field_idx] = new_field
            else:
                if isinstance(field, tuple):
                    new_field = self.replace_subtree_index(field, replace_index)
                    if new_field[0] != field[0]:
                        self.count[SubtreeIndex(prod_id=new_prod_id, child_value=field[0],
                                                field_index=field_idx)] -= 1
                        self.count[SubtreeIndex(prod_id=new_prod_id, child_value=new_field[0],
                                                field_index=field_idx)] += 1
                    new_fields[field_idx] = new_field

        return new_prod_id, new_fields

    def replace_subtree(self, subtree: 'Subtree', ast: CompressedAST) -> CompressedAST:
        prod_id, fields = ast
        pass

    def __return_state__(self) -> CounterT[SubtreeIndex]:
        # Return the top 1% counts only. We assume the samples are randomly shuffled, so the counts in each split
        # should be proportional to split size.
        counter: CounterT[SubtreeIndex] = Counter()
        for key, count in self.count.most_common(len(self.count) // 100):
            counter[key] = count
        return counter


class Terminal(NamedTuple):
    value: str


class NonTerminal(NamedTuple):
    prod_id: int
    children: Dict[int, 'FieldValue']  # (field_idx) -> value


class SingleValue(NamedTuple):  # for single & optional fields
    value: 'Subtree'


class MultipleValue(NamedTuple):  # for multiple fields
    value: Dict[int, 'Subtree']
    fully_filled: bool = False  # whether the list of values is completely filled


FieldValue = Union[SingleValue, MultipleValue]
Subtree = Union[Terminal, NonTerminal]


class Field(NamedTuple):
    name: str  # concat'ed name from production-field paths
    cardinality: Literal['single', 'optional', 'multiple']
    type: int  # type ID
    # Path leading to field in the original production rule.
    original_index: List[Tuple[int, Optional[int]]]  # [(field_index, value_index)]


class Idiom(NamedTuple):
    name: str
    subtree: NonTerminal  # subtree containing only built-in idioms (productions)
    subtree_index: Optional[SubtreeIndex]
    fields: List[Field]


class IdiomProcessor:
    def __init__(self):
        self.generator = RobustCGenerator()
        with open("asdl/lang/c/c_asdl.txt", "r") as f:
            self.grammar = ASDLGrammar.from_text(f.read())
        self.transition_system = CTransitionSystem(self.grammar)
        self.idioms: List[Idiom] = []

        for prod_id, prod in enumerate(self.grammar.productions):
            subtree = NonTerminal(prod_id, {})
            fields = [Field(field.name, field.cardinality, self.grammar.type2id[field.type], [(field_idx, None)])
                      for field_idx, field in enumerate(prod.fields)]
            self.idioms.append(Idiom(prod.constructor.name, subtree, None, fields))

    def _subtree_to_compressed_ast(self, subtree: Subtree) -> CompressedAST:
        if isinstance(subtree, Terminal):
            return subtree.value
        ast_fields = [[self.transition_system.UNFILLED] if field.cardinality == "multiple"
                      else self.transition_system.UNFILLED
                      for field in self.idioms[subtree.prod_id].fields]
        for idx, field in subtree.children.items():
            if isinstance(field, MultipleValue):
                if field.fully_filled:
                    new_field = [self._subtree_to_compressed_ast(val) for _, val in sorted(field.value.items())]
                else:
                    max_idx = max(field.value.keys(), default=0)
                    min_idx = min(field.value.keys(), default=0)
                    field_count = max_idx + 2 if min_idx >= 0 else max_idx + 2 + -min_idx
                    new_field = [self.transition_system.UNFILLED] * field_count
                    for val_idx, val in field.value.items():
                        new_field[val_idx] = self._subtree_to_compressed_ast(val)
                ast_fields[idx] = new_field
            else:
                ast_fields[idx] = self._subtree_to_compressed_ast(field.value)
        return subtree.prod_id, ast_fields

    def subtree_to_ast(self, subtree: Subtree) -> ASTNode:
        ast = self._subtree_to_compressed_ast(subtree)
        asdl_ast = self.transition_system.decompress_ast(ast)
        c_ast = c_utils.asdl_ast_to_c_ast(asdl_ast, self.grammar, ignore_error=True)
        return c_ast

    def subtree_to_code(self, subtree: Subtree) -> str:
        c_ast = self.subtree_to_ast(subtree)
        code = self.generator.generate_code(c_ast)
        return code

    def _get_subtree(self, child_value: Union[int, str, None]) -> Subtree:
        if isinstance(child_value, int):
            return copy.deepcopy(self.idioms[child_value].subtree)
        else:
            assert child_value is None or isinstance(child_value, str)
            return Terminal(child_value)

    def repr_subtree(self, node: Subtree) -> str:
        if isinstance(node, Terminal):
            return str(node.value)
        idiom = self.idioms[node.prod_id]
        str_pieces = []
        for field_idx, field_val in node.children.items():
            field = idiom.fields[field_idx]
            if isinstance(field_val, MultipleValue):
                if field_val.fully_filled:
                    value_str = ", ".join(self.repr_subtree(val) for _, val in sorted(field_val.value.items()))
                    str_pieces.append(f"{field.name}=[{value_str}]")
                else:
                    for idx, val in sorted(field_val.value.items()):
                        value_str = self.repr_subtree(val)
                        str_pieces.append(f"{field.name}[{idx}]={value_str}")
            else:
                value_str = self.repr_subtree(field_val.value)
                str_pieces.append(f"{field.name}={value_str}")
        if len(str_pieces) == 0:
            return idiom.name
        return f"{idiom.name}({', '.join(str_pieces)})"

    def add_idiom(self, index: SubtreeIndex) -> int:
        subtree = copy.deepcopy(self.idioms[index.prod_id].subtree)
        # Find the corresponding field & child indexes in the expanded subtree.
        original_index = self.idioms[index.prod_id].fields[index.field_index].original_index
        node, field_index = subtree, original_index[0][0]
        for field_idx, val_idx in original_index[1:]:
            node = node.children[field_index]
            if isinstance(node, MultipleValue):
                assert val_idx is not None
                if node.fully_filled and val_idx < 0:
                    node = node.value[len(node.value) + val_idx]
                else:
                    node = node.value[val_idx]
            else:
                node = node.value
            field_index = field_idx

        # Add the new value to its appropriate position in the subtree.
        new_child_values: List[Union[int, str, None]] = []
        new_child_indexes: List[Optional[int]] = []
        if isinstance(index.child_value, tuple):
            # Fill all indexes using previous children and current values.
            if field_index not in node.children:
                node.children[field_index] = MultipleValue({})
            filled_values = node.children[field_index].value
            all_values = [None] * (len(filled_values) + len(index.child_value))
            fill_mark = [False] * (len(filled_values) + len(index.child_value))
            for idx, value in filled_values.items():
                all_values[idx] = value
                fill_mark[idx] = True
            idx = 0
            for value in index.child_value:
                while fill_mark[idx]: idx += 1
                all_values[idx] = self._get_subtree(value)
                fill_mark[idx] = True
                new_child_indexes.append(idx)
                new_child_values.append(value)
            node.children[field_index] = MultipleValue(
                {idx: value for idx, value in enumerate(all_values)}, fully_filled=True)
        else:
            if index.value_index is not None:
                # Compute the actual value index given the filled values.
                if field_index not in node.children:
                    node.children[field_index] = MultipleValue({})
                filled_values = node.children[field_index].value
                iota = itertools.count() if index.value_index >= 0 else itertools.count(-1, step=-1)
                drop_count = index.value_index if index.value_index >= 0 else -index.value_index - 1
                value_index = next(flutes.drop(drop_count, filter(lambda x: x not in filled_values, iota)))
                assert value_index not in filled_values
                child_subtree = self._get_subtree(index.child_value)
                filled_values[value_index] = child_subtree
                new_child_indexes.append(value_index)
            else:
                assert field_index not in node.children
                node.children[field_index] = SingleValue(self._get_subtree(index.child_value))
                new_child_indexes.append(None)
            new_child_values.append(index.child_value)

        # Update fields for the idiom.
        fields = []
        parent_fields = self.idioms[index.prod_id].fields
        assert index.field_index < len(parent_fields)
        for idx, field in enumerate(parent_fields):
            if idx == index.field_index:
                # This field is filled, add its unfilled children fields.
                if index.value_index is not None:
                    fields.append(field)  # retain the field if we only filled one of its values
                collapse_indexes = [(val, idx) for val, idx in zip(new_child_values, new_child_indexes)
                                    if isinstance(val, int)]
                for idiom_id, val_idx in collapse_indexes:
                    child_fields = self.idioms[idiom_id].fields
                    name_prefix = self.idioms[idiom_id].name + "."
                    for child_field in child_fields:
                        assert child_field.original_index[0][1] is None
                        child_orig_idx = [(child_field.original_index[0][0], val_idx),
                                          *child_field.original_index[1:]]
                        fields.append(Field(name_prefix + child_field.name, child_field.cardinality,
                                            child_field.type, original_index + child_orig_idx))
            else:
                fields.append(field)

        # Create a new name for the idiom.
        idiom_name = self.repr_subtree(subtree)
        idiom = Idiom(idiom_name, subtree, index, fields)
        self.idioms.append(idiom)
        return len(self.idioms) - 1


def count_node_types(data: List[CompressedAST]) -> CounterT[int]:
    def _dfs(ast: CompressedAST) -> None:
        prod_id, fields = ast
        counter[prod_id] += 1
        for field in fields:
            values = field if isinstance(field, list) else [field]
            for value in values:
                if isinstance(value, tuple):
                    _dfs(value)

    counter = Counter()
    for tree in data:
        _dfs(tree)
    return counter


def indent(text: str, spaces: int) -> str:
    indentation = " " * spaces
    return "\n".join(indentation + line for line in text.split("\n"))


def main():
    sys.setrecursionlimit(32768)
    args = Args()
    if args.n_procs == 0:
        flutes.register_ipython_excepthook()
        for name in dir(FindSubtreesState):
            method = getattr(FindSubtreesState, name)
            if hasattr(method, "__wrapped__"):
                setattr(FindSubtreesState, name, method.__wrapped__)

    processor = IdiomProcessor()
    data_iterator = read_data(args.data_dir, verbose=False)
    if args.max_examples is not None:
        data_iterator = itertools.islice(data_iterator, args.max_examples)
    data = flutes.LazyList(data_iterator)

    with flutes.safe_pool(args.n_procs, state_class=FindSubtreesState) as pool:
        for _ in pool.imap_unordered(
                FindSubtreesState.count_subtrees, tqdm(data, desc="Counting subtrees"),
                chunksize=1024):
            pass
        initial_node_counts = count_node_types(data)
        for _ in range(args.num_idioms):
            top_counts: CounterT[SubtreeIndex] = Counter()
            for counts in pool.get_states():
                top_counts.update(counts)
            [(subtree_index, freq)] = top_counts.most_common(1)
            idiom_idx = processor.add_idiom(subtree_index)
            idiom = processor.idioms[idiom_idx]
            subtree = idiom.subtree
            flutes.log(f"({_}) " + colored(f"Idiom {idiom_idx}:", attrs=["bold"]) + f" {idiom.name}", "success")
            flutes.log(f"Count = {freq}, {subtree_index}")
            flutes.log(colored("AST:\n", attrs=["bold"]) + indent(str(processor.subtree_to_ast(subtree)), 22))
            flutes.log(colored("Code:\n", attrs=["bold"]) + indent(processor.subtree_to_code(subtree), 22))
            print()
            result = pool.map_async(
                FindSubtreesState.replace_subtree_index, tqdm(data, desc="Applying idiom", leave=False),
                chunksize=1024, kwds={"replace_index": (idiom_idx, subtree_index)})
            data = result.get()
        final_node_counts = count_node_types(data)
        print(initial_node_counts)
        print(final_node_counts)


if __name__ == '__main__':
    main()
