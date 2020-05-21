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
from tqdm import tqdm
from typing_extensions import Literal

from asdl.asdl import ASDLGrammar
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CTransitionSystem, CompressedAST, RobustCGenerator
from datasets.c.build_dataset import RawExample

T = TypeVar('T')
MaybeDict = Union[T, Dict[int, T]]
FieldValue = Union[CompressedAST, str, List[CompressedAST]]


class Args(Arguments):
    data_dir: str = "tranx_data_new"  # path to `create_c_dataset.py` outputs
    output_dir: str = "tranx_data_new"  # path to output folder where generated data will be stored
    n_procs: int = 0  # number of worker processes to spawn
    num_idioms: int = 100  # number of idioms to extract

    max_examples: Optional[int]


def read_data(data_dir: str) -> Iterator[CompressedAST]:
    files = [file for file in sorted(Path(data_dir).iterdir())
             if file.name.startswith("data_") and file.suffix == ".pkl"]
    for file in tqdm(files, desc="Reading file"):
        with file.open("rb") as f:
            data: List[RawExample] = pickle.load(f)
        for example in data:
            yield example.ast


class SubtreeIndex(NamedTuple):
    prod_id: int  # ID of the production rule
    child_value: Union[int, str, None]  # ID of the production rule/literal value in the child node
    field_index: int  # index of the field within the production rule
    value_index: Optional[int] = None  # index within a multi-value field


class FindSubtreesState(flutes.PoolState):
    def __init__(self):
        sys.setrecursionlimit(32768)
        self.count: CounterT[SubtreeIndex] = Counter()

    @staticmethod
    def _get_subtree_indexes(ast: CompressedAST) -> List[SubtreeIndex]:
        # Return subtree indexes for only the current layer.
        prod_id, fields = ast
        indexes = []
        for field_idx, field in enumerate(fields):
            if isinstance(field, list):
                for value_idx, value in enumerate(field):
                    child_value = value[0] if isinstance(value, tuple) else value
                    # Allow counting indices from both front and back.
                    indexes.append(SubtreeIndex(prod_id=prod_id, child_value=child_value,
                                                field_index=field_idx, value_index=value_idx))
                    indexes.append(SubtreeIndex(prod_id=prod_id, child_value=child_value,
                                                field_index=field_idx, value_index=-(len(field) - value_idx)))
            else:
                child_value = field[0] if isinstance(field, tuple) else field
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
            child_value = (child_field[0] if isinstance(index.child_value, int) and child_field is not None
                           else child_field)
            # If the current AST matches the replacement subtree...
            if child_value == index.child_value:
                # Decrement counters for all parent-child pairs.
                indexes = self._get_subtree_indexes((prod_id, fields))
                for idx in indexes:
                    self.count[idx] -= 1
                child_fields = []
                if isinstance(index.child_value, int):
                    child_fields = child_field[1]
                    # Decrement counters for the child node to collapse.
                    indexes = self._get_subtree_indexes(child_field)
                    for idx in indexes:
                        self.count[idx] -= 1
                if index.value_index is None:
                    new_fields[index.field_index:(index.field_index + 1)] = child_fields
                else:
                    del new_fields[index.field_index][index.value_index]
                    new_fields[(index.field_index + 1):(index.field_index + 1)] = child_fields
                new_prod_id = replace_index[0]
                # Increment counters for new parent-child pairs.
                indexes = self._get_subtree_indexes((new_prod_id, new_fields))
                self.count.update(indexes)

        # Recurse on each child, update counters where necessary.
        for field_idx, field in enumerate(new_fields):
            if isinstance(field, list):
                for value_idx, value in enumerate(field):
                    if isinstance(value, tuple):
                        new_value = self.replace_subtree_index(value, replace_index)
                        if new_value[0] != value[0]:
                            for val_idx in [value_idx, -(len(field) - value_idx)]:
                                self.count[SubtreeIndex(prod_id=new_prod_id, child_value=value[0],
                                                        field_index=field_idx, value_index=val_idx)] -= 1
                                self.count[SubtreeIndex(prod_id=new_prod_id, child_value=new_value[0],
                                                        field_index=field_idx, value_index=val_idx)] += 1
                    else:
                        new_value = value
                    field[value_idx] = new_value
            else:
                if isinstance(field, tuple):
                    new_field = self.replace_subtree_index(field, replace_index)
                    if new_field[0] != field[0]:
                        self.count[SubtreeIndex(prod_id=new_prod_id, child_value=field[0],
                                                field_index=field_idx)] -= 1
                        self.count[SubtreeIndex(prod_id=new_prod_id, child_value=new_field[0],
                                                field_index=field_idx)] += 1
                else:
                    new_field = field
                new_fields[field_idx] = new_field

        return new_prod_id, new_fields

    def replace_subtree(self, subtree: 'Subtree', ast: CompressedAST) -> CompressedAST:
        prod_id, fields = ast
        pass

    def __return_state__(self) -> CounterT[SubtreeIndex]:
        # Return the top 1% counts only. We assume the samples are randomly shuffled, so the counts in each split
        # should be proportional to split size.
        counter = Counter()
        for key, count in self.count.most_common(len(self.count) // 100):
            counter[key] = count
        return counter


class Terminal(NamedTuple):
    value: str


class NonTerminal(NamedTuple):
    prod_id: int
    children: 'SubtreeChildren'  # (field_idx) -> ( subtree | (value_idx) -> subtree )


Subtree = Union[Terminal, NonTerminal]
SubtreeChildren = Dict[int, MaybeDict[Subtree]]


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
        fields = [{} if field.cardinality == "multiple" else self.transition_system.UNFILLED
                  for field in self.idioms[subtree.prod_id].fields]
        for idx, field in subtree.children.items():
            if isinstance(field, dict):
                for val_idx, value in field.items():
                    fields[idx][val_idx] = self._subtree_to_compressed_ast(value)
            else:
                fields[idx] = self._subtree_to_compressed_ast(field)
        for idx, field in enumerate(fields):
            if isinstance(field, dict):
                if len(field) == 0:
                    new_field = []
                else:
                    max_idx = max(field.keys())
                    min_idx = min(field.keys())
                    field_count = max_idx + 1 if min_idx >= 0 else max_idx + 2 + -min_idx
                    new_field = [self.transition_system.UNFILLED] * field_count
                    for val_idx, val in field.items():
                        new_field[val_idx] = val
                fields[idx] = new_field
        return subtree.prod_id, fields

    def subtree_to_ast(self, subtree: Subtree) -> ASTNode:
        ast = self._subtree_to_compressed_ast(subtree)
        asdl_ast = self.transition_system.decompress_ast(ast)
        c_ast = c_utils.asdl_ast_to_c_ast(asdl_ast, self.grammar, ignore_error=True)
        return c_ast

    def subtree_to_code(self, subtree: Subtree) -> str:
        c_ast = self.subtree_to_ast(subtree)
        code = self.generator.generate_code(c_ast)
        return code

    def add_idiom(self, index: SubtreeIndex) -> int:
        subtree = copy.deepcopy(self.idioms[index.prod_id].subtree)
        if isinstance(index.child_value, int):
            child_subtree = copy.deepcopy(self.idioms[index.child_value].subtree)
        else:
            child_subtree: Subtree = Terminal(index.child_value)
        original_index = self.idioms[index.prod_id].fields[index.field_index].original_index
        node, field_index = subtree, original_index[0][0]
        for field_idx, val_idx in original_index[1:]:
            node = node.children[field_index]
            if val_idx is not None:
                node = node[val_idx]
            field_index = field_idx
        if index.value_index is not None:
            # Compute the actual value index given the filled values.
            filled_values = node.children.setdefault(field_index, {})
            iota = itertools.count() if index.value_index >= 0 else itertools.count(-1, step=-1)
            drop_count = index.value_index if index.value_index >= 0 else -index.value_index - 1
            value_index = next(flutes.drop(drop_count, filter(lambda x: x not in filled_values, iota)))
            assert value_index not in filled_values
            filled_values[value_index] = child_subtree
        else:
            assert field_index not in node.children
            node.children[field_index] = child_subtree
            value_index = None

        fields = []
        parent_fields = self.idioms[index.prod_id].fields
        assert index.field_index < len(parent_fields)
        for idx, field in enumerate(parent_fields):
            if idx == index.field_index:
                # This field is filled, add its unfilled children fields.
                if value_index is not None:
                    fields.append(field)  # retain the field if we only filled one of its values
                if isinstance(index.child_value, int):
                    child_fields = self.idioms[index.child_value].fields
                    name_prefix = self.idioms[index.child_value].name + "."
                    for child_field in child_fields:
                        assert child_field.original_index[0][1] is None
                        child_orig_idx = [(child_field.original_index[0][0], value_index),
                                          *child_field.original_index[1:]]
                        fields.append(Field(name_prefix + child_field.name, child_field.cardinality,
                                            child_field.type, original_index + child_orig_idx))
            else:
                fields.append(field)
        new_child_field_name = parent_fields[index.field_index].name
        if index.value_index is not None:
            new_child_field_name += f"[{value_index}]"
        new_child_value_str = (self.idioms[index.child_value].name if isinstance(index.child_value, int)
                               else index.child_value)
        idiom_name = self.idioms[index.prod_id].name
        if idiom_name.endswith(")"):
            idiom_name = f"{idiom_name[:-1]}, {new_child_field_name}={new_child_value_str})"
        else:
            idiom_name = f"{idiom_name}({new_child_field_name}={new_child_value_str})"
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
    data_iterator = read_data(args.data_dir)
    if args.max_examples is not None:
        data_iterator = itertools.islice(data_iterator, args.max_examples)
    data = flutes.LazyList(data_iterator)

    with flutes.safe_pool(args.n_procs, state_class=FindSubtreesState) as pool:
        result = pool.map_async(FindSubtreesState.count_subtrees, tqdm(data, desc="Counting subtrees"))
        initial_node_counts = count_node_types(data)
        result.wait()
        for _ in range(args.num_idioms):
            top_counts: CounterT[SubtreeIndex] = Counter()
            for counts in pool.get_states():
                top_counts.update(counts)
            [(subtree_index, freq)] = top_counts.most_common(1)
            idiom_idx = processor.add_idiom(subtree_index)
            idiom = processor.idioms[idiom_idx]
            subtree = idiom.subtree
            flutes.log(f"({_}) Idiom {idiom_idx}: {idiom.name}, {subtree_index} (count = {freq})", "success")
            flutes.log("AST:\n" + str(processor.subtree_to_ast(subtree)))
            flutes.log("Code:\n" + processor.subtree_to_code(subtree))
            result = pool.map_async(
                FindSubtreesState.replace_subtree_index, tqdm(data, desc="Applying idiom", leave=False),
                chunksize=1024, kwds={"replace_index": (idiom_idx, subtree_index)})
            data = result.get()
        final_node_counts = count_node_types(data)
        print(initial_node_counts)
        print(final_node_counts)


if __name__ == '__main__':
    main()
