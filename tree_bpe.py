import copy
import itertools
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Counter as CounterT, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union, overload

import flutes
from argtyped import Arguments, Switch
from pycparser.c_ast import Node as ASTNode
from termcolor import colored
from typing_extensions import Literal

from asdl.asdl import ASDLGrammar
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CTransitionSystem, CompressedAST, RobustCGenerator
from datasets.c.build_dataset import RawExample


class Args(Arguments):
    data_dir: str = "tranx_data_new"  # path to `create_c_dataset.py` outputs
    output_dir: str = "tranx_data_new"  # path to output folder where generated data will be stored
    n_procs: int = 0  # number of worker processes to spawn
    num_idioms: int = 100  # number of idioms to extract

    verbose: Switch = False
    max_input_files: Optional[int]
    max_elems_per_file: Optional[int]


T = TypeVar('T')
MTuple = Tuple[T, ...]
MaybeDict = Union[T, Dict[int, T]]


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


def get_data_files(data_dir: str) -> List[Path]:
    files = [file for file in sorted(Path(data_dir).iterdir())
             if file.name.startswith("data_") and file.suffix == ".pkl"]
    return files


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


class SubtreeIndex(NamedTuple):
    prod_id: int  # ID of the production rule
    # ID of the production rule/literal value in the child node(s)
    child_value: Union[int, str, MTuple[int], MTuple[str], None]
    field_index: int  # index of the field within the production rule
    value_index: Optional[int] = None  # index within a multi-value field


class FindSubtreesState(flutes.PoolState):
    def __init__(self, proxy: Optional[flutes.ProgressBarManager.Proxy]):
        sys.setrecursionlimit(32768)
        self.proxy = proxy
        self.data: List[CompressedAST] = []
        self.count: CounterT[TreeIndex.t] = Counter()

    @staticmethod
    def _get_ast_value(ast: Optional[CompressedAST]) -> Union[str, int, None]:
        if ast is None or AST.is_leaf(ast):
            return ast
        return ast[0]

    def _update_subtree_indexes(self, ast: CompressedAST, delta: int = 1) -> None:
        # Return subtree indexes for only the current layer.
        prod_id, fields = ast
        for field_idx, field in enumerate(fields):
            if AST.is_multiple_field(field):
                # Field size.
                self.count[TreeIndex.Size(prod_id, field_idx, len(field))] += delta
                for value_idx, value in enumerate(field):
                    child_value = self._get_ast_value(value)
                    # Allow counting indices from both front and back.
                    self.count[TreeIndex.Value(prod_id, field_idx, child_value, value_idx)] += delta
                    self.count[TreeIndex.Value(prod_id, field_idx, child_value, -(len(field) - value_idx))] += delta
            else:
                child_value = self._get_ast_value(field)
                self.count[TreeIndex.Value(prod_id, field_idx, child_value)] += delta

    def _count_subtrees(self, ast: CompressedAST):
        prod_id, fields = ast
        self._update_subtree_indexes(ast)

        for field in fields:
            values = field if AST.is_multiple_field(field) else [field]
            for value in values:
                if AST.is_non_leaf(value):
                    self._count_subtrees(value)

    @flutes.exception_wrapper()
    def read_and_count_subtrees(self, path: str, max_elements: Optional[int] = None) -> None:
        with open(path, "rb") as f:
            data: List[RawExample] = pickle.load(f)
            if max_elements is not None:
                data = data[:max_elements]
        for idx, example in enumerate(
                self.proxy.new(data, update_frequency=0.01, desc=f"Worker {flutes.get_worker_id()}", leave=False)):
            self.data.append(example.ast)
            self._count_subtrees(example.ast)

    _target_id: int
    _current_index: TreeIndex.t

    def _replace_subtree_index(self, ast: CompressedAST) -> int:
        # Only the new production ID is returned; the fields are modified in-place.
        prod_id, fields = ast
        index = self._current_index
        new_prod_id = prod_id

        if prod_id == index.prod_id:
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
            # If the current AST matches the replacement subtree...
            if subtree_match:
                # Decrement counters for all parent-child pairs.
                self._update_subtree_indexes(ast, -1)
                if isinstance(index, TreeIndex.Size):
                    # Expand "multiple field" into multiple direct fields.
                    fields[index.field_index:(index.field_index + 1)] = child_field
                else:
                    # Collapse the child node (if exists) and move its fields to be fields of the current node.
                    if isinstance(index.child_value, int):
                        child_fields = child_field[1]
                        # Decrement counters for the child node to collapse.
                        self._update_subtree_indexes(child_field, -1)
                    else:
                        child_fields = []
                    if index.value_index is None:
                        fields[index.field_index:(index.field_index + 1)] = child_fields
                    else:
                        # Remove element in the original field with cardinality='multiple'.
                        del fields[index.field_index][index.value_index]
                        fields[(index.field_index + 1):(index.field_index + 1)] = child_fields
                new_prod_id = self._target_id
                # Increment counters for new parent-child pairs.
                self._update_subtree_indexes((new_prod_id, fields))

        # Recurse on each child, update counters where necessary.
        for field_idx, field in enumerate(fields):
            if AST.is_multiple_field(field):
                for value_idx, value in enumerate(field):
                    if AST.is_non_leaf(value):
                        value_prod_id = self._replace_subtree_index(value)
                        if value_prod_id != value[0]:
                            for val_idx in [value_idx, -(len(field) - value_idx)]:
                                self.count[TreeIndex.Value(new_prod_id, field_idx, value[0], val_idx)] -= 1
                                self.count[TreeIndex.Value(new_prod_id, field_idx, value_prod_id, val_idx)] += 1
                            field[value_idx] = (value_prod_id, value[1])
            else:
                if AST.is_non_leaf(field):
                    field_prod_id = self._replace_subtree_index(field)
                    if field_prod_id != field[0]:
                        self.count[TreeIndex.Value(new_prod_id, field_idx, field[0])] -= 1
                        self.count[TreeIndex.Value(new_prod_id, field_idx, field_prod_id)] += 1
                        fields[field_idx] = (field_prod_id, field[1])

        return new_prod_id

    @flutes.exception_wrapper()
    def replace_index(self, replace_id: int, index: TreeIndex.t) -> None:
        self._target_id = replace_id
        self._current_index = index
        for idx, ast in enumerate(
                self.proxy.new(self.data, update_frequency=0.01, desc=f"Worker {flutes.get_worker_id()}", leave=False)):
            new_prod_id = self._replace_subtree_index(ast)
            if new_prod_id != ast[0]:
                self.data[idx] = (new_prod_id, ast[1])

    @flutes.exception_wrapper()
    def get_top_counts(self, n: Union[int, float, None] = 0.01) -> CounterT[TreeIndex.t]:
        # By default, return the top 1% counts only. We assume the samples are randomly shuffled, so the counts in each
        # split should be proportional to split size.
        if n is None:
            return self.count
        top_n = n if isinstance(n, int) else int(len(self.count) * n)
        counter: CounterT[TreeIndex.t] = Counter()
        for key, count in self.count.most_common(top_n):
            counter[key] = count
        return counter

    @flutes.exception_wrapper()
    def count_node_types(self) -> CounterT[int]:
        def _dfs(ast: CompressedAST) -> None:
            prod_id, fields = ast
            counter[prod_id] += 1
            for field in fields:
                values = field if isinstance(field, list) else [field]
                for value in values:
                    if isinstance(value, tuple):
                        _dfs(value)

        counter: CounterT[int] = Counter()
        for tree in self.data:
            _dfs(tree)
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
    type: int  # type ID
    original_index: List[Union[FieldIndex, ValueIndex]]  # path leading to field in the original production rule


class Idiom(NamedTuple):
    name: str
    subtree: NonTerminal  # subtree containing only built-in idioms (productions)
    subtree_index: Optional[TreeIndex.t]
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
            fields = [
                Field(field.name, field.cardinality, self.grammar.type2id[field.type], [Field.FieldIndex(field_idx)])
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
                if field.size is not None:
                    field_count = field.size
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
                if field_val.size is not None:
                    values = ["<?>"] * field_val.size
                    for idx, val in field_val.value.items():
                        values[idx] = self.repr_subtree(val)
                    value_str = ", ".join(values)
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

    def add_idiom(self, index: TreeIndex.t) -> int:
        subtree = copy.deepcopy(self.idioms[index.prod_id].subtree)
        # Find the corresponding field & child indexes in the expanded subtree.
        original_index = self.idioms[index.prod_id].fields[index.field_index].original_index
        node, value = subtree, None
        for idx in original_index[:-1]:
            if isinstance(idx, Field.FieldIndex):
                value = node.children[idx]
                if isinstance(value, SingleValue):
                    node = value.value
            else:  # Field.ValueIndex
                assert value is not None
                node = value.value[idx]
        final_index = original_index[-1]

        # Add the new value to its appropriate position in the subtree.
        new_fields: List[Tuple[int, Optional[int]]] = []  # [(child_prod_id, value_index)]
        parent_fields = self.idioms[index.prod_id].fields
        if isinstance(index, TreeIndex.Size):
            # `Size` only applies to "multiple field"s, so `final_index` must point to a field.
            assert isinstance(final_index, Field.FieldIndex)
            field_index = final_index
            # Fill all indexes using previous children and current values.
            if field_index not in node.children:
                field_object = node.children[field_index] = MultipleValue({})
            else:
                field_object = node.children[field_index]
            assert field_object.size is None
            filled_values = field_object.value
            field_size = len(filled_values) + index.field_size
            node.children[field_index] = field_object._replace(size=field_size)

            # Expand the field into multiple direct fields, corresponding to unfilled children.
            fill_mark = [False] * field_size
            for idx, value in filled_values.items():
                fill_mark[idx] = True
            child_field = parent_fields[index.field_index]
            for idx, mark in enumerate(fill_mark):
                if mark: continue
                assert isinstance(child_field.original_index[0], Field.FieldIndex)
                new_fields.append(Field(f"{child_field.name}[{idx}]", "single", child_field.type,
                                        original_index + [Field.ValueIndex(idx)]))
        else:
            if index.value_index is not None:
                # `Size` only applies to "multiple field"s, so `final_index` must point to a field.
                assert isinstance(final_index, Field.FieldIndex)
                field_index = final_index
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

                if not node.children[field_index].fully_filled:
                    # Retain the field if it is a partially filled "multiple field".
                    new_fields.append(parent_fields[index.field_index])
            else:
                if isinstance(final_index, Field.FieldIndex):
                    assert final_index not in node.children
                    node.children[final_index] = SingleValue(self._get_subtree(index.child_value))
                else:  # Field.ValueIndex
                    assert value is not None and final_index not in value.value
                    value.value[final_index] = self._get_subtree(index.child_value)
                value_index = None

            if isinstance(index.child_value, int):
                # Move fields of the child node to the parent node.
                child_fields = self.idioms[index.child_value].fields
                name_prefix = self.idioms[index.child_value].name + "."
                new_original_index = original_index + ([] if value_index is None else [Field.ValueIndex(value_index)])
                for child_field in child_fields:
                    assert isinstance(child_field.original_index[0], Field.FieldIndex)
                    new_fields.append(Field(name_prefix + child_field.name, child_field.cardinality, child_field.type,
                                            new_original_index + child_field.original_index))

        # Update fields for the idiom.
        fields = parent_fields.copy()
        fields[index.field_index:(index.field_index + 1)] = new_fields

        # Create a new name for the idiom.
        idiom_name = self.repr_subtree(subtree)
        idiom = Idiom(idiom_name, subtree, index, fields)
        self.idioms.append(idiom)
        return len(self.idioms) - 1


def indent(text: str, spaces: int) -> str:
    indentation = " " * spaces
    return "\n".join(indentation + line for line in text.split("\n"))


def merge_counters(counters: List[CounterT[T]]) -> CounterT[T]:
    # Merge multiple counters; may modify counter contents.
    if len(counters) == 0:
        return Counter()
    counters.sort(key=len)
    merged_counter = counters[-1]
    for counter in counters[:-1]:
        merged_counter.update(counter)
    return merged_counter


def main() -> None:
    sys.setrecursionlimit(32768)
    args = Args()
    # flutes.register_ipython_excepthook()
    if args.n_procs == 0:
        for name in dir(FindSubtreesState):
            method = getattr(FindSubtreesState, name)
            if hasattr(method, "__wrapped__"):
                setattr(FindSubtreesState, name, method.__wrapped__)

    processor = IdiomProcessor()
    data_files = get_data_files(args.data_dir)
    if args.max_input_files is not None:
        data_files = data_files[:args.max_input_files]

    manager = flutes.ProgressBarManager(verbose=args.verbose)
    proxy = manager.proxy
    with flutes.safe_pool(args.n_procs, state_class=FindSubtreesState, init_args=(proxy,), closing=[manager]) as pool:
        iterator = pool.imap_unordered(FindSubtreesState.read_and_count_subtrees, data_files,
                                       kwds={"max_elements": args.max_elems_per_file})
        for _ in proxy.new(iterator, total=len(data_files), desc="Reading files"):
            pass
        initial_node_counts = merge_counters(pool.broadcast(FindSubtreesState.count_node_types))
        for _ in proxy.new(list(range(args.num_idioms)), desc="Finding idioms"):
            top_counts = merge_counters(pool.broadcast(FindSubtreesState.get_top_counts, args=(100,)))
            [(subtree_index, freq)] = top_counts.most_common(1)
            idiom_idx = processor.add_idiom(subtree_index)
            idiom = processor.idioms[idiom_idx]
            subtree = idiom.subtree
            flutes.log(f"({_}) " + colored(f"Idiom {idiom_idx}:", attrs=["bold"]) + f" {idiom.name}", "success")
            flutes.log(f"Count = {freq}, {subtree_index}")
            flutes.log(colored("AST:\n", attrs=["bold"]) + indent(str(processor.subtree_to_ast(subtree)), 22))
            flutes.log(colored("Code:\n", attrs=["bold"]) + indent(processor.subtree_to_code(subtree), 22))
            flutes.log("", timestamp=False)
            pool.broadcast(FindSubtreesState.replace_index, args=(idiom_idx, subtree_index))
        final_node_counts = merge_counters(pool.broadcast(FindSubtreesState.count_node_types))
    counts = {idx: (initial_node_counts[idx], final_node_counts[idx]) for idx in range(len(processor.idioms))}
    print()
    print("Idx  PrevCnt      NewCnt      Diff  Name")
    for idx, (prev_count, new_count) in sorted(counts.items(), key=lambda xs: xs[1][1] - xs[1][0]):
        if prev_count == new_count: continue
        print(f"{idx:3d} {prev_count:8d} -> {new_count:8d} ({new_count - prev_count:8d}) {processor.idioms[idx].name}")


if __name__ == '__main__':
    main()
