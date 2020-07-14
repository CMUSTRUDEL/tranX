import copy
import functools
import itertools
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Counter as CounterT, Dict, List, Optional, Tuple, TypeVar, Union

import flutes
from argtyped import Arguments, Switch
from termcolor import colored

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.lang.c import c_utils
from asdl.transition_system import CompressedAST, TransitionSystem
from asdl.tree_bpe import *
from datasets.c.build_dataset import RawExample


class Args(Arguments):
    data_dir: str = "tranx_data_new"  # path to `create_c_dataset.py` outputs
    output_dir: str = "tranx_data_bpe"  # path to output folder where generated data will be stored
    n_procs: int = 0  # number of worker processes to spawn
    num_idioms: int = 100  # number of idioms to extract
    revert_low_freq_idioms: Switch = True  # will revert all idioms under a given threshold
    revert_freq: Optional[int]  # the threshold frequency. Will be queried interactively if `None`
    log_file: Optional[str]
    include_src_ast: Switch = False  # whether to include source-side AST in training; only target-side is used by def.

    verbose: Switch = False
    max_input_files: Optional[int]
    max_elems_per_file: Optional[int]
    save: Switch = True
    verify: Switch = False
    pdb: Switch = False

    def __init__(self):
        super().__init__()
        if self.verify and self.n_procs > 0:
            raise ValueError("Verification requires that `n_procs == 0`")
        if self.revert_low_freq_idioms and self.revert_freq is None:
            flutes.log("'--revert-low-freq-idioms' is enabled but '--revert-freq' is not set. Will query interactively "
                       "after all idioms are computed.", "warning")


T = TypeVar('T')
MTuple = Tuple[T, ...]
MaybeDict = Union[T, Dict[int, T]]


def get_data_files(data_dir: str) -> List[Path]:
    files = [file for file in sorted(Path(data_dir).iterdir())
             if file.name.startswith("data_") and file.suffix == ".pkl"]
    return files


class ComputeState(TreeBPEMixin, flutes.PoolState):
    def __init__(self, args: Args, proxy: Optional[flutes.ProgressBarManager.Proxy]):
        sys.setrecursionlimit(32768)
        self.args = args
        self.proxy = proxy
        self.file_paths: List[Path] = []
        self.file_sizes: List[int] = []
        self.data: List[CompressedAST] = []
        self.count: CounterT[TreeIndex.t] = Counter()
        self._bar_desc = f"Worker {flutes.get_worker_id()}"

    def _update_subtree(self, ast: CompressedAST, delta: int = 1) -> None:
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

    def _count_subtrees(self, ast: CompressedAST) -> None:
        prod_id, fields = ast
        self._update_subtree(ast)

        for field in fields:
            values = field if AST.is_multiple_field(field) else [field]
            for value in values:
                if AST.is_non_leaf(value):
                    self._count_subtrees(value)

    @flutes.exception_wrapper()
    def read_and_count_subtrees(self, path: Path, max_elements: Optional[int] = None) -> None:
        with path.open("rb") as f:
            data: List[RawExample] = pickle.load(f)
            if max_elements is not None:
                data = data[:max_elements]
        self.file_paths.append(path)
        self.file_sizes.append(len(data))
        for idx, example in enumerate(self.proxy.new(data, update_frequency=0.01, desc=self._bar_desc, leave=False)):
            if isinstance(example, RawExample):
                self.data.append(example.ast)
                self._count_subtrees(example.ast)
            else:  # RawExampleSrc
                if self.args.include_src_ast and example.src_ast is not None:
                    self.data.append(example.src_ast)
                    self._count_subtrees(example.src_ast)
                self.data.append(example.tgt_ast)
                self._count_subtrees(example.tgt_ast)

    @flutes.exception_wrapper()
    def save(self, directory: Path) -> None:
        data_count = 0
        for path, size in zip(self.proxy.new(self.file_paths, desc=self._bar_desc, leave=False), self.file_sizes):
            data = self.data[data_count:(data_count + size)]
            with (directory / path.name).open("wb") as f:
                pickle.dump(data, f)
            data_count += size

    @functools.lru_cache()
    def _traverse(self, func: Callable[[CompressedAST], int]) -> Callable[[CompressedAST], int]:
        def traverse_fn(ast: CompressedAST) -> int:
            new_prod_id = func(ast)
            fields = ast[1]
            # Recurse on each child, update counters where necessary.
            for field_idx, field in enumerate(fields):
                if AST.is_multiple_field(field):
                    for value_idx, value in enumerate(field):
                        if AST.is_non_leaf(value):
                            value_prod_id = traverse_fn(value)
                            if value_prod_id != value[0]:
                                for val_idx in [value_idx, -(len(field) - value_idx)]:
                                    self.count[TreeIndex.Value(new_prod_id, field_idx, value[0], val_idx)] -= 1
                                    self.count[TreeIndex.Value(new_prod_id, field_idx, value_prod_id, val_idx)] += 1
                                field[value_idx] = (value_prod_id, value[1])
                else:
                    if AST.is_non_leaf(field):
                        field_prod_id = traverse_fn(field)
                        if field_prod_id != field[0]:
                            self.count[TreeIndex.Value(new_prod_id, field_idx, field[0])] -= 1
                            self.count[TreeIndex.Value(new_prod_id, field_idx, field_prod_id)] += 1
                            fields[field_idx] = (field_prod_id, field[1])
            return new_prod_id

        return traverse_fn

    def _map_traverse(self, func: Callable[[CompressedAST], int]) -> None:
        traverse_fn = self._traverse(func)
        for idx, ast in enumerate(
                self.proxy.new(self.data, update_frequency=0.01, desc=self._bar_desc, leave=False)):
            new_prod_id = traverse_fn(ast)
            if new_prod_id != ast[0]:
                self.data[idx] = (new_prod_id, ast[1])

    def verify(self, idiom: 'Idiom') -> None:
        data = copy.deepcopy(self.data)
        count = +self.count  # `__pos__` on a counter returns a new counter stripping off keys with value of zero
        self.replace_idiom(idiom)
        self.revert_idiom(idiom)
        for old, new in zip(data, self.data):
            assert old == new
        new_count = +self.count
        assert count == new_count

    @flutes.exception_wrapper()
    def replace_idiom(self, idiom: 'Idiom') -> None:
        self._current_idiom = idiom
        self._map_traverse(self._replace_idiom)

    @flutes.exception_wrapper()
    def revert_idiom(self, idiom: 'Idiom') -> None:
        self._current_idiom = idiom
        self._map_traverse(self._revert_idiom)

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


class IdiomProcessor:
    def __init__(self, asdl_grammar_path: str, transition_system_lang: str):
        with open(asdl_grammar_path, "r") as f:
            self.grammar = ASDLGrammar.from_text(f.read())
        self.transition_system = TransitionSystem.get_class_by_lang(transition_system_lang)(self.grammar)
        self.idioms: List[Idiom] = []

        for prod_id, prod in enumerate(self.grammar.productions):
            subtree = NonTerminal(prod_id, {})
            fields = [
                Field(field.name, field.cardinality, self.grammar.type2id[field.type],
                      original_index=[Field.FieldIndex(field_idx)])
                for field_idx, field in enumerate(prod.fields)]
            self.idioms.append(Idiom(prod_id, prod.constructor.name, subtree, None, fields, slice(-1, -1)))

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

    def subtree_to_ast(self, subtree: Subtree) -> AbstractSyntaxTree:
        ast = self._subtree_to_compressed_ast(subtree)
        asdl_ast = self.transition_system.decompress_ast(ast)
        return asdl_ast

    def subtree_to_code(self, subtree: Subtree) -> str:
        asdl_ast = self.subtree_to_ast(subtree)
        code = self.transition_system.ast_to_surface_code(asdl_ast)
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

    def add_idiom(self, index: TreeIndex.t) -> Idiom:
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
                # The new field corresponds to all children in the specified field index in the previous idiom, but
                # could map to only a subset of fields in the original representation.
                new_fields.append(Field(
                    f"{child_field.name}[{idx}]", "single", child_field.type,
                    original_index=original_index + [Field.ValueIndex(idx)]))
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

                # The field must be a "multiple" field without explicit size; retain it for future fillings.
                assert not node.children[field_index].fully_filled
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
                for idx, child_field in enumerate(child_fields):
                    assert isinstance(child_field.original_index[0], Field.FieldIndex)
                    new_fields.append(Field(
                        name_prefix + child_field.name, child_field.cardinality, child_field.type,
                        original_index=new_original_index + child_field.original_index))

        # Update fields for the idiom.
        fields = parent_fields.copy()
        fields[index.field_index:(index.field_index + 1)] = new_fields
        left_field_index = index.field_index
        right_field_index = left_field_index + len(new_fields)
        if isinstance(index, TreeIndex.Value) and index.value_index is not None:
            left_field_index += 1  # `index.field_index` is the original "multiple" field

        # Create a new name for the idiom.
        idiom_name = self.repr_subtree(subtree)
        idiom_id = len(self.idioms)
        idiom = Idiom(idiom_id, idiom_name, subtree, index, fields, slice(left_field_index, right_field_index))
        self.idioms.append(idiom)
        return idiom


def indent(text: str, spaces: int = 22) -> str:
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


C_ASDL_GRAMMAR_PATH = "asdl/lang/c/c_asdl.txt"


def main() -> None:
    sys.setrecursionlimit(32768)
    args = Args()
    if args.pdb:
        flutes.register_ipython_excepthook()
        if args.n_procs == 0:
            for name in dir(ComputeState):
                method = getattr(ComputeState, name)
                if hasattr(method, "__wrapped__"):
                    setattr(ComputeState, name, method.__wrapped__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file is not None:
        flutes.set_log_file(args.log_file)
    processor = IdiomProcessor(C_ASDL_GRAMMAR_PATH, "c")
    data_files = get_data_files(args.data_dir)
    if args.max_input_files is not None:
        data_files = data_files[:args.max_input_files]

    manager = flutes.ProgressBarManager(verbose=args.verbose)
    proxy = manager.proxy
    with flutes.safe_pool(args.n_procs, state_class=ComputeState, init_args=(args, proxy), closing=[manager]) as pool:
        iterator = pool.imap_unordered(ComputeState.read_and_count_subtrees, data_files,
                                       kwds={"max_elements": args.max_elems_per_file})
        for _ in proxy.new(iterator, total=len(data_files), desc="Reading files"):
            pass

        if args.verify:
            data = copy.deepcopy(pool._pool._process_state.data)
            count = +pool._pool._process_state.count

        initial_node_counts = merge_counters(pool.broadcast(ComputeState.count_node_types))
        for _ in proxy.new(list(range(args.num_idioms)), desc="Finding idioms"):
            top_counts = merge_counters(pool.broadcast(ComputeState.get_top_counts, args=(100,)))
            [(subtree_index, freq)] = top_counts.most_common(1)
            idiom = processor.add_idiom(subtree_index)
            subtree = idiom.subtree
            flutes.log(f"({_}) " + colored(f"Idiom {idiom.id}:", attrs=["bold"]) + f" {idiom.name}", "success")
            flutes.log(f"Count = {freq}, {subtree_index}")
            c_ast = c_utils.asdl_ast_to_c_ast(processor.subtree_to_ast(subtree), processor.grammar, ignore_error=True)
            flutes.log(colored("AST:\n", attrs=["bold"]) + indent(str(c_ast)))
            flutes.log(colored("Code:\n", attrs=["bold"]) + indent(processor.subtree_to_code(subtree)))
            flutes.log("", timestamp=False)
            # if args.verify:
            #     pool.broadcast(ComputeState.verify, args=(idiom,))
            pool.broadcast(ComputeState.replace_idiom, args=(idiom,))
        final_node_counts = merge_counters(pool.broadcast(ComputeState.count_node_types))

        if args.verify:
            backup_data = copy.deepcopy(pool._pool._process_state.data)
            backup_count = +pool._pool._process_state.count
            from tqdm import tqdm
            for idx in tqdm(list(range(args.num_idioms)), desc="Reverting everything", position=1):
                pool.broadcast(ComputeState.revert_idiom, args=(processor.idioms[-(idx + 1)],))
            assert data == pool._pool._process_state.data
            assert count == +pool._pool._process_state.count
            flutes.log("Verification sucess!", "success")
            pool._pool._process_state.data = backup_data
            pool._pool._process_state.count = backup_count

        # Print counts before and after adding all idioms.
        counts = {idx: (initial_node_counts[idx], final_node_counts[idx]) for idx in range(len(processor.idioms))}
        flutes.log("", timestamp=False)
        flutes.log("Idx  PrevCnt      NewCnt      Diff  Name", timestamp=False)
        for idx, (prev_count, new_count) in sorted(counts.items(), key=lambda xs: xs[1][1] - xs[1][0]):
            if prev_count == new_count: continue
            flutes.log(f"{idx:3d} {prev_count:8d} -> {new_count:8d} ({new_count - prev_count:8d}) "
                       f"{processor.idioms[idx].name}", timestamp=False)

        revert_ids = []
        if args.revert_low_freq_idioms:
            proxy.close()
            revert_freq = args.revert_freq
            if revert_freq is None:
                flutes.log("Enter threshold frequency for reverting idioms. All idioms with counts lower than "
                           "the threshold will be removed.", timestamp=False)
                while True:
                    try:
                        user_input = input("> ")
                        revert_freq = int(user_input)
                        break
                    except ValueError:
                        flutes.log(f"Invalid input: {user_input}", timestamp=False)
            for idiom in processor.idioms:
                if idiom.tree_index is not None and final_node_counts[idiom.id] <= revert_freq:
                    revert_ids.append(idiom.id)
            flutes.log(f"The following {len(revert_ids)} idiom(s) will be reverted:\n" +
                       indent("\n".join(f"{idx:3d} {processor.idioms[idx].name}" for idx in revert_ids)), "warning")

        if args.save:
            for idx in proxy.new(list(reversed(revert_ids)), desc="Reverting idioms"):
                pool.broadcast(ComputeState.revert_idiom, args=(processor.idioms[idx],))
            pool.broadcast(ComputeState.save, args=(output_dir,))

    bpe = TreeBPE(processor.idioms, revert_ids)
    bpe.save(output_dir / "tree_bpe_model.pkl")
    if args.verify:
        bpe = TreeBPE.load(output_dir / "tree_bpe_model.pkl")
        final_data = pool._pool._process_state.data
        for orig_ast, target_ast in zip(data, final_data):
            encoded_ast = bpe.encode(orig_ast)
            assert encoded_ast == target_ast
            decoded_ast = bpe.decode(target_ast)
            assert decoded_ast == orig_ast
    flutes.log(f"Output stored to {args.output_dir}", "success")


if __name__ == '__main__':
    main()
