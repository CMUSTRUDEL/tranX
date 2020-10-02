import contextlib
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from typing import Sequence, TypeVar

import flutes
import numpy as np
import sentencepiece as spm
from argtyped import Arguments, Switch

from asdl import CTransitionSystem
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import CompressedAST
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CGenTokenAction
from asdl.tree_bpe import TreeBPE
from components.vocab import VocabEntry
from datasets.c.build_dataset import RawExampleSrc


class Args(Arguments):
    data_dir: str = "tranx_data_small"
    output_dir: str = "tree2tree_data_bpe_small"
    asdl_path: str = "asdl/lang/c/c_asdl.txt"
    spm_model_path: str = "tranx_data/vocab.model"
    tree_bpe_path: Optional[str] = None

    data_percentage: float = 1.0
    compress: Switch = True
    n_procs: int = 0
    pdb: Switch = False
    verify: Switch = False
    show_progress: Switch = True


ProcessedAST = Tuple[int, List[Tuple[int, List[Any]]]]  # (str, [AST])
JSON = Dict[str, Any]


class Portion:
    def __init__(self):
        self.number = 0
        self.total = 0

    def add(self, number: int, total: int = 1) -> None:
        self.number += number
        self.total += total

    def __float__(self):
        return self.number / self.total


T = TypeVar('T')


def lcs(a: Sequence[T], b: Sequence[T]) -> int:
    r"""Compute the longest common subsequence (LCS) between two lists.

    :return: Length of the LCS.
    """
    return lcs_matrix(a, b)[len(a), len(b)]


def lcs_matrix(a: Sequence[T], b: Sequence[T]) -> np.ndarray:
    r"""Compute the edit distance between two lists.

    :return: The DP cost matrix.
    """
    n, m = len(a), len(b)
    f = np.zeros((n + 1, m + 1), dtype=np.int16)
    for i in range(n):
        for j in range(m):
            f[i + 1, j + 1] = max(f[i, j + 1], f[i + 1, j])
            if a[i] == b[j]:
                f[i + 1, j + 1] = max(f[i + 1, j + 1], f[i, j] + 1)
    return f


class LegacyT2TDataProcessor:
    def __init__(self, asdl_path: str, spm_model_path: str, tree_bpe_path: Optional[str] = None, strict: bool = True):
        with open(asdl_path) as f:
            self.grammar = ASDLGrammar.from_text(f.read())
        self.tree_bpe = None
        if tree_bpe_path is not None:
            self.tree_bpe = TreeBPE.load(tree_bpe_path)
            self.grammar = self.tree_bpe.patch_grammar(self.grammar)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.reserved_words = set(c_utils.RESERVED_WORDS)
        self.trans = CTransitionSystem(self.grammar, self.sp)

        self.vocab = VocabEntry()
        self.EOS = self.vocab.add(CGenTokenAction.STOP_SIGNAL)
        self.NONE = self.vocab.add(self._encode_node("None"))
        self.min_node_id = len(self.vocab)
        for prod in self.grammar.productions:
            self.vocab.add(self._encode_node(prod.constructor.name))
        self.vocab.add(self._encode_node("NoQuals"))
        self.vocab.add(self._encode_node("ConstVolatile"))
        self.max_node_id = len(self.vocab) - 1
        with Path(spm_model_path).with_suffix(".vocab").open() as f:
            vocab_lines = [line.split("\t")[0] for line in f if line]
        for word in vocab_lines:
            self.vocab.add(word)

        self.CONST_PROD_ID = self.grammar.prod2id[self.grammar.get_prod_by_ctr_name("Const")]
        self.VOLATILE_PROD_ID = self.grammar.prod2id[self.grammar.get_prod_by_ctr_name("Volatile")]

        self.var_map: Optional[Dict[str, List[str]]] = None
        self.strict = strict

    def _convert_quals_field(self, quals: List[CompressedAST]) -> str:
        qual_names = [self.grammar.id2prod[prod_id].constructor.name for prod_id, _ in quals]
        const = "Const" in qual_names
        volatile = "Volatile" in qual_names
        return (("Const" if const else "") + ("Volatile" if volatile else "")) or "NoQuals"

    def _transform_literal(self, literal: str) -> List[ProcessedAST]:
        if self.var_map is not None and literal in self.var_map:
            tokens = self.var_map[literal]
        else:
            tokens = self.sp.encode_as_pieces(literal)
        nodes = [(self.vocab[token], []) for token in tokens]
        nodes.append((self.EOS, []))
        return nodes

    def _encode_node(self, node: str) -> str:
        return f"[{node}]"

    def _transform_node(self, node: str) -> int:
        return self.vocab[self._encode_node(node)]

    def _is_node(self, node_id: int) -> bool:
        if self.min_node_id <= node_id <= self.max_node_id:
            s = self.vocab.id2word[node_id]
            assert s[0] == '[' and s[-1] == ']'
            return True
        # The name check can be false positive in string literals
        return False

    def _detransform_literal(self, node_ids: List[int]) -> str:
        if self.strict:
            assert node_ids[-1] == self.EOS
        return self.sp.decode_pieces([self.vocab.id2word[node_id] for node_id in node_ids[:-1]])

    def _detransform_node(self, node_id: int) -> str:
        return self.vocab.id2word[node_id][1:-1]

    def set_var_map(self, var_map: Optional[Dict[str, str]]) -> None:
        if var_map is None:
            self.var_map = None
        else:
            self.var_map = {key: self.sp.encode_as_pieces(val) for key, val in var_map.items()}

    def transform_ast(self, ast: CompressedAST) -> ProcessedAST:
        if ast is None: return (self.NONE, [])

        prod_id, fields = ast
        prod_name = self.grammar.id2prod[prod_id].constructor.name
        node_id = self._transform_node(prod_name)

        # `IdentifierType` is the only node with more than one identifier/literal value.
        if prod_name == "IdentifierType":
            return (node_id, self._transform_literal(" ".join(fields[0])))

        # For declaration nodes, we only keep the type qualifiers, and convert that to a singular field
        quals = None
        if prod_name == "Decl":
            quals = fields[1]
            fields = fields[:1] + fields[4:]
        elif prod_name == "PtrDecl":
            quals = fields[0]
            fields = fields[1:]
        elif prod_name == "ArrayDecl":
            fields = fields[:2]  # drop dim_quals
        elif prod_name in {"TypeDecl", "Typedef", "Typename"}:
            quals = fields[1]
            fields = [fields[0], fields[-1]]  # only identifier, quals, and type

        children = []
        if quals is not None:
            children.append((self._transform_node(self._convert_quals_field(quals)), []))
        for field in fields:
            for value in (field if isinstance(field, list) else [field]):
                if isinstance(value, str):
                    children.extend(self._transform_literal(value))
                else:
                    children.append(self.transform_ast(value))
        return (node_id, children)

    def _suppress(self, *ex_typs):
        assert len(ex_typs) > 0
        if self.strict:
            return contextlib.nullcontext()
        return contextlib.suppress(*ex_typs)

    def detransform_ast(self, ast: ProcessedAST) -> CompressedAST:
        node_id, children = ast
        assert self._is_node(node_id)
        prod_name = self._detransform_node(node_id)
        prod = self.grammar.get_prod_by_ctr_name(prod_name)
        prod_id = self.grammar.prod2id[prod]

        qual_idx = None
        fields = [None] * len(prod.fields)
        fields_to_fill = list(range(len(prod.fields)))
        if prod_name == "Decl":
            qual_idx = 1
            fields_to_fill = [0, 4, 5, 6]
        elif prod_name == "PtrDecl":
            qual_idx = 0
            fields_to_fill = [1]
        elif prod_name == "ArrayDecl":
            fields_to_fill = [0, 1]
        elif prod_name in {"TypeDecl", "Typedef", "Typename"}:
            qual_idx = 1
            fields_to_fill = [0, len(prod.fields) - 1]

        if qual_idx is not None:
            with self._suppress(IndexError):
                qual_node = self._detransform_node(children[0][0])
                children = children[1:]
                quals = []
                if "Const" in qual_node: quals.append(self.CONST_PROD_ID)
                if "Volatile" in qual_node: quals.append(self.VOLATILE_PROD_ID)
                fields[qual_idx] = [(qual, []) for qual in quals]

        processed_children = []
        idx = 0
        while idx < len(children):
            child_node_id, _ = child = children[idx]
            idx += 1
            if child_node_id == self.NONE:
                processed_children.append(None)
            elif self._is_node(child_node_id):
                processed_children.append(self.detransform_ast(child))
            else:
                ids = [child_node_id]
                while idx < len(children):
                    child_node_id, _ = children[idx]
                    if self.strict:
                        assert not self._is_node(child_node_id)
                    elif self._is_node(child_node_id):
                        break
                    ids.append(child_node_id)
                    idx += 1
                    if child_node_id == self.EOS: break
                processed_children.append(self._detransform_literal(ids))

        sequential_idx = next((idx for idx, field in enumerate(prod.fields) if field.cardinality == 'multiple'), None)
        for field_idx in fields_to_fill:
            if sequential_idx == field_idx:
                l = field_idx
                r = ((field_idx + 1) - len(prod.fields)) or None
                fields[field_idx] = processed_children[l:r]
            else:
                child_idx = field_idx
                if sequential_idx is not None and field_idx > sequential_idx:
                    child_idx = field_idx - len(prod.fields)
                with self._suppress(IndexError, AssertionError):
                    child = processed_children[child_idx]
                    if prod.fields[field_idx].cardinality != "optional":
                        assert child is not None
                    if child is not None:
                        if prod.fields[field_idx].type.name in {"IDENT", "LITERAL"}:
                            assert isinstance(child, str)
                        else:
                            assert not isinstance(child, str)
                    fields[field_idx] = child

        return (prod_id, fields)

    def expand(self, ast: ProcessedAST) -> JSON:
        node_id, children = ast
        return {"root": self.vocab.id2word[node_id],
                "children": [self.expand(child) for child in children]}

    def compress(self, js: JSON) -> ProcessedAST:
        node_id = self.vocab[js["root"]]
        children = [self.compress(child) for child in js["children"]]
        return (node_id, children)

    def _verify_transform_src_ast(self, ast):
        if not isinstance(ast, tuple):
            if ast in self.var_map:
                return self.sp.decode_pieces(self.var_map[ast])
            return ast
        proc_id, fields = ast
        new_fields = [[self._verify_transform_src_ast(value) for value in field] if isinstance(field, list)
                      else self._verify_transform_src_ast(field) for field in fields]
        return proc_id, new_fields

    def _verify(self, orig_ast: CompressedAST, check_ast: CompressedAST):
        orig_code = self.trans.ast_to_surface_code(self.trans.decompress_ast(orig_ast))
        check_code = self.trans.ast_to_surface_code(self.trans.decompress_ast(check_ast))
        if (orig_code != check_code and
                lcs(orig_code, check_code) < len(check_code) - check_code.count(' ⁇ ') * 3 - orig_code.count('\t')):
            print(orig_code)
            print("-" * 80)
            print(check_code)
            print("=" * 80)

    def verify(self, processed: ProcessedAST, original: CompressedAST, src: bool):
        src_ast_orig = self.detransform_ast(processed)
        if src:
            original = self._verify_transform_src_ast(original)
        self._verify(original, src_ast_orig)


class T2TDataProcessor(LegacyT2TDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.END_OF_SEQ = self.vocab.add(self._encode_node("EOS"))

    class FieldInfo(NamedTuple):
        quals_idx: Optional[int]  # index of the `quals` field
        flatten_field: List[bool]  # [should_flatten_field for field in fields]
        keep_field: List[bool]  # [should_keep_field for field in fields]

    @lru_cache()
    def get_field_info(self, prod_id: int) -> FieldInfo:
        prod = self.grammar.id2prod[prod_id]
        quals_idx = None
        flatten_fields = []
        keep_fields = []
        for idx, field in enumerate(prod.fields):
            flatten = False
            if field.cardinality != "multiple":
                keep = True  # keep all non-sequential fields
            else:
                if field.type.name == "QUAL":
                    # For declaration nodes, we only keep the type qualifiers, and convert that to a singular field
                    # Theoretically there can be multiple QUAL fields now that we have TreeBPE, but these fields should
                    # be collapsed first during BPE training.
                    assert quals_idx is None
                    quals_idx = idx
                    keep = False
                elif field.type.name == "EXPR":
                    keep = True
                elif field.type.name in {"IDENT", "LITERAL"}:
                    # `IdentifierType` is the only node with more than one identifier/literal value.
                    flatten = True
                    keep = True
                else:
                    keep = False  # drop sequential non-literal terminals
            keep_fields.append(keep)
            flatten_fields.append(flatten)
        return self.FieldInfo(quals_idx, flatten_fields, keep_fields)

    def _transform_ast(self, ast: CompressedAST) -> ProcessedAST:
        if ast is None: return (self.NONE, [])

        prod_id, fields = ast
        prod = self.grammar.id2prod[prod_id].constructor
        node_id = self._transform_node(prod.name)

        fields_info = self.get_field_info(prod_id)
        children = []
        if fields_info.quals_idx is not None:
            children.append((self._transform_node(self._convert_quals_field(fields[fields_info.quals_idx])), []))
        for idx, field in enumerate(fields):
            if not fields_info.keep_field[idx]: continue
            if isinstance(field, list):
                assert prod.fields[idx].cardinality == "multiple"
                values = field
                is_sequential = True
            else:
                values = [field]
                is_sequential = False
            if fields_info.flatten_field[idx] and isinstance(field, list):
                # Flatten it into a single field, if it's not already one
                values = [" ".join(values)]
                is_sequential = False
            for value in values:
                if isinstance(value, str):
                    children.extend(self._transform_literal(value))
                else:
                    children.append(self._transform_ast(value))
            if is_sequential:
                children.append((self.END_OF_SEQ, []))
        return (node_id, children)

    def _detransform_ast(self, ast: ProcessedAST) -> CompressedAST:
        node_id, children = ast
        assert self._is_node(node_id)
        prod_name = self._detransform_node(node_id)
        prod = self.grammar.get_prod_by_ctr_name(prod_name)
        prod_id = self.grammar.prod2id[prod]

        fields = [None] * len(prod.fields)
        field_info = self.get_field_info(prod_id)

        if field_info.quals_idx is not None:
            with self._suppress(IndexError):
                qual_node = self._detransform_node(children[0][0])
                children = children[1:]
                quals = []
                if "Const" in qual_node: quals.append(self.CONST_PROD_ID)
                if "Volatile" in qual_node: quals.append(self.VOLATILE_PROD_ID)
                fields[field_info.quals_idx] = [(qual, []) for qual in quals]

        processed_children = []
        idx = 0
        while idx < len(children):
            child_node_id, _ = child = children[idx]
            idx += 1
            if child_node_id == self.NONE or child_node_id == self.END_OF_SEQ:
                processed_children.append(None)
            elif self._is_node(child_node_id):
                processed_children.append(self._detransform_ast(child))
            else:
                ids = [child_node_id]
                while idx < len(children):
                    child_node_id, _ = children[idx]
                    if self.strict:
                        assert not self._is_node(child_node_id)
                    elif self._is_node(child_node_id):
                        break
                    ids.append(child_node_id)
                    idx += 1
                    if child_node_id == self.EOS: break
                processed_children.append(self._detransform_literal(ids))

        idx = 0
        for field_idx in range(len(prod.fields)):
            if not field_info.keep_field[field_idx]: continue
            if prod.fields[field_idx].cardinality == "multiple" and prod.fields[field_idx].type.name != "IDENT":
                children = []
                while idx < len(processed_children):
                    child = processed_children[idx]
                    idx += 1
                    if child is None: break
                    children.append(child)
                if self.strict:
                    assert child is None
                fields[field_idx] = children
            else:
                with self._suppress(IndexError):
                    child = processed_children[idx]
                    idx += 1
                    fields[field_idx] = child
                if self.strict:
                    if prod.fields[field_idx].cardinality != "optional":
                        assert child is not None
                    if child is not None:
                        if prod.fields[field_idx].type.name in {"IDENT", "LITERAL"}:
                            assert isinstance(child, str)
                        else:
                            assert not isinstance(child, str)

        return (prod_id, fields)

    def transform_ast(self, ast: CompressedAST) -> ProcessedAST:
        if self.tree_bpe is not None:
            ast = self.tree_bpe.encode(ast)
        return self._transform_ast(ast)

    def detransform_ast(self, ast: ProcessedAST) -> CompressedAST:
        de_ast = self._detransform_ast(ast)
        if self.tree_bpe is not None:
            de_ast = self.tree_bpe.decode(de_ast)
        return de_ast


class ProcessState(flutes.PoolState):
    var_map: Dict[str, List[str]]

    def __init__(self, args: Args, bar: flutes.ProgressBarManager.Proxy):
        sys.setrecursionlimit(32768)
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.reserved_words = set(c_utils.RESERVED_WORDS)
        self.processor = T2TDataProcessor(args.asdl_path, args.spm_model_path, args.tree_bpe_path)

        vocab = self.processor.vocab
        with (self.output_dir / "vocab.pkl").open("wb") as f:
            word_list = [vocab.id2word[i] for i in range(len(vocab))]
            pickle.dump({"source": word_list, "target": word_list}, f)

        self.bar = bar
        self.stats = defaultdict(Portion)

    def process(self, path: Path, verify: bool = False, compress: bool = False) -> str:
        with path.open("rb") as f:
            data = pickle.load(f)

        examples = []
        json_examples = []
        for idx, _ex in enumerate(self.bar.new(data,  # update_frequency=0.01,
                                               desc=f"Worker {flutes.get_worker_id()}")):
            assert isinstance(_ex, tuple) and len(_ex) == 5
            ex = RawExampleSrc(*_ex)
            if ex.src_ast is None: continue
            self.processor.set_var_map({  # use decompiled variable name
                key: decomp for key, (decomp, orig) in ex.meta['var_names'].items()})

            src_ast = self.processor.transform_ast(ex.src_ast)
            tgt_ast = self.processor.transform_ast(ex.tgt_ast)
            if verify:
                self.processor.verify(src_ast, ex.src_ast, src=True)
                self.processor.verify(tgt_ast, ex.tgt_ast, src=False)
            examples.append((src_ast, tgt_ast))
            if not compress:
                json_examples.append({"source_ast": self.processor.expand(src_ast),
                                      "target_ast": self.processor.expand(tgt_ast)})
                if verify:
                    assert self.processor.compress(json_examples[-1]["source_ast"]) == src_ast
                    assert self.processor.compress(json_examples[-1]["target_ast"]) == tgt_ast

        output_path = self.output_dir / (path.relative_to(self.data_dir))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with output_path.with_suffix(".pkl").open("wb") as f:
                pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with output_path.with_suffix(".json").open("w") as f:
                json.dump(json_examples, f, separators=(',', ':'))

        return str(path)


def main():
    args = Args()
    assert 0.0 < args.data_percentage <= 1.0
    if args.pdb:
        flutes.register_ipython_excepthook()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with flutes.ProgressBarManager(verbose=args.show_progress) as manager, \
            flutes.safe_pool(args.n_procs, state_class=ProcessState, init_args=(args, manager.proxy)) as pool:
        data_files: List[Path] = []
        for path, subdir, files in os.walk(args.data_dir):
            cur_data_files = [Path(os.path.join(path, file))
                              for file in files if file.startswith("data") and file.endswith(".pkl")]
            if args.data_percentage < 1.0:
                cur_data_files = cur_data_files[:int(math.ceil(len(cur_data_files) * args.data_percentage))]
            data_files.extend(cur_data_files)
        for _ in manager.proxy.new(pool.imap_unordered(ProcessState.process, data_files,
                                                       kwds={"verify": args.verify, "compress": args.compress}),
                                   total=len(data_files), desc="Main"):
            pass


if __name__ == '__main__':
    main()
