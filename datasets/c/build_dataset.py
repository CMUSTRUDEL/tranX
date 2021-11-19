import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, TypeVar, Union, cast

import flutes
import sentencepiece as spm
import ujson
from pycparser.c_ast import Node as ASTNode
from typing_extensions import TypedDict

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CTransitionSystem, RobustCGenerator
from asdl.transition_system import CompressedAST
from . import constants

T = TypeVar('T')
JSONNode = Dict[str, Any]
MaybeList = Union[T, List[T]]


def assert_ast_equal(ast1: ASTNode, ast2: ASTNode) -> None:
    assert ast1.__class__ == ast2.__class__
    for name in ast1.attr_names:
        if ast1.__class__.__name__ == "Constant" and name == "type":
            continue  # special case; may have `unsigned long int` -> `int` or `float` -> `double`.
        assert getattr(ast1, name) == getattr(ast2, name)
    for child1, child2 in zip(ast1, ast2):
        if child1 and child2:
            assert_ast_equal(child1, child2)
        else:
            assert not child1 and not child2


def dict_to_ast(node_dict: JSONNode) -> ASTNode:
    r"""Recursively build an AST from dictionary representation. Coordinate information is discarded.
    """
    class_name = node_dict[constants.NODE_TYPE_ATTR]
    klass = c_utils.get_c_ast_node_class(class_name)

    # Create a new dict containing the key-value pairs which we can pass to node constructors.
    kwargs: Dict[str, Any] = {'coord': None}
    children: Dict[str, MaybeList[JSONNode]] = node_dict[constants.CHILDREN_ATTR]
    for name, child in children.items():
        if isinstance(child, list):
            kwargs[name] = [dict_to_ast(item) for item in child]
        else:
            kwargs[name] = dict_to_ast(child) if child is not None else None

    for key, value in node_dict.items():
        if key in constants.ATTRS:
            continue
        kwargs[key] = value  # must be primitive attributes

    return klass(**kwargs)


class Repository:
    repo: str  # owner/name
    _file_path: List[str]  # paths to `matched_funcs.json`; the first resolved path will be used

    def __init__(self, repo: str, paths: Union[str, List[str]]):
        self.repo = repo
        self._file_path = [paths] if isinstance(paths, str) else paths

    @property
    def file_path(self) -> str:
        for path in self._file_path:
            if os.path.exists(path):
                return path
        raise ValueError(f"Repository {self.repo} not found in data")


def exception_handler(e: Exception, self: 'ParseState', repo: Repository) -> None:
    flutes.log_exception(e, f"Exception occurred when processing {repo.repo}", force_console=True)
    self.queue.put(self.END_SIGNATURE)


class MetaDict(TypedDict):
    var_names: Dict[str, Tuple[str, str]]
    repo: str
    hash: str
    func_name: str
    raw_tgt_code: str


class RawExample(NamedTuple):
    src: str  # concatenated source code
    tgt: str  # concatenated target code
    ast: CompressedAST  # compressed target AST
    meta: MetaDict  # meta data


class RawExampleSrc(NamedTuple):
    src: str  # concatenated source code
    tgt: str  # concatenated target code
    src_ast: Optional[CompressedAST]  # compressed source AST
    tgt_ast: CompressedAST  # compressed target AST
    meta: MetaDict  # meta data


class ParseState(flutes.PoolState):
    END_SIGNATURE = b"END_REPO"
    PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
    TOKEN_DELIMITER = constants.TOKEN_DELIMITER

    class Arguments(NamedTuple):
        asdl_path: str
        queue: 'mp.Queue'
        spm_model_path: Optional[str] = None
        include_src_ast: bool = False
        bar: Optional[flutes.ProgressBarManager.Proxy] = None
        verbose: bool = False
        sanity_check: bool = False

    def __init__(self, asdl_path: str, queue: 'mp.Queue', spm_model_path: Optional[str] = None,
                 include_src_ast: bool = False,
                 bar: Optional[flutes.ProgressBarManager.Proxy] = None, verbose: bool = False,
                 sanity_check: bool = False) -> None:
        sys.setrecursionlimit(32768)
        self.queue = queue
        self.verbose = verbose
        self.sanity_check = sanity_check
        self.bar = bar
        self.include_src_ast = include_src_ast

        self.sp = None
        if spm_model_path is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model_path)

        with open(asdl_path) as f:
            asdl_text = f.read()
        self.grammar = ASDLGrammar.from_text(asdl_text)
        self.transition_system = CTransitionSystem(self.grammar, self.sp)
        self.generator = RobustCGenerator()

        self.reserved_words = set(c_utils.RESERVED_WORDS)
        self.ast_converter = c_utils.ASTConverter(self.grammar)

    def gather_strings(self, ast: AbstractSyntaxTree) -> List[str]:
        string = []
        if ast.production.constructor.name == "Constant":
            if cast(AbstractSyntaxTree, ast.fields[0].value).production.constructor.name == "StringLiteral":
                string.append(cast(str, ast.fields[1].value))
        else:
            for field in ast.fields:
                if field.type.name == "EXPR":
                    for value in field.as_value_list:
                        string += self.gather_strings(cast(AbstractSyntaxTree, value))
        return string

    @flutes.exception_wrapper(exception_handler)
    def parse_file(self, repo: Repository) -> None:
        lines = flutes.get_file_lines(repo.file_path)
        with open(repo.file_path) as f:
            if self.bar is not None:
                progress = self.bar.new(f, total=lines, desc=f"Worker {flutes.get_worker_id()}", update_frequency=0.1)
                self.bar.update(postfix={"repo": repo.repo})
            else:
                progress = f
            for line in progress:
                if not line: continue
                try:
                    ex = ujson.loads(line)
                except ValueError:
                    # `ujson` has a hard-coded depth limit of 1024. If limit is reached, fallback to built-in `json`.
                    ex = json.loads(line)
                # if self.include_src_ast and ex['decompiled_ast_json'] is None:
                #     continue

                src_tokens = ex['decompiled_tokens']
                var_names = {k: (decomp, orig) for k, [decomp, orig] in ex['variable_names'].items()}

                # # Filter bad examples
                # if not (0 < len(src_tokens) <= 512 and
                #         0 < len(tgt_tokens) <= 512 and
                #         0.5 <= len(src_tokens) / len(tgt_tokens) <= 3):
                #     continue

                # Convert original AST to ASDL format.
                tgt_ast = ex['original_ast_json']

                tgt_ast_node = dict_to_ast(tgt_ast)
                    
                code = self.generator.visit(tgt_ast_node)
                # Clean code of backslashes. The pycparser lexer hangs when encountering
                # string literals with backslashes.
                # Convert newlines to spaces first so that words are appropriately separated.
                code = code.replace("\\n", " ")
                code = code.replace("\\", "")
                tgt_tokens = self.transition_system.lexer.lex(code)
                tgt_asdl_ast = self.ast_converter.c_ast_to_asdl_ast(tgt_ast_node)
                if self.sanity_check:
                    assert_ast_equal(tgt_ast_node, self.ast_converter.asdl_ast_to_c_ast(tgt_asdl_ast))
                    tgt_actions = self.transition_system.get_actions(tgt_asdl_ast)
                    tgt_action_infos, tgt_hyp = self.get_action_infos(src_tokens, tgt_actions)
                    reconstruct_ast = self.ast_converter.asdl_ast_to_c_ast(tgt_hyp.tree)
                    assert_ast_equal(tgt_ast_node, reconstruct_ast)
                

                r"""
                About string literals in decompiled C code:
                
                1. In most cases, the string literals can be copied verbatim from the decompiled code.
                2. If the literals ends with '\n', the decompiled code might use `puts` and chomp off the newline.
                3. Non-ASCII characters are probably not supported, and those strings are not seen in the code.
                4. Short strings could be converted to integer literals. For example, for the original code:
                   ```c
                   char shellname[1050];
                   ...
                   strcat(shellname, "/myshell");
                   ```
                   It is decompiled to:
                   ```c
                   char *v3; char shellname[1050];
                   v3 = &shellname[strlen(shellname)];
                   *(QWORD *)v3 = 7812730952869309743LL;
                   v3[8] = 0;
                   ```
                """

                # # Split only identifiers and literals in the decompiled code.
                # if self.sp is not None:
                #     src_subwords = []
                #     for token in src_tokens:
                #         if token in self.reserved_words:
                #             src_subwords.append(token)
                #         else:
                #             src_subwords.extend(self.sp.EncodeAsPieces(token))
                #     src_tokens = src_subwords

                meta_info: MetaDict = {
                    "var_names": var_names,
                    "repo": repo.repo,
                    "hash": ex['binary_hash'],
                    "func_name": ex['func_name'],
                    "raw_tgt_code": self.TOKEN_DELIMITER.join(ex['original_tokens']),
                }
                compressed_tgt_ast = self.transition_system.compress_ast(tgt_asdl_ast)

                src = self.TOKEN_DELIMITER.join(src_tokens)
                tgt = self.TOKEN_DELIMITER.join(tgt_tokens)
                if self.include_src_ast:
                    src_ast = ex['decompiled_ast_json']
                    if src_ast is None:
                        compressed_src_ast = None
                    else:
                        src_ast_node = dict_to_ast(src_ast)
                        src_asdl_ast = self.ast_converter.c_ast_to_asdl_ast(src_ast_node)
                        compressed_src_ast = self.transition_system.compress_ast(src_asdl_ast)
                    example = RawExampleSrc(
                        src=src, tgt=tgt, src_ast=compressed_src_ast, tgt_ast=compressed_tgt_ast, meta=meta_info)
                else:
                    example = RawExample(src=src, tgt=tgt, ast=compressed_tgt_ast, meta=meta_info)

                self.queue.put(example)
        self.queue.put(self.END_SIGNATURE)


def process_c_dataset(repos: List[Repository], spm_model_path: Optional[str] = None, n_procs: int = 0,
                      include_src_ast: bool = False,
                      queue_size: int = 1024, verbose: bool = False, sanity_check: bool = False) \
        -> Iterator[RawExample]:
    if n_procs == 0:
        # Don't swallow exceptions in single-process mode.
        ParseState.parse_file = ParseState.parse_file.__wrapped__

    manager = mp.Manager()
    queue: 'mp.Queue' = manager.Queue(queue_size)
    bar_manager = flutes.ProgressBarManager(verbose=verbose)
    proxy = bar_manager.proxy
    progress = proxy.new(total=len(repos), desc="Generating data")
    init_args = ParseState.Arguments(
        constants.ASDL_FILE_PATH, queue, spm_model_path, include_src_ast=include_src_ast,
        bar=proxy, verbose=verbose, sanity_check=sanity_check)
    with flutes.safe_pool(
            n_procs, closing=[manager, bar_manager], state_class=ParseState, init_args=init_args) as pool:
        pool.map_async(ParseState.parse_file, repos)

        end_signals = 0
        while end_signals < len(repos):
            elem = queue.get()
            if elem == ParseState.END_SIGNATURE:
                progress.update(1)
                end_signals += 1
                continue

            # ex: RawExample = pickle.loads(elem)
            ex: RawExample = elem
            yield ex
