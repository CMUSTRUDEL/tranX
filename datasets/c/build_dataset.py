import json
import multiprocessing as mp
import pickle
import sys
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, TypeVar, Union, cast

import flutes
import sentencepiece as spm
import ujson
from pycparser.c_ast import Node as ASTNode
from typing_extensions import TypedDict

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CTransitionSystem
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


class Repository(NamedTuple):
    repo: str  # owner/name
    file_path: str  # path to `matched_funcs.json`


def exception_handler(e: Exception, self: 'ParseState', repo: Repository) -> None:
    flutes.log_exception(e, f"Exception occurred when processing {repo.repo}", force_console=True)
    self.queue.put(self.END_SIGNATURE)


class MetaDict(TypedDict):
    var_names: Dict[str, Tuple[str, str]]
    repo: str
    hash: str
    func_name: str


class RawExample(NamedTuple):
    src: str  # concatenated source code
    tgt: str  # concatenated target code
    ast: CompressedAST  # compressed target AST
    meta: MetaDict  # meta data


class ParseState(flutes.PoolState):
    END_SIGNATURE = b"END_REPO"
    PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
    TOKEN_DELIMITER = constants.TOKEN_DELIMITER

    class Arguments(NamedTuple):
        asdl_path: str
        queue: 'mp.Queue[bytes]'
        spm_model_path: Optional[str] = None
        bar: Optional[flutes.ProgressBarManager.Proxy] = None
        verbose: bool = False
        sanity_check: bool = False

    def __init__(self, asdl_path: str, queue: 'mp.Queue[bytes]', spm_model_path: Optional[str] = None,
                 bar: Optional[flutes.ProgressBarManager.Proxy] = None, verbose: bool = False,
                 sanity_check: bool = False) -> None:
        sys.setrecursionlimit(32768)
        self.queue = queue
        self.verbose = verbose
        self.sanity_check = sanity_check
        self.bar = bar

        self.sp = None
        if spm_model_path is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model_path)

        with open(asdl_path) as f:
            asdl_text = f.read()
        self.grammar = ASDLGrammar.from_text(asdl_text)
        self.transition_system = CTransitionSystem(self.grammar, self.sp)

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
        if self.bar is not None:
            self.bar.new(total=lines, desc=f"Worker {flutes.get_worker_id()}")
            self.bar.update(postfix={"repo": repo.repo})
        with open(repo.file_path, "r") as f:
            for line in f:
                if not line: continue
                try:
                    ex = ujson.loads(line)
                except ValueError:
                    # `ujson` has a hard-coded depth limit of 1024. If limit is reached, fallback to built-in `json`.
                    ex = json.loads(line)
                src_tokens = ex['decompiled_tokens']
                tgt_tokens = ex['original_tokens']
                var_names = {k: (decomp, orig) for k, [decomp, orig] in ex['variable_names'].items()}

                # # Filter bad examples
                # if not (0 < len(src_tokens) <= 512 and
                #         0 < len(tgt_tokens) <= 512 and
                #         0.5 <= len(src_tokens) / len(tgt_tokens) <= 3):
                #     continue

                # Convert original AST to ASDL format.
                tgt_ast = ex['original_ast_json']
                tgt_ast_node = dict_to_ast(tgt_ast)
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
                }
                compressed_ast = self.transition_system.compress_ast(tgt_asdl_ast)
                example = RawExample(
                    src=self.TOKEN_DELIMITER.join(src_tokens),
                    tgt=self.TOKEN_DELIMITER.join(tgt_tokens),
                    ast=compressed_ast, meta=meta_info)

                # Dump it here; otherwise the queue thread will do all the dumping.
                example_ser = pickle.dumps(example, protocol=self.PICKLE_PROTOCOL)
                self.queue.put(example_ser)
                if self.bar is not None:
                    self.bar.update(1)
        self.queue.put(self.END_SIGNATURE)


def process_c_dataset(repos: List[Repository], spm_model_path: Optional[str] = None, n_procs: int = 0,
                      queue_size: int = 1024, verbose: bool = False, sanity_check: bool = False) \
        -> Iterator[RawExample]:
    if n_procs == 0:
        # Don't swallow exceptions in single-process mode.
        ParseState.parse_file = ParseState.parse_file.__wrapped__

    manager = mp.Manager()
    queue: 'mp.Queue[bytes]' = manager.Queue(queue_size)
    if not verbose:
        bar_manager = proxy = progress = None
    else:
        bar_manager = flutes.ProgressBarManager()
        proxy = bar_manager.proxy
        progress = proxy.new(total=len(repos), desc="Generating data")
    with flutes.safe_pool(
            n_procs, closing=[manager, bar_manager], state_class=ParseState, init_args=ParseState.Arguments(
                constants.ASDL_FILE_PATH, queue, spm_model_path, bar=proxy, verbose=verbose,
                sanity_check=sanity_check)) as pool:
        pool.map_async(ParseState.parse_file, repos)

        end_signals = 0
        while end_signals < len(repos):
            elem = queue.get()
            if elem == ParseState.END_SIGNATURE:
                if progress is not None:
                    progress.update(1)
                end_signals += 1
                continue

            ex: RawExample = pickle.loads(elem)
            yield ex
