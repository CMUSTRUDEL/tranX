import multiprocessing as mp
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import flutes
import numpy as np
import sentencepiece as spm
import ujson
from pycparser.c_ast import Node as _C_ASTNode
from tqdm import tqdm

from asdl.asdl import ASDLGrammar
from asdl.hypothesis import Hypothesis
from asdl.lang.c import CTransitionSystem, c_utils
from asdl.transition_system import Action, GenTokenAction
from components.action_info import ActionInfo
from components.dataset import Example

T = TypeVar('T')
JSONNode = Dict[str, Any]
MaybeList = Union[T, List[T]]

NODE_TYPE_ATTR = "_t"
CHILDREN_ATTR = "_c"
TOKEN_POS_ATTR = "_p"

ASDL_FILE_PATH = "asdl/lang/c/c_asdl.txt"


class ASTNode(_C_ASTNode):
    attr_names: Sequence[str]


def assert_ast_equal(ast1: ASTNode, ast2: ASTNode) -> None:
    assert ast1.__class__ == ast2.__class__
    for name in ast1.attr_names:
        assert getattr(ast1, name) == getattr(ast2, name)
    for child1, child2 in zip(ast1, ast2):
        if child1 and child2:
            assert_ast_equal(child1, child2)
        else:
            assert not child1 and not child2


def dict_to_ast(node_dict: JSONNode) -> ASTNode:
    r"""Recursively build an AST from dictionary representation. Coordinate information is discarded.
    """
    class_name = node_dict[NODE_TYPE_ATTR]
    klass = c_utils.get_c_ast_node_class(class_name)

    # Create a new dict containing the key-value pairs which we can pass to node constructors.
    kwargs: Dict[str, Any] = {'coord': None}
    children: Dict[str, MaybeList[JSONNode]] = node_dict[CHILDREN_ATTR]
    for name, child in children.items():
        if isinstance(child, list):
            kwargs[name] = [dict_to_ast(item) for item in child]
        else:
            kwargs[name] = dict_to_ast(child) if child is not None else None

    for key, value in node_dict.items():
        if key in [NODE_TYPE_ATTR, CHILDREN_ATTR, TOKEN_POS_ATTR]:
            continue
        kwargs[key] = value  # must be primitive attributes

    return klass(**kwargs)


class Repository(NamedTuple):
    repo: str  # owner/name
    file_path: str  # path to `matched_funcs.json`


def exception_handler(e: Exception, self: 'ParseState', repo: Repository) -> None:
    flutes.log_exception(e, f"Exception occurred when processing {repo.repo}", force_console=True)
    self.queue.put(self.END_SIGNATURE)


class ParseState(flutes.PoolState):
    END_SIGNATURE = b"END_REPO"
    PICKLE_PROTOCOL = 4

    class Arguments(NamedTuple):
        asdl_path: str
        spm_model_path: str
        queue: 'mp.Queue[bytes]'
        bar: Optional[flutes.ProgressBarManager.Proxy] = None
        verbose: bool = False
        sanity_check: bool = False

    def __init__(self, asdl_path: str, spm_model_path: str, queue: 'mp.Queue[bytes]',
                 bar: Optional[flutes.ProgressBarManager.Proxy] = None, verbose: bool = False,
                 sanity_check: bool = False) -> None:
        self.queue = queue
        self.verbose = verbose
        self.sanity_check = sanity_check
        self.bar = bar

        with open(asdl_path) as f:
            asdl_text = f.read()
        self.grammar = ASDLGrammar.from_text(asdl_text)
        self.transition_system = CTransitionSystem(self.grammar)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_path)

    def get_action_infos(self, src_query: List[str], tgt_actions: List[Action]) -> Tuple[List[ActionInfo], Hypothesis]:
        action_infos = []
        hyp = Hypothesis()
        src_token_index = {token: idx for idx, token in enumerate(src_query)}
        for t, action in enumerate(tgt_actions):
            action_info = ActionInfo(action)
            action_info.t = t
            if hyp.frontier_node:
                action_info.parent_t = hyp.frontier_node.created_time
                action_info.frontier_prod = hyp.frontier_node.production
                action_info.frontier_field = hyp.frontier_field.field

            if isinstance(action, GenTokenAction):
                tok_src_idx = src_token_index.get(str(action.token), None)
                if tok_src_idx is not None:
                    action_info.copy_from_src = True
                    action_info.src_token_position = tok_src_idx

            hyp.apply_action(action)
            action_infos.append(action_info)

        return action_infos, hyp

    @flutes.exception_wrapper(exception_handler)
    def parse_file(self, repo: Repository) -> None:
        lines = flutes.get_file_lines(repo.file_path)
        if self.bar is not None:
            self.bar.new(total=lines, desc=f"Worker {flutes.get_worker_id()}")
            self.bar.update(postfix={"repo": repo.repo})
        with open(repo.file_path, "r") as f:
            for line in f:
                if not line: continue
                ex = ujson.loads(line)
                src_tokens = ex['decompiled_tokens']
                tgt_tokens = ex['original_tokens']

                # Filter bad examples
                if not (0 < len(src_tokens) <= 512 and
                        0 < len(tgt_tokens) <= 512 and
                        0.5 <= len(src_tokens) / len(tgt_tokens) <= 3):
                    continue

                tgt_ast = ex['original_ast_json']
                tgt_ast_node = dict_to_ast(tgt_ast)
                tgt_asdl_ast = c_utils.c_ast_to_asdl_ast(tgt_ast_node, self.transition_system.grammar)
                tgt_actions = self.transition_system.get_actions(tgt_asdl_ast)
                tgt_action_infos, tgt_hyp = self.get_action_infos(src_tokens, tgt_actions)
                if self.sanity_check:
                    reconstruct_ast = c_utils.asdl_ast_to_c_ast(tgt_hyp.tree, self.transition_system.grammar)
                    assert_ast_equal(tgt_ast_node, reconstruct_ast)

                meta_info = {
                    "repo": repo.repo,
                    "hash": ex['binary_hash'],
                    "func_name": ex['func_name'],
                }
                example = Example(src_sent=src_tokens, tgt_code=tgt_tokens,
                                  tgt_ast=tgt_asdl_ast, tgt_actions=tgt_action_infos, meta=meta_info)

                # Dump it here; otherwise the queue thread will do all the dumping.
                example_ser = pickle.dumps(example, protocol=self.PICKLE_PROTOCOL)
                self.queue.put(example_ser)
                if self.bar is not None:
                    self.bar.update(1)
        self.queue.put(self.END_SIGNATURE)


def process_c_dataset(repos: List[Repository], spm_model_path: str,
                      vocab_freq_cutoff: int = 15, n_procs: int = 0,
                      queue_size: int = 1024, verbose: bool = False) -> List[Example]:
    examples = []
    action_len = []

    manager = mp.Manager()
    queue: 'mp.Queue[bytes]' = manager.Queue(queue_size)
    bar_manager = flutes.ProgressBarManager()
    progress = bar_manager.proxy.new(total=len(repos), desc="Generating data")
    with flutes.safe_pool(n_procs, state_class=ParseState, init_args=ParseState.Arguments(
            ASDL_FILE_PATH, spm_model_path, queue, bar=bar_manager.proxy, verbose=verbose, sanity_check=True)) as pool:
        pool.map_async(ParseState.parse_file, repos)

        end_signals = 0
        while end_signals < len(repos):
            elem = queue.get()
            if elem == ParseState.END_SIGNATURE:
                progress.update(1)
                end_signals += 1
                continue

            ex: Example = pickle.loads(elem)
            examples.append(ex)
            action_len.append(len(ex.tgt_actions))

    if verbose:
        print(f"Max action len: {max(action_len)}")
        print(f"Avg action len: {np.average(action_len)}")
        print(f"Actions larger than 100: {sum(int(x > 100) for x in action_len)}")

    return examples
