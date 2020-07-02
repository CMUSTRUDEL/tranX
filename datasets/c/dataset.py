import pickle
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, TypeVar, Union

import sentencepiece as spm
import torch
import torch.utils.data.dataloader
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torch.utils.data._utils.fetch import _IterableDatasetFetcher, _MapDatasetFetcher
from typing_extensions import Literal

from asdl.asdl import ASDLGrammar
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CHypothesis, CTransitionSystem
from asdl.transition_system import Action, ApplyRuleAction
from asdl.tree_bpe import TreeBPE
from common.registerable import Registrable
from common.utils import Args
from components.action_info import ActionInfo
from components.dataset import Dataset, Example
from datasets.c.build_dataset import RawExample, RawExampleSrc
from .constants import ASDL_FILE_PATH, TOKEN_DELIMITER

__all__ = [
    "CDataset",
]

T = TypeVar('T')


class VarReplacingCTransitionSystem(CTransitionSystem):
    var_map: Dict[str, List[str]]

    def __init__(self, grammar: ASDLGrammar, spm_model: Optional[spm.SentencePieceProcessor] = None):
        super().__init__(grammar, spm_model)

    def set_var_map(self, var_map: Dict[str, List[str]]) -> None:
        self.var_map = var_map

    def _tokenize(self, value: str) -> List[str]:
        if value in self.var_map:
            return self.var_map[value]
        return super()._tokenize(value)


class CIterDataset(IterableDataset):
    vocab: spm.SentencePieceProcessor
    transition_system: CTransitionSystem
    src_transition_system: VarReplacingCTransitionSystem
    tree_bpe: Optional[TreeBPE]

    RESERVED_WORDS = set(c_utils.RESERVED_WORDS)
    DEFAULT_MAX_ACTIONS = 512
    DEFAULT_SENT_LENGTH = 512

    def __init__(self, file_paths: List[Path], vocab_path: Path, mode: str,
                 variable_name_mode: Literal['decompiled', 'original'] = "decompiled",
                 tree_bpe_model: Optional[str] = None,
                 max_actions: int = DEFAULT_MAX_ACTIONS, max_src_len: int = DEFAULT_SENT_LENGTH,
                 max_tokens_per_batch: Optional[int] = None, src_repr_mode: Literal['text', 'action_seq'] = "text",
                 src_action_seq_tree_bpe: bool = True):
        self.file_paths = file_paths
        self.vocab_path = vocab_path
        self.shuffle = True
        self.max_actions = max_actions
        self.max_src_len = max_src_len
        self.mode = mode
        self.variable_name_mode = variable_name_mode
        self.var_name_idx = {"decompiled": 0, "original": 1}[variable_name_mode]
        self.tree_bpe_path = tree_bpe_model
        self.max_tokens_per_batch = max_tokens_per_batch
        self.src_repr_mode = src_repr_mode
        self.src_action_seq_tree_bpe = src_action_seq_tree_bpe

    def _create_action_infos(self, actions: List[Action]) -> List[ActionInfo]:
        action_infos = []
        hyp = CHypothesis(use_subword=self.vocab is not None)
        # copied = [False] * len(src_seq)
        # primitive_action_infos: List[ActionInfo] = []
        for t, action in enumerate(actions):
            action_info = ActionInfo(action)
            action_info.t = t
            if hyp.frontier_node:
                action_info.parent_t = hyp.frontier_node.created_time
                action_info.frontier_prod = hyp.frontier_node.production
                action_info.frontier_field = hyp.frontier_field.field

            # if isinstance(action, GenTokenAction):
            #     if not action.is_stop_signal():
            #         primitive_action_infos.append(action_info)
            #     else:
            #         assert len(primitive_action_infos) > 0
            #         word = "".join([info.action.token for info in primitive_action_infos])
            #         word = word.replace(c_utils.SPM_SPACE, " ").strip()
            #         if word in src_primitive_map:
            #             src_idx, word_len = src_primitive_map[word]
            #             assert word_len == len(primitive_action_infos)
            #             for idx, action_info in enumerate(primitive_action_infos):
            #                 action_info.copy_from_src = True
            #                 action_info.src_token_position = src_idx + idx
            #             copied[src_idx:(src_idx + word_len)] = [True] * word_len
            #         else:
            #             for action_info in primitive_action_infos:
            #                 src_idx = src_token_pos.get(action_info.action.token, None)
            #                 if src_idx is not None:
            #                     action_info.copy_from_src = True
            #                     action_info.src_token_position = src_idx
            #                     copied[src_idx] = True
            #         primitive_action_infos = []

            # TBH we don't really care whether the token is copied or not; the information is not used anyway.

            hyp.apply_action(action)
            action_infos.append(action_info)
        return action_infos

    def process(self, example: Union[RawExample, RawExampleSrc]) -> Optional[Example]:
        src = example.src.split(TOKEN_DELIMITER)
        tgt = example.tgt.split(TOKEN_DELIMITER)
        var_map = example.meta['var_names']
        # src_primitive_map: Dict[str, Tuple[int, int]] = {}
        # src_token_pos: Dict[str, int] = {}

        src_tokens = []
        cur_var_map: Dict[str, List[str]] = {}
        for word in src:
            if word in self.RESERVED_WORDS:
                src_tokens.append(c_utils.SPM_SPACE + word)
            else:
                if word in var_map:
                    new_name = var_map[word][self.var_name_idx]
                    subwords = self.vocab.EncodeAsPieces(new_name)
                    cur_var_map[word] = subwords
                else:
                    subwords = self.vocab.EncodeAsPieces(word)
                # src_primitive_map[word] = len(src_seq), len(subwords)
                # for idx, subword in enumerate(subwords):
                #     if subword not in src_token_pos:
                #         src_token_pos[subword] = len(src_seq) + idx
                src_tokens.extend(subwords)

        if self.src_repr_mode == "text":
            assert isinstance(example, RawExample)
            src_seq = src_tokens
        else:
            assert isinstance(example, RawExampleSrc)
            ast = example.src_ast
            if ast is None:
                # Some examples in dev/test sets have no parsable decompiled code.
                # To prevent errors, we still give them at least one action.
                src_seq = [ApplyRuleAction(self.transition_system.grammar.get_prod_by_ctr_name("FileAST"))]
            else:
                if self.tree_bpe is not None and self.src_action_seq_tree_bpe:
                    ast = self.tree_bpe.encode(ast)
                self.src_transition_system.set_var_map(cur_var_map)
                src_seq = self.src_transition_system.get_actions_from_compressed(ast)

        if len(src_seq) > self.max_src_len:
            if self.mode == "train":
                return None
            src_seq = src_seq[:self.max_src_len]  # truncate under eval mode

        if self.src_repr_mode == "text":
            ast = example.ast
            src_action_infos = None
        else:
            ast = example.tgt_ast
            src_action_infos = self._create_action_infos(src_seq)
        if self.tree_bpe is not None:
            ast = self.tree_bpe.encode(ast)
        tgt_actions = self.transition_system.get_actions_from_compressed(ast)
        if len(tgt_actions) > self.max_actions and self.mode == "train":
            return None
        tgt_action_infos = self._create_action_infos(tgt_actions)

        return Example(src_sent=src_tokens, src_actions=src_action_infos,
                       tgt_ast=None, tgt_code=tgt, tgt_actions=tgt_action_infos, meta=example.meta)

    def iterate_dataset(self, shuffle: Optional[bool] = None) -> Iterator[Example]:
        # Resource initialization is postponed to this point, so that the resources are initialized on the worker
        # processes.
        if shuffle is None:
            shuffle = self.shuffle
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load(str(self.vocab_path))
        with open(ASDL_FILE_PATH, "r") as f:
            grammar = ASDLGrammar.from_text(f.read())
        self.tree_bpe = None
        if self.tree_bpe_path is not None:
            self.tree_bpe = TreeBPE.load(self.tree_bpe_path)
            grammar = self.tree_bpe.patch_grammar(grammar)
        self.transition_system = CTransitionSystem(grammar, self.vocab)
        if self.src_repr_mode == "action_seq":
            self.src_transition_system = VarReplacingCTransitionSystem(grammar, self.vocab)

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # Interleave all files between workers, starting from worker 0.
            files = [self.file_paths[idx] for idx in range(worker_id, len(self.file_paths), worker_info.num_workers)]
        else:
            worker_id = "main"
            files = self.file_paths.copy()

        if shuffle:
            random.shuffle(files)
        for file in files:
            with file.open("rb") as f:
                data: List[RawExample] = pickle.load(f)
            print(f"Worker {worker_id}: Loaded file {file}", flush=True)
            if shuffle:
                random.shuffle(data)
            yield from filter(lambda e: e is not None, map(self.process, data))

    def __iter__(self) -> Iterator[Example]:
        return self.iterate_dataset(shuffle=self.shuffle)


@Registrable.register("c_dataset")
class CDataset(Dataset):
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return Dataset(args[0])
        if 'examples' in kwargs:
            return Dataset(kwargs['examples'])
        return super().__new__(cls)

    def __init__(self, file_paths: List[Path], vocab_path: Path, mode: str, args: Args):
        self.args = args
        self.dataset = CIterDataset(file_paths, vocab_path, mode, args.variable_name, args.tree_bpe_model,
                                    args.max_actions, args.max_src_len, args.max_tokens_per_batch, args.src_repr_mode)
        self.dataloader: Optional[DataLoader] = None
        self.random_seed = 19260817
        self.mode = mode

    def set_random_seed(self, seed: int) -> None:
        self.random_seed = seed

    def worker_init_fn(self, worker_id) -> None:
        import numpy as np
        seed = self.random_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(int(seed * 13 / 7))

    @staticmethod
    def from_bin_file(data_dir: str, args: Args, mode: str = "eval") -> 'CDataset':
        path = Path(data_dir)
        file_paths = sorted([file for file in path.iterdir() if file.name.startswith("data")])
        return CDataset(file_paths, vocab_path=path / "vocab.model", mode=mode, args=args)

    def batch_iter(self, batch_size: int, shuffle: bool = False,
                   collate_fn: Optional[Callable[[List['Example']], T]] = None,
                   *, decode_max_time_step: int, **kwargs) -> Iterator[List['Example']]:
        assert collate_fn is not None
        self.dataset.shuffle = shuffle
        self.dataset.max_actions = decode_max_time_step
        if self.dataloader is None:
            collate_fn = self.create_collate_fn(collate_fn, decode_max_time_step)
            self.dataloader = DataLoader(
                self.dataset, batch_size, collate_fn=collate_fn, worker_init_fn=self.worker_init_fn, **kwargs)
            if self.args.max_tokens_per_batch is not None:
                # PyTorch's stupid design makes it hard to do anything other than the most ordinary stuff.
                # Here we patch `_DatasetKind.create_fetcher` to use our custom fetcher.
                torch.utils.data.dataloader._DatasetKind.create_fetcher = create_fetcher
        print("Begin dataset iteration", flush=True)
        yield from self.dataloader

    def __iter__(self):
        return self.dataset.iterate_dataset(shuffle=False)

    def __len__(self):
        return len(self.dataset.file_paths)  # we don't know, but just return some non-zero number


def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
    if kind == torch.utils.data.dataloader._DatasetKind.Iterable:
        # Use our custom fetcher.
        return DynamicBatchFetcher(dataset, auto_collation, collate_fn, drop_last)
    else:
        return _MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class DynamicBatchFetcher(_IterableDatasetFetcher):
    max_src_len: int
    max_tgt_len: int
    cur_batch_size: int

    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.max_tokens_per_batch = dataset.max_tokens_per_batch
        self._previous_item = None

    def reset_batch(self) -> None:
        self.max_src_len = 0
        self.max_tgt_len = 0
        self.cur_batch_size = 0

    def add_example(self, ex: Optional[Example]) -> bool:
        if ex is None:
            return False
        max_src_len = max(self.max_src_len, len(ex.src_sent))
        max_tgt_len = max(self.max_tgt_len, len(ex.tgt_actions))
        if (self.cur_batch_size + 1) * max(max_src_len, max_tgt_len) > self.max_tokens_per_batch:
            return False
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.cur_batch_size += 1
        return True

    def fetch(self, possibly_batched_index):
        if self.max_tokens_per_batch is None:
            return super().fetch(possibly_batched_index)

        self.reset_batch()
        data = []
        if self._previous_item is not None:
            if not self.add_example(self._previous_item):
                raise ValueError("Batching strategy refused to add example to empty batch")
            data.append(self._previous_item)
            self._previous_item = None
        while True:
            try:
                item = next(self.dataset_iter)
            except StopIteration:
                break
            if self.add_example(item):
                data.append(item)
            else:
                self._previous_item = item
                break
        if len(data) == 0:
            raise StopIteration
        return self.collate_fn(data)
