import pickle
import random
from pathlib import Path
from typing import Callable, Iterator, List, Optional, TypeVar

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from asdl.asdl import ASDLGrammar
from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CHypothesis, CTransitionSystem
from common.registerable import Registrable
from components.action_info import ActionInfo
from components.dataset import Dataset, Example
from datasets.c.build_dataset import RawExample
from .constants import ASDL_FILE_PATH, TOKEN_DELIMITER

T = TypeVar('T')


class CIterDataset(IterableDataset):
    vocab: spm.SentencePieceProcessor
    grammar: ASDLGrammar
    transition_system: CTransitionSystem

    RESERVED_WORDS = set(c_utils.RESERVED_WORDS)
    DEFAULT_MAX_ACTIONS = 512
    DEFAULT_SENT_LENGTH = 512

    def __init__(self, file_paths: List[Path], vocab_path: Path):
        self.file_paths = file_paths
        self.vocab_path = vocab_path
        self.shuffle = True
        self.max_actions = self.DEFAULT_MAX_ACTIONS
        self.max_src_len = self.DEFAULT_SENT_LENGTH

    def process(self, example: RawExample) -> Optional[Example]:
        src = example.src.split(TOKEN_DELIMITER)
        tgt = example.tgt.split(TOKEN_DELIMITER)
        src_tokens = []
        # src_primitive_map: Dict[str, Tuple[int, int]] = {}
        # src_token_pos: Dict[str, int] = {}
        for word in src:
            if word in self.RESERVED_WORDS:
                src_tokens.append(c_utils.SPM_SPACE + word)
            else:
                subwords = self.vocab.EncodeAsPieces(word)
                # src_primitive_map[word] = len(src_tokens), len(subwords)
                # for idx, subword in enumerate(subwords):
                #     if subword not in src_token_pos:
                #         src_token_pos[subword] = len(src_tokens) + idx
                src_tokens.extend(subwords)
        if len(src_tokens) > self.max_src_len:
            return None

        tgt_actions = self.transition_system.get_actions_from_compressed(example.ast)
        if len(tgt_actions) > self.max_actions:
            return None

        action_infos = []
        hyp = CHypothesis(use_subword=self.vocab is not None)
        # copied = [False] * len(src_tokens)
        # primitive_action_infos: List[ActionInfo] = []
        for t, action in enumerate(tgt_actions):
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

        return Example(src_sent=src_tokens, tgt_ast=None, tgt_code=tgt, tgt_actions=action_infos, meta=example.meta)

    def iterate_dataset(self, shuffle: Optional[bool] = None) -> Iterator[Example]:
        if shuffle is None:
            shuffle = self.shuffle
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load(str(self.vocab_path))
        with open(ASDL_FILE_PATH, "r") as f:
            self.grammar = ASDLGrammar.from_text(f.read())
        self.transition_system = CTransitionSystem(self.grammar, self.vocab)

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            split_size = len(self.file_paths) // worker_info.num_workers
            files = self.file_paths[(split_size * worker_id):(split_size * (worker_id + 1))]
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

    def __init__(self, file_paths: List[Path], vocab_path: Path):
        self.dataset = CIterDataset(file_paths, vocab_path)
        self.dataloader: Optional[DataLoader] = None
        self.random_seed = 19260817

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
    def from_bin_file(data_dir: str) -> 'CDataset':
        path = Path(data_dir)
        file_paths = sorted([file for file in path.iterdir() if file.name.startswith("data")])
        return CDataset(file_paths, vocab_path=path / "vocab.model")

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
        print("Begin dataset iteration", flush=True)
        yield from self.dataloader

    def __iter__(self):
        return self.dataset.iterate_dataset(shuffle=False)

    def __len__(self):
        return len(self.dataset.file_paths)  # we don't know, but just return some non-zero number
