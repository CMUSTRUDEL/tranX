# coding=utf-8
import functools
from collections import defaultdict
from typing import Callable, Iterator, List, NamedTuple, Optional, TypeVar

import numpy as np
import torch
from six.moves import cPickle as pickle
from torch.utils.data import DataLoader
from typing_extensions import Literal

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.transition_system import ApplyRuleAction, GenTokenAction, ReduceAction
from common.registerable import Registrable
from common.utils import Args
from components.action_info import ActionInfo
from components.vocab import Vocab
from model import nn_utils

T = TypeVar('T')


@Registrable.register("default_dataset")
class Dataset(object):
    def __init__(self, examples: List['Example']):
        self.examples = examples

    @property
    def all_source(self) -> List[List[str]]:
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self) -> List[List[str]]:
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path, args: Args, mode: str = "eval") -> 'Dataset':
        # `mode` can be "train" or "eval". The default dataset does not distinguish between two modes.
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    @classmethod
    def _collate_fn(cls, examples: List['Example'], collate_fn: Callable[[List['Example']], T],
                    decode_max_time_step: int) -> T:
        examples = [e for e in examples if len(e.tgt_actions) <= decode_max_time_step]
        return collate_fn(examples)

    def create_collate_fn(self, collate_fn: Callable[[List['Example']], T],
                          decode_max_time_step: int) -> Callable[[List['Example']], T]:
        return functools.partial(self._collate_fn, collate_fn=collate_fn, decode_max_time_step=decode_max_time_step)

    def batch_iter(self, batch_size: int, shuffle: bool = False,
                   collate_fn: Optional[Callable[[List['Example']], T]] = None,
                   *, decode_max_time_step: int, **kwargs) -> Iterator[List['Example']]:
        if collate_fn is None:
            # The old approach.
            index_arr = np.arange(len(self.examples))
            if shuffle:
                np.random.shuffle(index_arr)

            batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
            for batch_id in range(batch_num):
                batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
                batch_examples = [self.examples[i] for i in batch_ids]
                batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= decode_max_time_step]
                batch_examples.sort(key=lambda e: -len(e.src_sent))

                yield batch_examples
        else:
            collate_fn = self.create_collate_fn(collate_fn, decode_max_time_step)
            data_loader = DataLoader(self.examples, batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
            yield from data_loader

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, src_sent: List[str], tgt_actions: List[ActionInfo], tgt_code: List[str],
                 tgt_ast: AbstractSyntaxTree, src_actions: Optional[List[ActionInfo]] = None, idx: int = 0, meta=None):
        self.src_sent = src_sent
        self.src_actions = src_actions
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta


class BatchTensors(NamedTuple):
    apply_rule_idx_matrix: torch.LongTensor
    apply_rule_mask: torch.Tensor
    primitive_idx_matrix: torch.LongTensor
    gen_token_mask: torch.Tensor

    frontier_field_idx_matrix: torch.LongTensor
    # frontier_prod_idx_matrix: torch.LongTensor
    # frontier_type_idx_matrix: torch.LongTensor
    parent_idx_matrix: torch.LongTensor

    def to(self, device: torch.device) -> 'BatchTensors':
        return BatchTensors(
            apply_rule_idx_matrix=self.apply_rule_idx_matrix.to(device=device, non_blocking=True),
            apply_rule_mask=self.apply_rule_mask.to(device=device, non_blocking=True),
            primitive_idx_matrix=self.primitive_idx_matrix.to(device=device, non_blocking=True),
            gen_token_mask=self.gen_token_mask.to(device=device, non_blocking=True),
            frontier_field_idx_matrix=self.frontier_field_idx_matrix.to(device=device, non_blocking=True),
            # frontier_prod_idx_matrix = self.frontier_prod_idx_matrix.to(device=device, non_blocking=True),
            # frontier_type_idx_matrix = self.frontier_type_idx_matrix.to(device=device, non_blocking=True),
            parent_idx_matrix=self.parent_idx_matrix.to(device=device, non_blocking=True),
        )


class Batch(object):
    apply_rule_idx_matrix: torch.LongTensor
    apply_rule_mask: torch.Tensor
    primitive_idx_matrix: torch.LongTensor
    gen_token_mask: torch.Tensor

    primitive_copy_mask: torch.Tensor
    primitive_copy_token_idx_mask: torch.Tensor

    frontier_field_idx_matrix: torch.LongTensor
    frontier_prod_idx_matrix: torch.LongTensor
    frontier_type_idx_matrix: torch.LongTensor
    parent_idx_matrix: torch.LongTensor

    src_sents_var: torch.LongTensor
    src_token_mask: torch.BoolTensor
    src_tensors: Optional[BatchTensors]
    tgt_tensors: BatchTensors

    def __init__(self, examples: List[Example], grammar: ASDLGrammar, vocab: Vocab,
                 copy: bool = True, cuda: bool = False, src_repr_mode: Literal['text', 'action_seq'] = "text"):
        self.examples = examples
        self.examples.sort(key=lambda e: -len(e.src_sent))
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]

        self.copy = copy
        self.cuda = cuda
        self.device = torch.device("cuda" if cuda else "cpu")

        if src_repr_mode == "text":
            self.src_sents_len = [len(e.src_sent) for e in self.examples]
            self.src_tensors = None
        else:
            src_actions = [e.src_actions or [] for e in self.examples]
            self.src_sents_len = [len(e.src_actions) for e in self.examples]
            self.src_tensors = self._create_batch_tensors(src_actions, grammar, vocab)
        self.src_sents_var = nn_utils.to_input_variable(self.src_sents, vocab.source)
        self.src_token_mask = nn_utils.length_array_to_mask_tensor(self.src_sents_len)

        tgt_actions = [e.tgt_actions for e in self.examples]
        self.tgt_tensors = self._create_batch_tensors(tgt_actions, grammar, vocab)
        self.primitive_copy_mask, self.primitive_copy_token_idx_mask = \
            self._create_copy_tensors(tgt_actions, src_repr_mode)

        if self.cuda:
            self.to(self.device)

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, item):
        if hasattr(self.__dict__.get('tgt_tensors', None), item):
            # resolve on target side tensors
            return getattr(self.__dict__['tgt_tensors'], item)
        return super().__getattr__(item)

    def get_frontier_field_idx(self, grammar: ASDLGrammar, t: int) -> torch.LongTensor:
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(grammar.field2id[e.tgt_actions[t].frontier_field])
                # assert self.grammar.id2field[ids[-1]] == e.tgt_actions[t].frontier_field
            else:
                ids.append(0)

        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_prod_idx(self, grammar: ASDLGrammar, t: int) -> torch.LongTensor:
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(grammar.prod2id[e.tgt_actions[t].frontier_prod])
                # assert self.grammar.id2prod[ids[-1]] == e.tgt_actions[t].frontier_prod
            else:
                ids.append(0)

        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_field_type_idx(self, grammar: ASDLGrammar, t: int) -> torch.LongTensor:
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(grammar.type2id[e.tgt_actions[t].frontier_field.type])
                # assert self.grammar.id2type[ids[-1]] == e.tgt_actions[t].frontier_field.type
            else:
                ids.append(0)

        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _create_copy_tensors(self, batch_actions: List[List[ActionInfo]], src_repr_mode: Literal['text', 'action_seq']):
        batch_size = len(batch_actions)
        max_actions = max((len(xs) for xs in batch_actions), default=1)

        primitive_copy_mask = torch.zeros(max_actions, batch_size, dtype=torch.float)
        primitive_copy_token_idx_mask = torch.zeros(max_actions, batch_size, max(self.src_sents_len), dtype=torch.float)
        if not self.copy:
            return primitive_copy_mask, primitive_copy_token_idx_mask

        token_pos_lists = []
        if src_repr_mode == "text":
            for src_sent in self.src_sents:
                pos_lists = defaultdict(list)
                for idx, token in enumerate(src_sent):
                    pos_lists[token].append(idx)
                token_pos_lists.append(pos_lists)
        else:
            for example in self.examples:
                pos_lists = defaultdict(list)
                for idx, action_info in enumerate(example.src_actions):
                    if isinstance(action_info.action, GenTokenAction):
                        pos_lists[action_info.action.token].append(idx)
                token_pos_lists.append(pos_lists)

        for t in range(self.max_action_num):
            for e_id, e in enumerate(self.examples):
                if t < len(e.tgt_actions):
                    action = e.tgt_actions[t].action
                    if isinstance(action, GenTokenAction):
                        token = str(action.token)
                        pos_list = token_pos_lists[e_id][token]
                        if len(pos_list) > 0:
                            primitive_copy_mask[t, e_id] = 1
                            primitive_copy_token_idx_mask[t, e_id, pos_list] = 1.

        return primitive_copy_mask, primitive_copy_token_idx_mask

    def _create_batch_tensors(self, batch_actions: List[List[ActionInfo]], grammar: ASDLGrammar, vocab: Vocab):
        batch_size = len(batch_actions)
        max_actions = max((len(xs) for xs in batch_actions), default=1)

        apply_rule_idx_matrix = torch.zeros(max_actions, batch_size, dtype=torch.long)
        apply_rule_mask = torch.zeros(max_actions, batch_size, dtype=torch.float)
        primitive_idx_matrix = torch.zeros(max_actions, batch_size, dtype=torch.long)
        gen_token_mask = torch.zeros(max_actions, batch_size, dtype=torch.float)

        frontier_field_idx_matrix = torch.zeros(max_actions, batch_size, dtype=torch.long)
        # frontier_prod_idx_matrix = torch.zeros(max_actions, batch_size, dtype=torch.long)
        # frontier_type_idx_matrix = torch.zeros(max_actions, batch_size, dtype=torch.long)
        # frontier_type_idx_matrix[0] = grammar.type2id[grammar.root_type]
        parent_idx_matrix = torch.zeros(max_actions, batch_size, dtype=torch.long)

        for t in range(max_actions):
            for e_id, e in enumerate(self.examples):
                app_rule_idx = apply_rule = token_idx = gen_token = 0
                if t < len(e.tgt_actions):
                    action = e.tgt_actions[t].action
                    action_info = e.tgt_actions[t]
                    if t > 0:
                        frontier_field_idx_matrix[t, e_id] = grammar.field2id[action_info.frontier_field]
                        # frontier_prod_idx_matrix[t, e_id] = grammar.prod2id[action_info.frontier_prod]
                        # frontier_type_idx_matrix[t, e_id] = grammar.type2id[action_info.frontier_field.type]
                        parent_idx_matrix[t, e_id] = action_info.parent_t

                    if isinstance(action, ApplyRuleAction):
                        app_rule_idx = grammar.prod2id[action.production]
                        # assert self.grammar.id2prod[app_rule_idx] == action.production
                        apply_rule = 1
                    elif isinstance(action, ReduceAction):
                        app_rule_idx = len(grammar)
                        apply_rule = 1
                    else:
                        assert isinstance(action, GenTokenAction)
                        token = str(action.token)
                        token_idx = vocab.primitive[action.token]

                        # if token_can_copy is False or token_idx != vocab.primitive.unk_id:
                        #     # if the token is not copied, we can only generate this token from the vocabulary,
                        #     # even if it is a <unk>.
                        #     # otherwise, we can still generate it from the vocabulary
                        #     gen_token = 1
                        gen_token = 1
                        # If `gen_token != 1` here, then we're basically saying there's nothing to do for this token.
                        # This will give a probability of zero during training, resulting in an `inf` loss and
                        # `nan` weights.

                apply_rule_idx_matrix[t, e_id] = app_rule_idx
                apply_rule_mask[t, e_id] = apply_rule
                primitive_idx_matrix[t, e_id] = token_idx
                gen_token_mask[t, e_id] = gen_token

        return BatchTensors(
            apply_rule_idx_matrix=apply_rule_idx_matrix,
            apply_rule_mask=apply_rule_mask,
            primitive_idx_matrix=primitive_idx_matrix,
            gen_token_mask=gen_token_mask,
            frontier_field_idx_matrix=frontier_field_idx_matrix,
            # frontier_prod_idx_matrix=frontier_prod_idx_matrix,
            # frontier_type_idx_matrix=frontier_type_idx_matrix,
            parent_idx_matrix=parent_idx_matrix,
        )

    def to(self, device: torch.device) -> 'Batch':
        if self.src_tensors is not None:
            self.src_tensors = self.src_tensors.to(device)
        self.tgt_tensors = self.tgt_tensors.to(device)

        self.src_sents_var = self.src_sents_var.to(device)
        self.src_token_mask = self.src_token_mask.to(device)
        self.primitive_copy_mask = self.primitive_copy_mask.to(device)
        self.primitive_copy_token_idx_mask = self.primitive_copy_token_idx_mask.to(device)

        self.device = device
        return self

    @property
    def primitive_mask(self) -> torch.Tensor:
        return 1. - torch.eq(self.gen_token_mask + self.primitive_copy_mask, 0).float()
