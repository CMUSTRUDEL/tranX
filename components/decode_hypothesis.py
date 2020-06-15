# coding=utf-8
from typing import List

from asdl.hypothesis import Hypothesis
from asdl.lang.c.c_transition_system import CHypothesis
from components.action_info import ActionInfo


class DecodeHypothesis(Hypothesis):
    action_infos: List[ActionInfo]

    def __init__(self):
        super(DecodeHypothesis, self).__init__()

        self.action_infos = []
        self.code = None

    def clone_and_apply_action_info(self, action_info: ActionInfo) -> 'DecodeHypothesis':
        action = action_info.action

        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)

        return new_hyp

    def copy(self) -> 'DecodeHypothesis':
        new_hyp = self.__class__()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.action_infos = list(self.action_infos)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t
        new_hyp.code = self.code

        new_hyp.update_frontier_info()

        return new_hyp


class CDecodeHypothesis(CHypothesis, DecodeHypothesis):
    pass
