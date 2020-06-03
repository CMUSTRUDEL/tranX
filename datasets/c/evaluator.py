from typing import List, Optional

import nltk
import numpy as np

from asdl.hypothesis import Hypothesis
from asdl.transition_system import TransitionSystem
from common.registerable import Registrable
from components.evaluator import Evaluator

__all__ = [
    "CEvaluator",
]


@Registrable.register('c_evaluator')
class CEvaluator(Evaluator):
    def __init__(self, transition_system: Optional[TransitionSystem] = None, args=None):
        super().__init__(transition_system, args)
        self.default_metric = "bleu"

    def evaluate_dataset(self, examples, decode_results, fast_mode=False, args=None):
        correct_array = []
        oracle_array = []
        reference = []
        hypothesis = []
        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list: List[Hypothesis] = hyp_list[:1]

            if hyp_list:
                ref_code_tokens = example.tgt_code
                reference.append([ref_code_tokens])
                for hyp_id, hyp in enumerate(hyp_list):
                    hyp_code_tokens = self.transition_system.tokenize_code(hyp.code)
                    if hyp_id == 0:
                        hypothesis.append(hyp_code_tokens)
                    is_correct = ref_code_tokens == hyp_code_tokens
                    hyp.is_correct = is_correct

                correct_array.append(hyp_list[0].is_correct)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
            else:
                correct_array.append(False)
                oracle_array.append(False)

        acc = np.average(correct_array)
        bleu = nltk.translate.bleu_score.corpus_bleu(reference, hypothesis)

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc,
                            bleu=bleu)

        return eval_results
