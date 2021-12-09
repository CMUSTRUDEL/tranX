from typing import List, Optional

import nltk
import numpy as np

from asdl.hypothesis import Hypothesis
from asdl.transition_system import TransitionSystem, GenTokenAction, ApplyRuleAction, ReduceAction
from common.registerable import Registrable
from common.utils import Args
from components.evaluator import Evaluator

__all__ = [
    "CEvaluator",
]


@Registrable.register('c_evaluator')
class CEvaluator(Evaluator):
    def __init__(self, transition_system: Optional[TransitionSystem] = None, args=None):
        super().__init__(transition_system, args)
        self.default_metric = "bleu"

    def evaluate_dataset(self, examples, decode_results, all_results=None, fast_mode=False, args: Optional[Args] = None):
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
        
        # Compute a BLEU score based off of actions. Because some hypothesized action sequences are not printable,
        # it is difficult to determine how close the model's predictions are to being correct. Completing a BLEU
        # score on actions  helps with this.
        if all_results:
            reference = []
            hypothesis = []
            for example, hyp_list in zip(examples, all_results):
                if hyp_list:

                    # Build the reference sequence of actions as strings.
                    ref_actions = []
                    for actioninfo in example.tgt_actions:
                        if isinstance(actioninfo.action, ApplyRuleAction):
                            ref_actions.append(actioninfo.action.production.constructor.name)
                        elif isinstance(actioninfo.action, GenTokenAction):
                            ref_actions.append(actioninfo.action.token)
                        else:
                            assert isinstance(actioninfo.action, ReduceAction)
                            ref_actions.append("Reduce")

                    # Build the hypothesis sequence of actions as strings.
                    hyp_actions = []
                    for action in hyp_list[0].actions: # represent only the top hypothesis
                        if isinstance(action, ApplyRuleAction):
                            hyp_actions.append(action.production.constructor.name)
                        elif isinstance(action, GenTokenAction):
                            hyp_actions.append(action.token)
                        else:
                            assert isinstance(action, ReduceAction)
                            hyp_actions.append("Reduce")
                    
                    reference.append([ref_actions])
                    hypothesis.append(hyp_actions)

            action_bleu = nltk.translate.bleu_score.corpus_bleu(reference, hypothesis)
            eval_results['action_bleu'] = action_bleu

        return eval_results
