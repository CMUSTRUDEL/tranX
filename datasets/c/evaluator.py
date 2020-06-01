from typing import List, Optional

import nltk
import numpy as np

from asdl.hypothesis import Hypothesis
from asdl.transition_system import TransitionSystem
from common.registerable import Registrable
from components.evaluator import Evaluator

from datasets.tree_bpe import TreeBPE

__all__ = [
    "CEvaluator",
]


@Registrable.register('c_evaluator')
class CEvaluator(Evaluator):
    def __init__(self, transition_system: Optional[TransitionSystem] = None, args=None):
        super().__init__(transition_system, args)
        self.default_metric = "bleu"
        self.tree_bpe = None
        if args.tree_bpe_model is not None:
            self.tree_bpe = TreeBPE.load(args.tree_bpe_model)

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
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
                    if self.tree_bpe is not None:
                        ast = self.transition_system.compress_ast(hyp.tree)
                        ast = self.tree_bpe.decode(ast)
                        hyp_tree = self.transition_system.decompress_ast(ast)
                    else:
                        hyp_tree = hyp.tree
                    hyp_code = self.transition_system.ast_to_surface_code(hyp.tree)
                    hyp_code_tokens = self.transition_system.tokenize_code(hyp_code)
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
