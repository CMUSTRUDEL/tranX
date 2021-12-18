# coding=utf-8
from __future__ import print_function

import sys
import time
import traceback

import torch
from termcolor import colored
from tqdm import tqdm

from asdl.lang.c.c_utils import SPM_SPACE
from asdl.tree_bpe import TreeBPE
from common.utils import Args

__all__ = [
    "decode",
    "evaluate",
]


@torch.no_grad()
def decode(examples, model, args: Args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset
    
    num_examples = len(examples)
    if verbose:
        print('evaluating %d examples' % num_examples)

    was_training = model.training
    model.eval()

    is_wikisql = args.parser == 'wikisql_parser'
    tree_bpe = None
    if args.tree_bpe_model is not None:
        tree_bpe = TreeBPE.load(args.tree_bpe_model)

    decode_results = []
    all_results = []
    count = 0
    for example_idx, example in enumerate(tqdm(iter(examples), desc='Decoding', file=sys.stdout, total=num_examples)):
        start = time.time()
        if is_wikisql:
            hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
        else:
            hyps = model.parse(example.src_sent, context=example, beam_size=args.beam_size,
                               allow_incomplete=args.allow_incomplete_hypotheses)
        time_elapsed = time.time() - start
        decoded_hyps = []
        all_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                if tree_bpe is not None:
                    hyp.tree = model.transition_system.decompress_ast(
                        tree_bpe.decode(model.transition_system.compress_ast(hyp.tree)))
            except Exception:
                pass
            all_hyps.append(hyp)
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                decoded_hyps.append(hyp)
            except Exception:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print(f'Example: {example.idx}\nIntent: {" ".join(example.src_sent)}\n'
                          f'Target Code:\n{example.tgt_code}', file=sys.stdout)
                    try:
                        print(f'Hypothesis[{hyp_id:d}]:\n{hyp.tree.to_string()}', file=sys.stdout)
                    except Exception:
                        print("Failed to print hypothesis.")
                        traceback.print_exc(file=sys.stdout)
                    if got_code:
                        print()
                        print(hyp.code)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1
        if verbose:
            print(colored(f"Time elapsed: {time_elapsed:.2f}, {len(decoded_hyps)} hypotheses", "red"))
            print(colored("Src:", "green"), "".join(example.src_sent).replace(SPM_SPACE, " ").strip())
            print(colored("Tgt:", "green"), " ".join(example.tgt_code))
            for idx, hyp in enumerate(decoded_hyps[:5]):
                print(colored(f"Hyp {idx}:", "green"), hyp.code)
            print(colored("=" * 60, "yellow"), flush=True)

        decode_results.append(decoded_hyps)
        all_results.append(all_hyps)

    if was_training: model.train()

    return decode_results, all_results


def evaluate(examples, parser, evaluator, args: Args, verbose=False, return_decode_result=False,
             eval_top_pred_only=False):
    decode_results, all_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, all_results=all_results, fast_mode=eval_top_pred_only, args=args)
    
    result_tuples = [] # tuple of (source code tokens, target code tokens, top hypothesis code tokens, score of top hypothesis)
    for example, hyp_list in zip(examples, decode_results):
        if hyp_list:
            hyp_code_tokens = evaluator.transition_system.tokenize_code(hyp_list[0].code)
            result_tuples.append((example.src_sent, example.tgt_code, hyp_code_tokens, hyp_list[0].score.item()))
        else:
            result_tuples.append(None)

    if return_decode_result:
        return eval_result, result_tuples
    else:
        return eval_result
