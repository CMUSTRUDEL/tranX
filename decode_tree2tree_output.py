import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import flutes
from argtyped import Arguments, Switch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm

from asdl.asdl_ast import AbstractSyntaxTree, CompressedAST
from asdl.lang.c.c_utils import asdl_ast_to_c_ast
from create_tree2tree_data import LegacyT2TDataProcessor, ProcessedAST, T2TDataProcessor

Sentence = List[str]


class Args(Arguments):
    asdl_path: str = "asdl/lang/c/c_asdl.txt"
    spm_model_path: str = "tranx_data/vocab.model"
    vocab_path: Optional[str]
    test_data_file: Optional[str]
    tree_bpe_path: Optional[str] = None
    eval_file: str
    pdb: Switch = False
    allow_subset: Switch = False
    legacy: Optional[bool] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.legacy is None:
            self.legacy = bool(self.tree_bpe_path is None)
        else:
            if self.legacy:
                assert self.tree_bpe_path is None
            else:
                assert self.tree_bpe_path is not None
        if self.tree_bpe_path is None:
            data_dir = "../Tree2Tree/tree2tree_data/"
        else:
            data_dir = "../Tree2Tree/tree2tree_data_bpe/"
        if self.vocab_path is None:
            self.vocab_path = Path(data_dir + "vocab.pkl")
        if self.test_data_file is None:
            self.test_data_file = Path(data_dir + "test/data.pkl")


def main():
    sys.setrecursionlimit(32768)
    args = Args()
    if args.pdb:
        flutes.register_ipython_excepthook()

    processor_klass = LegacyT2TDataProcessor if args.legacy else T2TDataProcessor
    processor = processor_klass(args.asdl_path, args.spm_model_path, tree_bpe_path=args.tree_bpe_path, strict=False)
    trans = processor.trans
    lexer = trans.lexer

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    assert vocab['source'] == [processor.vocab.id2word[i] for i in range(len(processor.vocab))]

    def decompress_unstrict(ast: CompressedAST) -> AbstractSyntaxTree:
        if ast is None or ast == trans.UNFILLED or isinstance(ast, str):
            return ast
        prod_id, fields = ast
        node = AbstractSyntaxTree(trans.grammar.id2prod[prod_id])
        for field, value in zip(node.fields, fields):
            if value is not None:
                value = value if isinstance(value, list) else [value]
                for val in value:
                    if isinstance(val, tuple):
                        field.add_value(decompress_unstrict(val))
                    else:
                        field.add_value(val)
        return node

    def to_code_robust(ast: ProcessedAST) -> str:
        return trans.generator.generate_code(
            asdl_ast_to_c_ast(
                decompress_unstrict(
                    processor.detransform_ast(ast)
                ), trans.grammar, ignore_error=True
            ), fill_field_name=False)

    with open(args.test_data_file, "rb") as f:
        test_data: List[Tuple[ProcessedAST, ProcessedAST]] = pickle.load(f)
    with open(args.eval_file, "rb") as f:
        outputs: List[ProcessedAST] = pickle.load(f)
    if not args.allow_subset:
        assert len(test_data) == len(outputs)

    references: List[List[Sentence]] = []
    hypothesis: List[Sentence] = []
    for (_, target_ast), output_ast in zip(tqdm(test_data), outputs):
        processor.strict = True
        target = trans.ast_to_surface_code(trans.decompress_ast(processor.detransform_ast(target_ast)))
        processor.strict = False
        output = to_code_robust(output_ast)
        references.append([lexer.lex(target)])
        hypothesis.append(lexer.lex(output))
    corpus_bleu_score = corpus_bleu(references, hypothesis)
    sentence_bleu_score = sum(sentence_bleu(ref, hyp) for ref, hyp in zip(references, hypothesis)) / len(test_data)
    print(f"Corpus BLEU: {corpus_bleu_score}, Sentence BLEU: {sentence_bleu_score}")

    with open(args.eval_file + ".txt", "w") as f:
        for code in hypothesis:
            f.write(" ".join(code) + "\n")


if __name__ == '__main__':
    main()
