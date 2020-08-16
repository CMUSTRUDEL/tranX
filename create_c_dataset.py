import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

import flutes
from argtyped import Arguments, Switch
from tqdm import tqdm

from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CGenTokenAction
from components.vocab import VocabEntry, Vocab
from datasets.c.build_dataset import RawExample, Repository, process_c_dataset
from datasets.c.constants import TOKEN_DELIMITER


SPLITS = {
    "train_extra": "train_extra.txt",
    "dev": "valid.txt",
    "test": "test.txt",
}

class Args(Arguments):
    # Comma-separated paths to `match_functions.py` outputs; later directories will be searched only if the repo is
    # not found in previous ones.
    data_dirs: str = "../github/match_output_test/,../github/match_output_varnames"
    ghcc_path: str = "../github/"  # path to `ghcc` library
    db_config_path: str = "../github/database-config.json"  # path to database config file required by `ghcc.DB`
    output_dir: str = "tranx_data"  # path to output folder where generated data will be stored
    chunk_size: int = 10000  # number of examples per output file
    spm_model_path: Optional[str] = "../code-translation/vocab_varnames/vocab.model"  # path to SentencePiece model
    n_procs: int = 0  # number of worker processes to spawn
    reference_data_dir: str = "../github/data/processed_varnames"
    # > Whether to also include AST for source (decompiled code). `src_ast` field will be set to `None` if not parsable.
    include_src_ast: Switch = True

    # Verification settings
    sanity_check: Switch = False
    skip_to_index: Optional[int]
    skip_to_repo: Optional[str]
    max_repos: Optional[int]


def main():
    flutes.register_ipython_excepthook()
    args = Args()
    print(args.to_string())
    if args.n_procs == 0:
        flutes.log("Setting `n_procs` to 0 may result in deadlock", "warning")

    import sys
    sys.path.append(args.ghcc_path)
    import ghcc

    data_dirs = [Path(d.strip()) for d in args.data_dirs.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_data_dir = Path(args.reference_data_dir)
    split_hashes = {}
    split_tgt_text = {}
    for split, text_file in SPLITS.items():
        (output_dir / split).mkdir(exist_ok=True)
        with (ref_data_dir / text_file).open() as f:
            tgt_set = {}
            hash_set = set()
            for line in f:
                if not line: continue
                src, *_tgt, var_map, score, repo, sha = line.strip().split("\1")
                hash_set.add(sha)
                tgt = "\1".join(_tgt) if len(_tgt) != 1 else _tgt[0]
                tgt_set[tgt.replace("\0", TOKEN_DELIMITER)] = len(tgt_set)
        split_hashes[split] = hash_set
        split_tgt_text[split] = tgt_set
        flutes.log(f"Read {len(tgt_set)} examples from {split} set")
    split_examples = {key: [None] * len(split_tgt_text[key]) for key in SPLITS.keys()}

    repos = []
    db = ghcc.MatchFuncDB(config_file=args.db_config_path)
    for entry in tqdm(db.safe_iter(batch_size=10000, static=True), desc="Loading data"):
        repo = entry['repo_owner'] + "/" + entry['repo_name']
        paths = [data_dir / repo / "matched_funcs.jsonl" for data_dir in data_dirs]
        repos.append(Repository(repo, paths))

    if args.skip_to_index is not None:
        repos = repos[args.skip_to_index:]
    elif args.skip_to_repo is not None:
        repos = list(flutes.drop_until(lambda x: x.repo == args.skip_to_repo, repos))
    if args.max_repos is not None:
        repos = repos[:args.max_repos]

    flutes.log(f"Data generation begins. {len(repos)} repositories in total.")
    generator = process_c_dataset(
        repos, args.spm_model_path, include_src_ast=args.include_src_ast,
        n_procs=args.n_procs, verbose=True, sanity_check=args.sanity_check)
    n_examples = 0
    tgt_sent_set = set()

    # def get_func_name(s: str) -> str:
    #     tokens = s.split(TOKEN_DELIMITER)
    #     func_name = next(tokens[idx] for idx in range(len(tokens) - 1) if tokens[idx + 1] == '(')
    #     return func_name

    def filter_fn(ex: RawExample) -> bool:
        tgt_code = ex.meta['raw_tgt_code']
        if tgt_code in tgt_sent_set:
            return False
        tgt_sent_set.add(tgt_code)
        for key, hashes in split_hashes.items():
            # if ex.meta['hash'] in hashes:
            if tgt_code in split_tgt_text[key]:
                split_examples[key][split_tgt_text[key][tgt_code]] = ex
                return False
        return True

    # revert to normal tuples to not depend on a specific NamedTuple class
    generator = map(tuple, filter(filter_fn, generator))
    for idx, examples in enumerate(flutes.chunk(args.chunk_size, generator)):
        path = output_dir / f"data_{idx:03d}.pkl"
        with path.open("wb") as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_examples += len(examples)
        split_desc = ", ".join(f"{k}: {sum(x is not None for x in v)}" for k, v in split_examples.items())
        flutes.log(f"Written file {path}, size = {flutes.readable_size(flutes.get_folder_size(path))}, "
                   f"{n_examples} examples generated ({split_desc}).")

    assert all(x is not None for examples in split_examples.values() for x in examples)
    shutil.copy(args.spm_model_path, output_dir / "vocab.model")
    with Path(args.spm_model_path).with_suffix(".vocab").open() as f:
        vocab_lines = [line.split("\t")[0] for line in f if line]
    primitive_vocab_entry = VocabEntry()
    primitive_vocab_entry.add(CGenTokenAction.STOP_SIGNAL)
    for word in vocab_lines:
        primitive_vocab_entry.add(word)
    code_vocab_entry = VocabEntry()
    for word in c_utils.RESERVED_WORDS:
        code_vocab_entry.add(c_utils.SPM_SPACE + word)
    code_vocab_entry.merge(primitive_vocab_entry)
    vocab = Vocab(source=code_vocab_entry, primitive=primitive_vocab_entry, code=code_vocab_entry)
    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)

    for split in SPLITS:
        split_dir = output_dir / split
        with (split_dir / "data.pkl").open("wb") as f:
            pickle.dump(split_examples[split], f, protocol=pickle.HIGHEST_PROTOCOL)
        flutes.log(f"Written {split} set, containing {len(split_examples[split])} examples "
                   f"({len(split_tgt_text[split])} expected)")
        os.symlink("../vocab.model", split_dir / "vocab.model")


if __name__ == '__main__':
    main()
