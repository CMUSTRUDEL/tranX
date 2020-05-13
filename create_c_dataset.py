import pickle
import shutil
from pathlib import Path
from typing import Optional

import flutes
from argtyped import Arguments, Switch

from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CGenTokenAction
from components.vocab import VocabEntry, Vocab
from datasets.c.build_dataset import RawExample, Repository, process_c_dataset


class Args(Arguments):
    data_dir: str = "../github/match_output"  # path to `match_functions.py` outputs
    ghcc_path: str = "../github/"  # path to `ghcc` library
    db_config_path: str = "../github/database-config.json"  # path to database config file required by `ghcc.DB`
    output_dir: str = "tranx_data"  # path to output folder where generated data will be stored
    chunk_size: int = 10000  # number of examples per output file
    spm_model_path: Optional[str] = "../code-translation/vocab/new_vocab.model"  # path to SentencePiece model
    n_procs: int = 0  # number of worker processes to spawn

    # Verification settings
    sanity_check: Switch = False
    skip_to_index: Optional[int]
    skip_to_repo: Optional[str]
    max_repos: Optional[int]


def main():
    flutes.register_ipython_excepthook()
    args = Args()
    if args.n_procs == 0:
        flutes.log("Setting `n_procs` to 0 may result in deadlock", "warning")

    import sys
    sys.path.append(args.ghcc_path)
    import ghcc

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repos = []
    db = ghcc.MatchFuncDB(config_file=args.db_config_path)
    for entry in db.safe_iter(static=True):
        repo = entry['repo_owner'] + "/" + entry['repo_name']
        path = data_dir / repo / "matched_funcs.jsonl"
        repos.append(Repository(repo, path))

    if args.skip_to_index is not None:
        repos = repos[args.skip_to_index:]
    elif args.skip_to_repo is not None:
        repos = list(flutes.drop_until(lambda x: x.repo == args.skip_to_repo, repos))
    if args.max_repos is not None:
        repos = repos[:args.max_repos]

    flutes.log(f"Data generation begins. {len(repos)} repositories in total.")
    generator = process_c_dataset(
        repos, args.spm_model_path,
        n_procs=args.n_procs, verbose=True, sanity_check=args.sanity_check)
    n_examples = 0
    tgt_sent_set = set()

    def filter_fn(ex: RawExample) -> bool:
        if ex.tgt in tgt_sent_set:
            return False
        tgt_sent_set.add(ex.tgt)
        return True

    generator = filter(filter_fn, generator)
    for idx, examples in enumerate(flutes.chunk(args.chunk_size, generator)):
        path = output_dir / f"data_{idx:03d}.pkl"
        with path.open("wb") as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_examples += len(examples)
        flutes.log(f"Written file {path}, size = {flutes.readable_size(flutes.get_folder_size(path))}, "
                   f"{n_examples} examples generated.")

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


if __name__ == '__main__':
    main()
