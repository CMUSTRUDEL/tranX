from pathlib import Path
from typing import Optional

import flutes
from argtyped import Arguments

from datasets.c.dataset import Repository, process_c_dataset


class Args(Arguments):
    data_dir: str = "../github/match_output"
    ghcc_path: str = "../github/"
    db_config_path: str = "../github/database-config.json"
    token_output_path: Optional[str]
    output_dir: str = "tranx_data"
    vocab_freq_cutoff: int = 15
    spm_model_path: Optional[str] = "vocab.model"
    skip_to: Optional[int]
    n_procs: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not ((self.spm_model_path is None) ^ (self.token_output_path is None)):
            raise ValueError("Exactly one of 'spm_model_path' and 'token_output_path' should be None")


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

    if args.skip_to is not None:
        repos = repos[args.skip_to:]
    repos = repos[:100]
    dataset = process_c_dataset(
        repos, args.spm_model_path, args.token_output_path,
        vocab_freq_cutoff=args.vocab_freq_cutoff, n_procs=args.n_procs, verbose=True)


if __name__ == '__main__':
    main()
