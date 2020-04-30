import sys
from pathlib import Path
from typing import Optional

import flutes
from argtyped import Arguments

from datasets.c.dataset import Repository, process_c_dataset

sys.path.append("../github")
import ghcc


class Args(Arguments):
    data_dir: str = "../github/match_output"
    output_dir: str = "tranx_data"
    vocab_freq_cutoff: int = 15
    spm_model_path: str = "vocab.model"
    skip_to: Optional[int]
    n_procs: int = 0


def main():
    flutes.register_ipython_excepthook()
    args = Args()
    if args.n_procs == 0:
        flutes.log("Setting `n_procs` to 0 may result in deadlock", "warning")

    data_dir = Path(args.data_dir)
    repos = []
    db = ghcc.MatchFuncDB(config_file="../github/database-config.json")
    for entry in db.safe_iter(static=True):
        repo = entry['repo_owner'] + "/" + entry['repo_name']
        path = data_dir / repo / "matched_funcs.jsonl"
        repos.append(Repository(repo, path))

    if args.skip_to is not None:
        repos = repos[args.skip_to:]
    repos = repos[:100]
    dataset = process_c_dataset(
        repos, args.spm_model_path, vocab_freq_cutoff=args.vocab_freq_cutoff, n_procs=args.n_procs, verbose=True)
    breakpoint()


if __name__ == '__main__':
    main()
