import os
import sys
import pickle
import shutil
from pathlib import Path
from typing import Optional
from collections import defaultdict
import multiprocessing as mp
import functools
from typing import Any, Callable, Dict, List, Tuple, NamedTuple, Optional, Set
import json
import random

import flutes
from argtyped import Arguments, Switch
from tqdm import tqdm
import sentencepiece as spm

from asdl.lang.c import c_utils
from asdl.lang.c.c_transition_system import CGenTokenAction
from components.vocab import VocabEntry, Vocab
from datasets.c.build_dataset import RawExample, Repository, process_c_dataset
from datasets.c.constants import TOKEN_DELIMITER


END_SIGNATURE = b"END_REPO"
TOKEN_SEP = "\0"
TUPLE_SEP = "\1"

SPLITS = {
    "train_extra": "train_extra.txt",
    "dev": "valid.txt",
    "test": "test.txt",
}

class Args(Arguments):
    # Comma-separated paths to `match_functions.py` outputs; later directories will be searched only if the repo is
    # not found in previous ones.
    data_dir: str = "../github/match_output_test/" #,../github/match_output_varnames"
    output_dir: str = "tranx_data"  # path to output folder where generated data will be stored
    chunk_size: int = 10000  # number of examples per output file
    reference_data_dir: str = "../github/data/processed_varnames"
    # > Whether to also include AST for source (decompiled code). `src_ast` field will be set to `None` if not parsable.
    include_src_ast: Switch = True
    vocab_size: int = 32000

    # Verification settings
    sanity_check: Switch = False
    skip_to_index: Optional[int]
    skip_to_repo: Optional[str]
    max_repos: Optional[int]

    # Internals
    queue_size: int = 1024
    n_procs: int = 16  # number of worker processes to spawn

    # Data Splits
    test_split_size: Optional[int] = None # 3000
    test_split_portion: Optional[float] = 0.1
    extra_train_portion: float = 2.0 / 3  # how much of dev/test sets should be split out into an extra fine-tuning set
    max_test_repos: int = 50

    # For deletion
    ghcc_path: str = "../github/"  # path to `ghcc` library
    db_config_path: str = "../github/database-config.json"  # path to database config file required by `ghcc.DB`
    spm_model_path: Optional[str] = "../code-translation/vocab_varnames/vocab.model"  # path to SentencePiece model


class ExampleInfo(NamedTuple):
    decompiled_code: str
    original_code: str
    var_names: Dict[str, Tuple[str, str]]
    # decompiled_ast: Dict[str, Any]
    # original_ast: Dict[str, Any]
    repo: str
    sha: str

def exception_handler(e: Exception, repo_info: Tuple['posix.DirEntry', str], queue: 'mp.Queue[QueueElem]'):
    repo = f"{repo_info.repo_owner}/{repo_info[1]}"
    flutes.log_exception(e, f"Exception occurred when processing {repo}", force_console=True)
    queue.put(END_SIGNATURE)

def convert_code(code: List[str]) -> str:
    code_str = TOKEN_SEP.join(code)
    return code_str

@flutes.exception_wrapper(exception_handler)
def process(repo_info: Tuple[str, str], queue: 'mp.Queue[QueueElem]') -> None:
    with open(os.path.join(repo_info[0], "matched_funcs.jsonl")) as f:
        for line in f:
            if not line:
                continue
            
            matched_func = json.loads(line)
            decompiled_code = convert_code(matched_func['decompiled_tokens'])
            original_code = convert_code(matched_func['original_tokens'])
            var_names = {k: (decomp, orig) for k, [decomp, orig] in matched_func['variable_names'].items()}
            sha = matched_func['binary_hash']
            
            example = ExampleInfo(decompiled_code=decompiled_code,
                        original_code=original_code,
                        var_names=var_names, 
                        repo=repo_info[1],
                        sha=sha)

            queue.put(example)
    queue.put(END_SIGNATURE)

def create_excluded_split(repo_names, data_by_repo, target_size: int, max_repos: int, extra_train_portion: float, min_repo_size: int = 0) \
        -> Tuple[List[str], List[int], List[int]]:
    # ([name], [index])
    filtered_repos = repo_names
    if min_repo_size > 0:
        filtered_repos = [repo for repo in filtered_repos if len(data_by_repo[repo]) >= min_repo_size]
    while True:
        repo_count = random.randint(1, min(len(filtered_repos), max_repos))
        chosen_repos = random.sample(filtered_repos, k=repo_count)
        sample_size = sum(len(data_by_repo[name]) for name in chosen_repos)
        if 0.8 * target_size <= sample_size <= 1.1 * target_size:
            # Keep sampling until we get something with appropriate size.
            break
    extra_train_indices = []
    split_indices = []
    for name in chosen_repos:
        indices = data_by_repo[name].copy()
        random.shuffle(indices)
        split_size = int(len(indices) * extra_train_portion)
        extra_train_indices += indices[:split_size]
        split_indices += indices[split_size:]
    return repo_names, data_by_repo, chosen_repos, split_indices, extra_train_indices

def main():
    flutes.register_ipython_excepthook()
    args = Args()
    print(args.to_string())
    if args.n_procs == 0:
        flutes.log("Setting `n_procs` to 0 may result in deadlock", "warning")
    
    output_dir = Path(args.output_dir)

    # sys.path.append(args.ghcc_path)
    # import ghcc

    # Generate a list of all repositories in the input dataset
    repos = []
    for owner in os.scandir(args.data_dir):
        for reponame in os.scandir(owner):
            repos.append((reponame.path, f"{owner.name}/{reponame.name}"))

    # Open data serialized in the json format and convert it into a list of ExampleInfo tuples.
    # Each repository has a single json file that has an entry for each function that occurs in that repository.
    original_code_set: Set[str] = set()
    n_duplicate = 0
    n_examples = 0
    with mp.Manager() as manager:
        example_queue: 'mp.Queue[QueueElem]' = manager.Queue(args.queue_size)
        with flutes.safe_pool(args.n_procs) as pool:
            process_fn = functools.partial(process, queue=example_queue)
            pool.map_async(process_fn, repos, error_callback=flutes.log_exception)

            end_signals = 0
            progress =  tqdm(total=len(repos))

            examples = []

            while end_signals < len(repos):
                example = example_queue.get()
                if example == END_SIGNATURE:
                    progress.update(1)
                    end_signals += 1
                    # continue
                else: # process an example
                    # Perform deduplication
                    if example.original_code not in original_code_set:
                        original_code_set.add(example.original_code)
                        examples.append(example)
                        n_examples += 1
                    else:
                        n_duplicate += 1
                    
                    if (n_examples + n_duplicate) % 100 == 0:
                        progress.set_postfix({"duplicate": n_duplicate, "examples": n_examples}, refresh=False)
                        progress.refresh()
    
    del original_code_set # This should be quite large and we don't need it anymore.

    ### Generate data splits ###
    # Splits are generated by repository so that similar functions from one repository
    # aren't included in different data sets, making the task artificially easy.
    
    test_size = args.test_split_size or int(len(examples) * args.test_split_portion)
    data_by_repo = defaultdict(list)
    for index, example in enumerate(examples):
        data_by_repo[example.repo].append(index)
    repo_names = list(data_by_repo.keys())

    repo_names, data_by_repo, dev_repos, dev_split, extra_train_dev_split = create_excluded_split(
        repo_names, data_by_repo, test_size, args.max_test_repos, args.extra_train_portion)

    for repo_name in dev_repos:
        data_by_repo[repo_name]
            
    repo_names, data_by_repo, test_repos, test_split, extra_train_test_split = create_excluded_split(
        repo_names, data_by_repo, test_size, args.max_test_repos, args.extra_train_portion)
    excluded_indices = set(dev_split + extra_train_dev_split + test_split + extra_train_test_split)

    # Training set: all the remaining stuff.
    train_split = [idx for idx in range(len(examples)) if idx not in excluded_indices]
    extra_train_split = extra_train_dev_split + extra_train_test_split
    splits = {
        "train": train_split,
        "valid": dev_split,
        "test": test_split,
        "train_extra": extra_train_split,
    }

    os.makedirs(output_dir / "indices", exist_ok=True)
    with (output_dir / "indices" / "split_indices.pkl").open("wb") as f:
        pickle.dump(splits, f)

    ### Build the vocabulary. Write out all identifiers and string literals
    ### to a file, then call sentencepiece
    if not (output_dir / "vocab.model").exists():
        # Write out training text and train SentencePiece model.
        train_text_path = output_dir / "train_text.txt"
        with train_text_path.open("w") as f:
            for idx in tqdm(train_split, desc="Writing training text"):
                src_tokens = examples[idx].decompiled_code.split(TOKEN_SEP)
                new_src_tokens = []
                for token in src_tokens:
                    if token in examples[idx].var_names:
                        var1, var2 = examples[idx].var_names[token]
                        new_src_tokens += [var1, var2]
                    else:
                        new_src_tokens.append(token)
                f.write(" ".join(new_src_tokens) + "\n")
                f.write(examples[idx].original_code.replace(TOKEN_SEP, " ") + "\n")
        spm_train_args = {
            "input": train_text_path,
            "model_prefix": output_dir / "vocab",
            "vocab_size": args.vocab_size,
        }
        spm.SentencePieceTrainer.Train(" ".join(f"--{name}={str(value)}" for name, value in spm_train_args.items()))

    
    ### Beginning of original create_c_dataset.py ###
    sys.exit(0) # remove when ready to refactor this portion.

    data_dirs = [Path(d.strip()) for d in args.data_dirs.split(",")]
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
    spm_model_vocab_path = Path(args.spm_model_path).with_suffix(".vocab")
    shutil.copy(spm_model_vocab_path, output_dir / "vocab.vocab")
    with spm_model_vocab_path.open() as f:
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
