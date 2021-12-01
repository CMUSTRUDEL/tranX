# coding=utf-8
from pathlib import Path
from typing import Optional, Tuple

from argtyped import Arguments, Switch
from typing_extensions import Literal

__all__ = [
    "cached_property",
    "Args",
    "update_args",
]


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Args(Arguments):
    # General configuration
    seed: int = 0  # random seed
    cuda: Switch = False  # use GPU
    # > [Deprecated] language to parse. Deprecated, use --transition_system and --parser instead
    lang: Literal['python', 'lambda_dcs', 'wikisql', 'prolog', 'python3'] = "python"
    asdl_file: Optional[str]  # path to ASDL grammar specification
    mode: Literal['train', 'test', 'interactive', 'train_paraphrase_identifier', 'train_reconstructor', 'rerank']

    # Modularized configuration
    dataset: str = "default_dataset"  # name of dataset class to load
    parser: str = "default_parser"  # name of parser class to load
    transition_system: str = "python2"  # name of transition system to use
    evaluator: str = "default_evaluator"  # name of evaluator class to use

    # Model configuration
    encoder: Literal['lstm', 'transformer'] = "transformer"  # encoder architecture
    encoder_layers: int = 1  # number of layers for encoder
    decoder: Literal['lstm', 'transformer'] = "transformer" # decoder architecture
    decoder_layers: int = 1  # number of layers for decoder

    # Transformer-specific
    num_heads: int = 8  # number of attentional heads for Transformer encoder
    poswise_ff_dim: int = 1024  # size of the position-wise feed-forward network
    transformer_embedding_dropout: float = 0.1
    transformer_residual_dropout: float = 0.1

    # Embedding sizes
    embed_size: int = 128  # size of word embeddings
    action_embed_size: int = 128  # size of ApplyRule/GenToken action embeddings
    field_embed_size: int = 64  # embedding size of ASDL fields
    type_embed_size: int = 64  # embedding size of ASDL types

    # Hidden sizes
    hidden_size: int = 256  # size of LSTM hidden states
    ptrnet_hidden_dim: int = 32  # hidden dimension used in pointer network
    att_vec_size: int = 256  # size of attentional vector

    # Readout layer
    # > use additional linear layer to transform the attentional vector for computing action probabilities
    query_vec_to_action_map: Switch = True
    readout: Literal['identity', 'non_linear'] = "identity"  # type of activation if using addition linear layer
    query_vec_to_action_diff_map: Switch = False  # use different linear mapping

    # Supervised attention
    sup_attention: Switch = False  # use supervised attention

    # Parent information switch for decoder LSTM
    parent_production_embed: Switch = True  # use embedding of parent ASDL production to update LSTM state
    parent_field_embed: Switch = True  # use embedding of parent field to update LSTM state
    parent_field_type_embed: Switch = True  # use embedding of the ASDL type of parent field to update LSTM state
    parent_state: Switch = True  # use the parent hidden state to update LSTM state

    input_feed: Switch = True  # use input feeding
    copy: Switch = True  # use copy mechanism

    # WikiSQL-specific model configuration parameters
    column_att: Literal['dot_prod', 'affine'] = "affine"  # how to perform attention over table columns
    answer_prune: Switch = True  # use answer pruning

    # Training
    vocab: str  # path to the serialized vocabulary
    glove_embed_path: Optional[str]  # path to pre-trained GloVe embeddings

    train_file: Optional[str]  # path to training dataset file
    dev_file: Optional[str]  # path to dev dataset file
    pretrain: Optional[str]  # path to pre-trained model file

    batch_size: int = 10
    dropout: float = 0.  # dropout rate
    word_dropout: float = 0.  # word dropout rate
    decoder_word_dropout: float = 0.3  # word dropout rate on decoder
    primitive_token_label_smoothing: float = 0.0  # apply label smoothing when predicting primitive tokens
    src_token_label_smoothing: float = 0.0  # apply label smoothing in reconstruction model when predicting source tokens

    negative_sample_type: Literal['best', 'sample', 'all'] = "best"

    # Training schedule details
    valid_metric: Literal['acc'] = "acc"  # metric used for validation
    valid_every_epoch: int = 1  # perform validation every x epochs
    valid_every_iters: int = -1  # perform validation every x iterations
    log_every: int = 10  # log training statistics every x iterations

    output_dir: Path
    save_all_models: Switch = False  # save all intermediate checkpoints
    patience: int = 5  # training patience
    max_num_trial: int = 10  # stop training after x number of trials
    uniform_init: Optional[float]  # if specified, use uniform initialization for all parameters
    glorot_init: Switch = False  # use Glorot initialization
    clip_grad: float = 5.  # clip gradients
    max_epoch: int = -1  # maximum number of training epochs
    optimizer: str = "Adam"
    lr: float = 0.001  # learning rate
    betas: Tuple[float, float] = (0.9, 0.98) # for Adam optimizer
    eps: float = 1e-9 # for Adam optimizer
    lr_warmup_iters: int = 4000
    lr_decay: float = 0.5  # decay learning rate if the validation performance drops
    lr_decay_after_epoch: int = 0  # decay learning rate after x epochs
    decay_lr_every_epoch: Switch = False  # force to decay learning rate after each epoch
    reset_optimizer: Switch = False  # whether to reset optimizer when loading the best checkpoint
    verbose: Switch = False
    eval_top_pred_only: Switch = False  # only evaluate the top prediction in validation

    # Decoding / Validation / Testing
    load_model: Optional[str]  # load a pre-trained model
    beam_size: int = 5  # beam size for beam search
    decode_max_time_step: int = 100  # maximum number of time steps used in decoding and sampling

    sample_size: int = 5  # sample_size
    test_file: Optional[str]  # path to the test file

    # Reranking
    features: Optional[str]  # NOT YET SUPPORTED
    load_reconstruction_model: Optional[str]
    load_paraphrase_model: Optional[str]
    load_reranker: Optional[str]
    tie_embed: Switch = False  # tie source and target embeddings when training paraphrasing model
    train_decode_file: Optional[str]  # decoding results on training set
    test_decode_file: Optional[str]  # decoding results on test set
    dev_decode_file: Optional[str]  # decoding results on dev set
    metric: Literal['bleu', 'accuracy'] = "accuracy"
    num_workers: int = 1  # number of multiprocessing workers

    # Self-training
    load_decode_results: Optional[str]
    unsup_loss_weight: float = 1.  # loss of unsupervised learning weight
    unlabeled_file: Optional[str]  # path to the training source file used in semi-supervised self-training

    # Interactive mode
    example_preprocessor: Optional[str]  # name of the class that is used to pre-process raw input examples

    # Dataset-specific config
    sql_db_file: Optional[str]  # path to WikiSQL database file for evaluation (SQLite)

    # Stuff for C
    profile: Switch = False  # profiling mode
    variable_name: Literal['decompiled', 'original'] = "decompiled"
    tree_bpe_model: Optional[str]  # path to TreeBPE model
    max_src_len: int = 512
    max_actions: int = 512
    max_tokens_per_batch: Optional[int]
    # > return non-terminated hypotheses if the number of terminated hypotheses is less than beam size
    allow_incomplete_hypotheses: Switch = False
    # > representation of source (decompiled code):
    #   code in text form ("text") or AST as a sequence of actions ("action_seq")
    src_repr_mode: Literal['text', 'action_seq'] = "text"
    # > whether to encode the source AST with TreeBPE (only applies if `src_repr_mode` is "action_seq")
    src_action_seq_tree_bpe: Switch = True


def update_args(args: Args):
    for key in Args.__annotations__.keys():
        if not hasattr(args, key) and hasattr(Args, key):
            setattr(args, key, getattr(Args, key))
