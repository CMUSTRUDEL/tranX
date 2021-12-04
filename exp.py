import pickle
import random
import time
from typing import Type
import functools

import astor
import flutes
import torch.autograd.profiler
from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.tree_bpe import TreeBPE
from common.utils import Args
from components.dataset import Dataset
from components.evaluator import Evaluator
from components.reranker import *
from components.standalone_parser import StandaloneParser
from model import nn_utils
from model.paraphrase import ParaphraseIdentificationModel
from model.reconstruction_model import Reconstructor
from model.utils import GloveHelper

assert astor.__version__ == "0.7.1"


def init_config() -> Args:
    args = Args()

    # seed the RNG
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


class Validator:
    def __init__(self, args: Args, evaluator: Evaluator, model, optimizer, dev_set: Dataset):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.dev_set = dev_set
        self.history_dev_scores = []
        self.num_trial = 0
        self.patience = 0

        self.collate_fn = None
        if hasattr(model, 'create_collate_fn'):
            self.collate_fn = model.create_collate_fn()

    def validate(self, epoch: int, iteration: int) -> None:
        args = self.args

        if args.save_all_models:
            model_file = args.output_dir / f"model.iter{iteration:d}.bin"
            flutes.log('save model to [%s]' % model_file)
            self.model.save(model_file)

        # perform validation
        if len(self.dev_set) > 0:
            flutes.log('Epoch %d: begin validation' % epoch)
            eval_start = time.time()
            eval_results, decode_results = evaluation.evaluate(
                self.dev_set, self.model, self.evaluator, args,
                verbose=False, eval_top_pred_only=args.eval_top_pred_only, return_decode_result=True)
            with (self.args.output_dir / f"decode.dev.iter{iteration:d}.txt").open("w") as f:
                for result in decode_results:
                    if len(result) > 0:
                        f.write(result[0].code.replace("\n", " "))
                    else:
                        f.write("<decode failed>")
                    f.write("\n")
            dev_score = eval_results[self.evaluator.default_metric]

            dev_loss, dev_examples = 0., 0
            self.dev_set.mode = "train"
            with torch.no_grad():
                for batch_examples in self.dev_set.batch_iter(
                        args.batch_size, shuffle=False, collate_fn=self.collate_fn,
                        num_workers=min(args.num_workers, 1),
                        decode_max_time_step=args.decode_max_time_step):
                    ret_val = self.model.score(batch_examples)
                    loss = -ret_val[0]
                    dev_loss += torch.sum(loss).item()
                    dev_examples += len(batch_examples)
                    if dev_examples >= 300:
                        break
            self.dev_set.mode = "eval"

            flutes.log(
                f"[Epoch {epoch:d}] "
                f"evaluate details: {eval_results}, "
                f"dev {self.evaluator.default_metric}: {dev_score:.5f}, "
                f"dev loss: {dev_loss / dev_examples:.5f} "
                f"(took {time.time() - eval_start:.2f}s)"
            )

            if args.wandb_project is not None:
                wandb.log({
                    "dev_loss": dev_loss / dev_examples,
                    self.evaluator.default_metric: eval_results
                })

            is_better = self.history_dev_scores == [] or dev_score > max(self.history_dev_scores)
            self.history_dev_scores.append(dev_score)
        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = self.optimizer.param_groups[0]['lr'] * args.lr_decay
            flutes.log('decay learning rate to %f' % lr)

            # set new lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            self.patience = 0
            model_file = args.output_dir / "model.bin"
            flutes.log('save the current model ..')
            flutes.log('save model to [%s]' % model_file)
            self.model.save(model_file)
            # also save the optimizers' state
            torch.save(self.optimizer.state_dict(), args.output_dir / "model.optim.bin")
        elif self.patience < args.patience and epoch >= args.lr_decay_after_epoch:
            self.patience += 1
            flutes.log('hit patience %d' % self.patience)

        if epoch == args.max_epoch:
            flutes.log('reached max epoch, stop!')
            exit(0)

        if self.patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            self.num_trial += 1
            flutes.log('hit #%d trial' % self.num_trial)
            if self.num_trial == args.max_num_trial:
                flutes.log('early stop!')
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = self.optimizer.param_groups[0]['lr'] * args.lr_decay
            flutes.log('load previously best model and decay learning rate to %f' % lr)

            # load model
            params = torch.load(args.output_dir / "model.bin", map_location=lambda storage, loc: storage)
            self.model.load_state_dict(params['state_dict'])
            if args.cuda: self.model.cuda()

            # load optimizers
            if args.reset_optimizer:
                flutes.log('reset optimizer')
                self.optimizer.state.clear()
            else:
                flutes.log('restore parameters of the optimizers')
                self.optimizer.load_state_dict(torch.load(args.output_dir / "model.optim.bin"))

            # set new lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            self.patience = 0


def train(args: Args):
    """Maximum Likelihood Estimation"""

    dataset_cls: Type[Dataset] = Registrable.by_name(args.dataset)

    # load in train/dev set
    train_set = dataset_cls.from_bin_file(args.train_file, args, mode="train")

    if args.dev_file:
        dev_set = dataset_cls.from_bin_file(args.dev_file, args, mode="eval")
    else:
        dev_set = dataset_cls(examples=[])

    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    if args.tree_bpe_model is not None:
        tree_bpe = TreeBPE.load(args.tree_bpe_model)
        grammar = tree_bpe.patch_grammar(grammar)
    transition_system = Registrable.by_name(args.transition_system)(grammar)

    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    if args.pretrain:
        flutes.log('Finetune with: ' + args.pretrain)
        model = parser_cls.load(model_path=args.pretrain, cuda=args.cuda)
    else:
        model = parser_cls(args, vocab, transition_system)

    n_params = sum(param.numel() for param in model.parameters())
    print("#Parameters:", n_params)

    if args.wandb_project is not None:
        import wandb
        wandb.init(name=str(args.output_dir), project=args.wandb_project, config=args)

    model.train()
    if args.cuda: model.cuda()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)

    optimizer_cls = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps)
    
    if args.lr_warmup_iters >= 0:
        def schedule_lr_multiplier(iteration: int, warmup: int):
            multiplier = (min(1.0, iteration / warmup) *
                    (1 / math.sqrt(max(iteration, warmup))))
            return multiplier

        scheduler_lambda = functools.partial(
            schedule_lr_multiplier, warmup=args.lr_warmup_iters)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda)

    if not args.pretrain:
        if args.uniform_init:
            flutes.log('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init))
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
        elif args.glorot_init:
            flutes.log('use glorot initialization')
            nn_utils.glorot_init(model.parameters())

        # load pre-trained word embedding (optional)
        if args.glove_embed_path:
            flutes.log('load glove embedding from: %s' % args.glove_embed_path)
            glove_embedding = GloveHelper(args.glove_embed_path)
            glove_embedding.load_to(model.src_embed, vocab.source)

    flutes.log('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)))
    flutes.log('vocab: %s' % repr(vocab))

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    validator = Validator(args, evaluator, model, optimizer, dev_set)
    collate_fn = None
    if hasattr(model, 'create_collate_fn'):
        collate_fn = model.create_collate_fn()

    if args.profile:
        prof = torch.autograd.profiler.profile(use_cuda=args.cuda).__enter__()
        flutes.log("Profiling starts")
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(
                args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers,
                decode_max_time_step=args.decode_max_time_step):
            train_iter += 1
            optimizer.zero_grad()

            # breakpoint()
            ret_val = model.score(batch_examples)
            loss = -ret_val[0]

            # print(loss.data)
            loss_val = torch.sum(loss).item()
            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                breakpoint()
                model.score(batch_examples)
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data[0]
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if args.lr_warmup_iters >= 0:
                scheduler.step()

            if train_iter % args.log_every == 0:
                log_str = 'Iter %d: encoder loss=%.5f' % (train_iter, report_loss / report_examples)

                if args.wandb_project:
                    wandb.log({"training_loss": report_loss / report_examples})
            
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                flutes.log(log_str)
                report_loss = report_examples = 0.

            if args.profile and train_iter >= 20:
                break

            if args.valid_every_iters > 0 and train_iter % args.valid_every_iters == 0:
                validator.validate(epoch, train_iter)

        if args.profile and train_iter >= 20:
            break

        flutes.log('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin))

        if args.valid_every_epoch > 0 and epoch % args.valid_every_epoch == 0:
            validator.validate(epoch, train_iter)

    if args.profile:
        prof.__exit__(None, None, None)
        flutes.log(prof.key_averages().table(sort_by="cuda_time_total"))


def train_rerank_feature(args: Args):
    train_set = Dataset.from_bin_file(args.train_file, args, mode="train")
    dev_set = Dataset.from_bin_file(args.dev_file, args, mode="eval")
    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    if args.tree_bpe_model is not None:
        tree_bpe = TreeBPE.load(args.tree_bpe_model)
        grammar = tree_bpe.patch_grammar(grammar)
    transition_system = Registrable.by_name(args.transition_system)(grammar)

    train_paraphrase_model = args.mode == 'train_paraphrase_identifier'

    def _get_feat_class():
        if args.mode == 'train_reconstructor':
            return Reconstructor
        elif args.mode == 'train_paraphrase_identifier':
            return ParaphraseIdentificationModel

    def _filter_hyps(_decode_results):
        for i in range(len(_decode_results)):
            valid_hyps = []
            for hyp in _decode_results[i]:
                try:
                    transition_system.tokenize_code(hyp.code)
                    valid_hyps.append(hyp)
                except:
                    pass

            _decode_results[i] = valid_hyps

    model = _get_feat_class()(args, vocab, transition_system)

    if args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # if training the paraphrase model, also load in decoding results
    if train_paraphrase_model:
        print('load training decode results [%s]' % args.train_decode_file, file=sys.stderr)
        train_decode_results = pickle.load(open(args.train_decode_file, 'rb'))
        _filter_hyps(train_decode_results)
        train_decode_results = {e.idx: hyps for e, hyps in zip(train_set, train_decode_results)}

        print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
        dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
        _filter_hyps(dev_decode_results)
        dev_decode_results = {e.idx: hyps for e, hyps in zip(dev_set, dev_decode_results)}

    def evaluate_ppl():
        model.eval()
        cum_loss = 0.
        cum_tgt_words = 0.
        for batch in dev_set.batch_iter(args.batch_size):
            loss = -model.score(batch).sum()
            cum_loss += loss.data.item()
            cum_tgt_words += sum(len(e.src_sent) + 1 for e in batch)  # add ending </s>

        ppl = np.exp(cum_loss / cum_tgt_words)
        model.train()
        return ppl

    def evaluate_paraphrase_acc():
        model.eval()
        labels = []
        for batch in dev_set.batch_iter(args.batch_size):
            probs = model.score(batch).exp().data.cpu().numpy()
            for p in probs:
                labels.append(p >= 0.5)

            # get negative examples
            batch_decoding_results = [dev_decode_results[e.idx] for e in batch]
            batch_negative_examples = [get_negative_example(e, _hyps, type='best')
                                       for e, _hyps in zip(batch, batch_decoding_results)]
            batch_negative_examples = list(filter(None, batch_negative_examples))
            probs = model.score(batch_negative_examples).exp().data.cpu().numpy()
            for p in probs:
                labels.append(p < 0.5)

        acc = np.average(labels)
        model.train()
        return acc

    def get_negative_example(_example, _hyps, type='sample'):
        incorrect_hyps = [hyp for hyp in _hyps if not hyp.is_correct]
        if incorrect_hyps:
            incorrect_hyp_scores = [hyp.score for hyp in incorrect_hyps]
            if type in ('best', 'sample'):
                if type == 'best':
                    sample_idx = np.argmax(incorrect_hyp_scores)
                    sampled_hyp = incorrect_hyps[sample_idx]
                else:
                    incorrect_hyp_probs = [np.exp(score) for score in incorrect_hyp_scores]
                    incorrect_hyp_probs = np.array(incorrect_hyp_probs) / sum(incorrect_hyp_probs)
                    sampled_hyp = np.random.choice(incorrect_hyps, size=1, p=incorrect_hyp_probs)
                    sampled_hyp = sampled_hyp[0]

                sample = Example(idx='negative-%s' % _example.idx,
                                 src_sent=_example.src_sent,
                                 tgt_code=sampled_hyp.code,
                                 tgt_actions=None,
                                 tgt_ast=None)
                return sample
            elif type == 'all':
                samples = []
                for i, hyp in enumerate(incorrect_hyps):
                    sample = Example(idx='negative-%s-%d' % (_example.idx, i),
                                     src_sent=_example.src_sent,
                                     tgt_code=hyp.code,
                                     tgt_actions=None,
                                     tgt_ast=None)
                    samples.append(sample)

                return samples
        else:
            return None

    print('begin training decoder, %d training examples, %d dev examples' % (len(train_set), len(dev_set)),
          file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True,
                                                   decode_max_time_step=args.decode_max_time_step):
            if train_paraphrase_model:
                positive_examples_num = len(batch_examples)
                labels = [0] * len(batch_examples)
                negative_samples = []
                batch_decoding_results = [train_decode_results[e.idx] for e in batch_examples]
                # sample negative examples
                for example, hyps in zip(batch_examples, batch_decoding_results):
                    if hyps:
                        negative_sample = get_negative_example(example, hyps, type=args.negative_sample_type)
                        if negative_sample:
                            if isinstance(negative_sample, Example):
                                negative_samples.append(negative_sample)
                                labels.append(1)
                            else:
                                negative_samples.extend(negative_sample)
                                labels.extend([1] * len(negative_sample))

                batch_examples += negative_samples

            train_iter += 1
            optimizer.zero_grad()

            nll = -model(batch_examples)
            if train_paraphrase_model:
                idx_tensor = Variable(torch.LongTensor(labels).unsqueeze(-1), requires_grad=False)
                if args.cuda: idx_tensor = idx_tensor.cuda()
                loss = torch.gather(nll, 1, idx_tensor)
            else:
                loss = nll

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
        eval_start = time.time()
        # evaluate dev_score
        dev_acc = evaluate_paraphrase_acc() if train_paraphrase_model else -evaluate_ppl()
        print('[Epoch %d] dev_score=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args.patience:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args: Args):
    dataset_cls: Type[Dataset] = Registrable.by_name(args.dataset)
    test_set = dataset_cls.from_bin_file(args.test_file, args, mode="eval")
    assert args.load_model

    flutes.log('load model from [%s]' % args.load_model)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    flutes.log(str(eval_results))
    with (args.output_dir / "decode.test.txt").open("w") as f:
        for result in decode_results:
            if len(result) > 0:
                f.write(result[0].code.replace("\n", " "))
            else:
                f.write("<decode failed>")
            f.write("\n")
    with (args.output_dir / "decode.test.pkl").open("wb") as f:
        pickle.dump(decode_results, f)


def interactive_mode(args: Args):
    """Interactive mode"""
    print('Start interactive mode', file=sys.stderr)

    parser = StandaloneParser(args.parser,
                              args.load_model,
                              args.example_preprocessor,
                              beam_size=args.beam_size,
                              cuda=args.cuda)

    while True:
        utterance = input('Query:').strip()
        hypotheses = parser.parse(utterance, debug=True)

        for hyp_id, hyp in enumerate(hypotheses):
            print('------------------ Hypothesis %d ------------------' % hyp_id)
            print(hyp.code)
            # print(hyp.tree.to_string())
            # print('Actions:')
            # for action_t in hyp.action_infos:
            #     print(action_t.__repr__(True))


def train_reranker_and_test(args: Args):
    print('load dataset [test %s], [dev %s]' % (args.test_file, args.dev_file), file=sys.stderr)
    test_set = Dataset.from_bin_file(args.test_file, args, mode="eval")
    dev_set = Dataset.from_bin_file(args.dev_file, args, mode="eval")

    features = []
    i = 0
    while i < len(args.features):
        feat_name = args.features[i]
        feat_cls = Registrable.by_name(feat_name)
        print('Add feature %s' % feat_name, file=sys.stderr)
        if issubclass(feat_cls, nn.Module):
            feat_path = os.path.join('saved_models/conala/', args.features[i] + '.bin')
            feat_inst = feat_cls.load(feat_path)
            print('Load feature %s from %s' % (feat_name, feat_path), file=sys.stderr)
        else:
            feat_inst = feat_cls()

        features.append(feat_inst)
        i += 1

    transition_system = next(feat.transition_system for feat in features if hasattr(feat, 'transition_system'))
    evaluator = Registrable.by_name(args.evaluator)(transition_system)

    print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
    dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
    dev_eval_results = evaluator.evaluate_dataset(dev_set, dev_decode_results, fast_mode=False)

    print('load test decode results [%s]' % args.test_decode_file, file=sys.stderr)
    test_decode_results = pickle.load(open(args.test_decode_file, 'rb'))
    test_eval_results = evaluator.evaluate_dataset(test_set, test_decode_results, fast_mode=False)

    print('Dev Eval Results', file=sys.stderr)
    print(dev_eval_results, file=sys.stderr)
    print('Test Eval Results', file=sys.stderr)
    print(test_eval_results, file=sys.stderr)

    if args.load_reranker:
        reranker = GridSearchReranker.load(args.load_reranker)
    else:
        reranker = GridSearchReranker(features, transition_system=transition_system)

        if args.num_workers == 1:
            reranker.train(dev_set.examples, dev_decode_results, evaluator=evaluator)
        else:
            reranker.train_multiprocess(dev_set.examples, dev_decode_results, evaluator=evaluator,
                                        num_workers=args.num_workers)

        if args.save_to:
            print('Save Reranker to %s' % args.save_to, file=sys.stderr)
            reranker.save(args.save_to)

    test_score_with_rerank = reranker.compute_rerank_performance(test_set.examples, test_decode_results, verbose=True,
                                                                 evaluator=evaluator, args=args)

    print('Test Eval Results After Reranking', file=sys.stderr)
    print(test_score_with_rerank, file=sys.stderr)


def main():
    sys.setrecursionlimit(32768)
    flutes.register_ipython_excepthook(capture_keyboard_interrupt=True)
    args = init_config()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    flutes.set_log_file(args.output_dir / f"log.{args.mode}.txt")

    flutes.log(args.to_string(), timestamp=False)

    if args.mode == 'train':
        train(args)
    elif args.mode in ('train_reconstructor', 'train_paraphrase_identifier'):
        train_rerank_feature(args)
    elif args.mode == 'rerank':
        train_reranker_and_test(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'interactive':
        interactive_mode(args)
    else:
        raise RuntimeError('unknown mode')


if __name__ == '__main__':
    main()
