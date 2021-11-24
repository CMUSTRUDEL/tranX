from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from texar.torch.modules import TransformerDecoder, TransformerDecoderOutput
# from texar.torch.core import layers
# from texar.torch.modules.decoders.decoder_base import (
#     DecoderBase, TokenEmbedder, TokenPosEmbedder, _make_output_layer)
from texar.torch.modules.decoders.decoder_helpers import (
    EmbeddingHelper, Helper)
# from texar.torch.modules.encoders.multihead_attention import (
#    Cache, MultiheadAttentionEncoder)
# from texar.torch.modules.encoders.transformer_encoder import (
#    default_transformer_poswise_net_hparams)
# from texar.torch.modules.networks.networks import FeedForwardNetwork
from texar.torch.utils import transformer_attentions as attn
# from texar.torch.utils.beam_search import beam_search
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import sequence_mask


class CTransformerDecoder(TransformerDecoder):
    """
    Texar's transformer decoder internally performs embedding on a single tensor of word IDs as part of the forward pass.

    The documentation suggests creating a subclass to override this behavior if necessary. In our case, this is necessary
    as the embedding phase is more complicated.
    """
    def __init__(self, hparams, embedder, vocab_size=None, output_layer=None, token_embedder=None):
        super().__init__(hparams=hparams, token_embedder=token_embedder, vocab_size = vocab_size, output_layer=output_layer)
        self.custom_embedder = embedder
    
    def embed_tokens(self, input):
        return self.custom_embedder(input)
    
    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                memory: Optional[torch.Tensor] = None,
                memory_sequence_length: Optional[torch.LongTensor] = None,
                memory_attention_bias: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_sequence_length: Optional[torch.LongTensor] = None,
                helper: Optional[Helper] = None,
                decoding_strategy: str = 'train_greedy',
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None,
                beam_width: Optional[int] = None,
                length_penalty: float = 0.,
                **kwargs) -> torch.Tensor:
        """
        This function is a slightly modified version of forward for the default texar TransformerDecoder.

        However, this forward method allows for freedom in the arguments to the token_embedder.
        """
    
        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - sequence_mask(
                    memory_sequence_length, memory.size(1),
                    dtype=torch.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)

        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            if context_sequence_length is None:
                raise ValueError("'context_sequence_length' must not be None"
                                 "when 'context' is specified.")
            self._state_context = context[:, 1:]
            self._state_context_sequence_length = context_sequence_length - 1
        else:
            self._state_context = None
            self._state_context_sequence_length = None

        # Faster code path for teacher-forcing training
        if (helper is None and beam_width is None and
                decoding_strategy == 'train_greedy'):
            if inputs is None:
                raise ValueError("'input' must not be none "
                                 "when using 'train_greedy' decoding strategy.")
            # times = torch.arange(
            #     inputs.size(1), dtype=torch.long, device=inputs.device)
            # times = times.unsqueeze(0).expand(inputs.size(0), -1)

            ### This is a modification to the original Texar implementation. Instead of assuming the data type of "inputs"
            ### in order to calculate positional embeddings, everything is left to the custom_embedder function.
            inputs = self.embed_tokens(inputs)
            if sequence_length is not None:
                inputs = mask_sequences(inputs, sequence_length)

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(inputs.size(1)))

            decoder_output = self._self_attention_stack(
                inputs, memory, decoder_self_attention_bias,
                memory_attention_bias, cache=None)
            logits = self._output_layer(decoder_output)
            # sample_id = torch.argmax(logits, dim=-1)

            return logits

        # Inference code path.
        if max_decoding_length is None:
            max_decoding_length = self._hparams.max_decoding_length

        self._state_max_decoding_length = max_decoding_length

        if beam_width is None or beam_width == 1:  # Inference-like decoding
            # Prepare helper
            if helper is None:
                kwargs.update(decoding_strategy=decoding_strategy)
                if context is not None:
                    kwargs.update(start_tokens=context[:, 0])
                helper = self._create_or_get_helper(infer_mode, **kwargs)
            assert isinstance(helper, EmbeddingHelper)

            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=False, batch_size=helper.batch_size)
            if context is not None:
                assert self._state_context is not None
                pad_length = max_decoding_length - self._state_context.size(1)
                if pad_length > 0:
                    self._state_context = torch.cat((
                        self._state_context,
                        self._state_context.new_zeros(
                            self._state_context.size(0), pad_length)
                    ), dim=1)

            outputs, cache, sequence_lengths = self.dynamic_decode(
                helper, inputs=None, sequence_length=None,
                initial_state=None, max_decoding_length=max_decoding_length,
                impute_finished=impute_finished)
            del cache  # not used

            if context is not None:
                # Here the length of sample_id will be larger than that
                # of logit by 1, because there will be a additional
                # start_token in the returned sample_id.
                # the start_id should be the first token of the
                # given context
                start_tokens = context[:, 0]
                outputs = TransformerDecoderOutput(
                    logits=outputs.logits,
                    sample_id=torch.cat([
                        start_tokens.unsqueeze(1),
                        outputs.sample_id
                    ], dim=1))
                sequence_lengths = sequence_lengths + 1

            return outputs, sequence_lengths

        else:  # Beam-search decoding
            # Ignore `decoding_strategy` and # assume `helper` is not set.
            if helper is not None:
                raise ValueError("Must not set 'beam_width' and 'helper' "
                                 "simultaneously.")
            if context is not None:
                start_tokens = context[:, 0]
            else:
                if 'start_tokens' not in kwargs:
                    raise ValueError(
                        "'start_tokens' must be specified when using"
                        "beam search decoding.")
                start_tokens = kwargs['start_tokens']
            _batch_size = start_tokens.size(0)
            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=True,
                batch_size=_batch_size)
            end_token: int = kwargs.get('end_token')  # type: ignore

            # The output format is different when running beam search.
            sample_id, log_prob = self.beam_decode(
                start_tokens,
                end_token,
                embedding_fn=self.embed_tokens,
                beam_width=beam_width,
                length_penalty=length_penalty,
                decode_length=max_decoding_length)

            return {
                'sample_id': sample_id,
                'log_prob': log_prob
            }

