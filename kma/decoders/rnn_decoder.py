import logging
import random

import numpy as np
import torch
import torch.nn as nn

from kma.decoders.copy_decoder import CopyDecoder
from kma.decoders.default_rnn_decoder import RNNDecoder, validate_args

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)


class RNNDecoderPointer(RNNDecoder):
    def __init__(self, vocab_size, hidden_size, sos_id, eos_id, pad_id,
                 embeddings=None,
                 bidirectional_encoder=True,
                 use_attention=True,
                 **kwargs):
        super().__init__(vocab_size, hidden_size, sos_id, eos_id, pad_id,
                         embeddings=embeddings,
                         bidirectional_encoder=bidirectional_encoder,
                         use_attention=use_attention,
                         **kwargs)

        self.output_size = vocab_size
        self.decoder = CopyDecoder(self.hidden_size, self.output_size, self.vocab_embed_dim)
        self.concat = nn.Linear(self.hidden_size * 2 + self.vocab_embed_dim, self.hidden_size)

    def forward(self, input_seqs, encoder_hidden, encoder_outputs,
                teacher_forcing_ratio=0, src_input=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[RNNDecoder.KEY_ATTN_SCORE] = list()

        # valid arguments
        input_seqs, batch_size, max_length = validate_args(input_seqs, encoder_hidden, encoder_outputs,
                                                           use_attention=self.attention,
                                                           rnn_type=self.rnn.mode,
                                                           sos_id=self.sos_id,
                                                           teacher_forcing_ratio=teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def post_decode(step_output, step_symbols, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[RNNDecoder.KEY_ATTN_SCORE].append(step_attn)
            sequence_symbols.append(step_symbols)

            eos_batches = step_symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)

        # Manual unrolling is used to support teacher forcing
        if use_teacher_forcing:
            decoder_input = input_seqs[:, 0].unsqueeze(1)
            concated_contexts = []
            for di in range(max_length):
                context, decoder_hidden, attn, decoder_input_embed = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs)
                decoder_output, symbols = self.decoder(context, attn, src_input, decoder_input_embed, decoder_hidden)

                step_output = decoder_output.squeeze(1)
                step_symbols = symbols.squeeze(1)
                post_decode(step_output, step_symbols, attn)
                decoder_input = input_seqs[:, di+1].unsqueeze(1)

                concated_vec = self.concat(torch.cat([context, decoder_hidden[0][-1].unsqueeze(1),
                                                      self.embeddings(symbols.squeeze(2))], dim=2))
                concated_contexts.append(concated_vec)

            concated_contexts = torch.cat(concated_contexts, dim=1)
        else:
            decoder_input = input_seqs[:, 0].unsqueeze(1)

            concated_contexts = []
            for di in range(max_length):
                context, decoder_hidden, attn, decoder_input_embed = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs)
                decoder_output, symbols = self.decoder(context, attn, src_input, decoder_input_embed, decoder_hidden)

                step_output = decoder_output.squeeze(1)
                step_symbols = symbols.squeeze(1)
                post_decode(step_output, step_symbols, attn)
                decoder_input = step_symbols

                concated_vec = self.concat(torch.cat([context, decoder_hidden[0][-1].unsqueeze(1),
                                                      self.embeddings(symbols.squeeze(2))], dim=2))
                concated_contexts.append(concated_vec)

            concated_contexts = torch.cat(concated_contexts, dim=1)

        ret_dict[RNNDecoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[RNNDecoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, concated_contexts, ret_dict
