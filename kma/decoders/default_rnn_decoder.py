import logging

import torch
import torch.nn as nn

from kma.modules.attention import Attention

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)


class RNNDecoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, hidden_size, sos_id, eos_id, pad_id, embeddings,
                 bidirectional_encoder, use_attention, **kwargs):
        super(RNNDecoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, kwargs['vocab_embed_dim'])
        self.embeddings.padding_idx = pad_id
        if embeddings is not None:
            self.embeddings.weight = nn.Parameter(embeddings)
            self.embeddings.weight.requires_grad = False
        self.input_dropout = nn.Dropout(kwargs['input_dropout'])

        self.vocab_embed_dim = self.embeddings.embedding_dim
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, kwargs['rnn_type'])(input_size=self.embeddings.embedding_dim,
                                                   hidden_size=self.hidden_size,
                                                   num_layers=kwargs['rnn_layer'],
                                                   dropout=kwargs['rnn_dropout'],
                                                   batch_first=True)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.use_attention = use_attention
        self.bidirectional_encoder = bidirectional_encoder

        if self.use_attention:
            self.attention = Attention(self.hidden_size)

    def _init_state(self, encoder_hidden):
        """ Init decoder start with last state of the encoder """

        def _fix_enc_hidden(h):
            """ If encoder is bidirectional, do the following transformation.
            [layer*directions x batch x dim] -> [layer x batch x directions*dim]
            """
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):  # LSTM
            encoder_hidden = tuple([_fix_enc_hidden(h) for h in encoder_hidden])
        else:
            encoder_hidden = _fix_enc_hidden(encoder_hidden)
        return encoder_hidden

    def forward_step(self, input_seqs, hidden, encoder_outputs):
        embedded = self.embeddings(input_seqs)
        assert embedded.dim() == 3  # [batch x len x emb_dim]

        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        # TODO: check dropout layer for output
        return output, hidden, attn, embedded

    def forward(self, input_seqs, encoder_hidden, encoder_outputs,
                teacher_forcing_ratio=0,
                src_input=None):
        pass


def validate_args(input_seqs, encoder_hidden, encoder_output, use_attention, rnn_type, sos_id, teacher_forcing_ratio):
    if use_attention:
        if encoder_output is None:
            raise ValueError("Argument encoder_output cannot be None "
                             "when attention is used.")

    if input_seqs is None and encoder_hidden is None:
        batch_size = 1
    else:
        if input_seqs is not None:
            batch_size = input_seqs.size(0)  # [batch x max_len]
        else:
            if rnn_type == 'LSTM':
                batch_size = encoder_hidden[0].size(1)
            elif rnn_type == 'GRU':
                batch_size = encoder_hidden.size(1)
            else:
                raise ValueError("Unknown rnn mode is provided.")

    # set default input and max decoding length
    if input_seqs is None:
        if teacher_forcing_ratio > 0:
            raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        if rnn_type == 'LSTM':
            device = encoder_hidden[0].device
        elif rnn_type == 'GRU':
            device = encoder_hidden.device
        else:
            raise ValueError("Unknown rnn mode is provided.")

        input_seqs = torch.LongTensor([sos_id] * batch_size).view(batch_size, 1).to(device)

        if use_attention:
            max_length = int(encoder_output.size(1) * 1.5)
        else:
            max_length = 200
    else:
        max_length = input_seqs.size(1) - 1  # minus the start of sequence symbol

    return input_seqs, batch_size, max_length
