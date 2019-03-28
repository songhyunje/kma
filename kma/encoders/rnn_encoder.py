import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, pad_id, embeddings=None, **kwargs):
        super(RNNEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, kwargs['vocab_embed_dim'])
        self.embeddings.padding_idx = pad_id
        if embeddings is not None:
            self.embeddings.weight = nn.Parameter(embeddings)
            self.embeddings.weight.requires_grad = False

        self.input_dropout = nn.Dropout(kwargs['input_dropout'])
        self.rnn = getattr(nn, kwargs['rnn_type'])(input_size=self.embeddings.embedding_dim,
                                                   hidden_size=kwargs['hidden_size'],
                                                   num_layers=kwargs['rnn_layer'],
                                                   dropout=kwargs['rnn_dropout'],
                                                   bidirectional=kwargs['bidirectional'],
                                                   batch_first=True)

    def forward(self, input_seqs, input_lengths=None):
        embedded = self.embeddings(input_seqs)  # batch x s_len x emb_dim
        embedded = self.input_dropout(embedded)

        if input_lengths is not None:
            embedded = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)

        if input_lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)  # unpack (back to padded)

        return output, hidden
