import torch
import torch.nn as nn
import torch.nn.functional as F


class CopyDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_dim):
        super(CopyDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_dim = vocab_dim
        self.linear = nn.Linear(hidden_size, output_size)
        self.gen_linear = nn.Linear(hidden_size*2 + vocab_dim, 1)

    def forward(self, context, attn, src_input, decoder_input, decoder_hidden):
        batch_size, de_len = context.size(0), context.size(1)
        logits = self.linear(context.view(-1, self.hidden_size))
        vocab_prob = F.softmax(logits, dim=-1).view(batch_size, de_len, self.output_size)

        # assume that decoder is LSTM
        # decoder_hidden is only for last timestamp, but we need states all timestamps
        # hidden = torch.cat(decoder_hidden).transpose(0, 1)
        # TODO: OpenNMT-py 처럼 decoder를 LSTMCell로 작성해야할 듯

        p_gen_input = torch.cat((context.view(-1, self.hidden_size), 
                                 decoder_hidden[0][-1],
                                 decoder_input.view(-1, self.vocab_dim)), dim=1)
        gen_prob = torch.sigmoid(self.gen_linear(p_gen_input))
        vocab_prob = vocab_prob.view(-1, self.output_size) * gen_prob
        copy_prob = attn.view(-1, attn.size(2)) * (1 - gen_prob)

        vocab_prob = vocab_prob.view(batch_size, de_len, self.output_size)
        copy_prob = copy_prob.view(batch_size, de_len, -1)
        for i in range(copy_prob.size(1)):
            vocab_prob[:, i, :].scatter_add_(1, src_input, copy_prob[:, i, :])

        vocab_prob.log_()
        symbols = vocab_prob.topk(1, dim=2)[1]
        return vocab_prob, symbols
