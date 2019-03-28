import torch.nn as nn


class KMAModel(nn.Module):
    def __init__(self, encoder, decoder, tagger=None):
        super(KMAModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tagger = tagger

    def forward(self, input_seqs, target_lex_seqs, target_pos_seqs, input_lengths=None, teaching_force_ratio=0):
        # input_seqs: [batch x max_length]
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        decoder_outputs, last_hidden, others = self.decoder(target_lex_seqs, encoder_hidden, encoder_outputs,
                                                            teacher_forcing_ratio=teaching_force_ratio,
                                                            src_input=input_seqs)
        if self.tagger:
            tagger_loss = self.tagger(last_hidden, target_pos_seqs[:, 1:])
            return decoder_outputs, tagger_loss, others
        else:
            return decoder_outputs, last_hidden, others

    def infer(self, input_seqs, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        decoder_outputs, last_hidden, others = self.decoder(None, encoder_hidden, encoder_outputs,
                                                            teacher_forcing_ratio=0,
                                                            src_input=input_seqs)
        if self.tagger:
            tagger_outputs = self.tagger.infer(last_hidden)
            return decoder_outputs, tagger_outputs, others
        else:
            return decoder_outputs, last_hidden, others
