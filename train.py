import copy
import logging
import os

import dill
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kma.common.util import unique_everseen
from kma.dataset import SejongDataset
from kma.decoders.rnn_decoder import RNNDecoderPointer
from kma.encoders.rnn_encoder import RNNEncoder
from kma.models.model import KMAModel
from kma.taggers.crf_tagger import CRFTagger

with open(os.path.join('config', 'kma.yaml'), 'r') as f:
    config = yaml.load(f)

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
logger.info('Device: %s' % device)

WORD = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
LEX = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, is_target=True)
POS_TAG = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True,
                               unk_token=None, is_target=True)

train, valid = SejongDataset.splits(fields=(WORD, LEX, POS_TAG),
                                    train=config['train_file'],
                                    validation=config['validation_file'],
                                    test=None,
                                    max_token=config['preprocessing']['max_token'])

logger.info('---------- Sejong Dataset ---------')
logger.info('Train size: %d' % (len(train)))
logger.info('Validation size: %d' % (len(valid)))

# build vocab
min_freq = config['preprocessing']['min_freq']
WORD.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"], min_freq=min_freq)
LEX.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"], min_freq=min_freq)
POS_TAG.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"])

# Update LEX Vocab for pointer network
# TODO: Write the code elaborately
MERGE = copy.deepcopy(WORD)
for k in LEX.vocab.stoi:
    if k not in MERGE.vocab.stoi:
        MERGE.vocab.stoi[k] = len(MERGE.vocab.stoi)
MERGE.vocab.itos.extend(LEX.vocab.itos)
MERGE.vocab.itos = unique_everseen(MERGE.vocab.itos)
MERGE.is_target = True
MERGE.include_lengths = False
LEX = MERGE

train.fields['lex'] = LEX
valid.fields['lex'] = LEX

logger.info('Size of WORD vocab: %d' % len(WORD.vocab))
logger.info('Size of LEX vocab: %d' % len(LEX.vocab))
logger.info('Size of POS_TAG vocab: %d' % len(POS_TAG.vocab))

vocab = {"WORD": WORD, "LEX": LEX, "POS": POS_TAG}
with open(config['vocab_name'], 'wb') as fout:
    dill.dump(vocab, fout)

encoder = RNNEncoder(vocab_size=len(WORD.vocab), pad_id=WORD.vocab.stoi[WORD.pad_token], **config['encoder'])
if config['encoder']['bidirectional']:
    hidden_size = config['encoder']['hidden_size'] * 2
else:
    config['encoder']['hidden_size']
decoder = RNNDecoderPointer(vocab_size=len(LEX.vocab), hidden_size=hidden_size,
                            sos_id=LEX.vocab.stoi[LEX.init_token], eos_id=LEX.vocab.stoi[LEX.eos_token],
                            pad_id=LEX.vocab.stoi[LEX.pad_token], **config['decoder'])
tagger = CRFTagger(hidden_size=hidden_size, num_tags=len(POS_TAG.vocab))
model = KMAModel(encoder, decoder, tagger).to(device)
logger.info(model)

model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = getattr(optim, config['optimizer']['optim'])(model_params, lr=config['optimizer']['lr'])
scheduler = ReduceLROnPlateau(optimizer, 'min')
criterion = nn.NLLLoss(ignore_index=LEX.vocab.stoi[LEX.pad_token])

train_iter = torchtext.data.BucketIterator(train, batch_size=config['learning']['batch_size'],
                                           sort_key=lambda x: x.word.__len__(),
                                           sort_within_batch=True,
                                           shuffle=True)

valid_iter = torchtext.data.BucketIterator(valid, batch_size=config['learning']['batch_size'],
                                           sort_key=lambda x: x.word.__len__(),
                                           sort_within_batch=True,
                                           shuffle=True)

for epoch in range(config['learning']['epochs']):
    model.train()
    train_loss = 0

    for data in train_iter:
        decoder_outputs, tagger_loss, others = model(data.word[0].to(device), data.lex.to(device), data.pos.to(device),
                                                     input_lengths=None,
                                                     teaching_force_ratio=config['decoder']['teaching_force_ratio'])
        optimizer.zero_grad()
        lex_loss = 0
        for step, step_output in enumerate(decoder_outputs):
            batch_size = data.lex.size(0)
            lex_loss += criterion(step_output.view(batch_size, -1), data.lex[:, step + 1].to(device))

        loss = lex_loss + tagger_loss
        train_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['learning']['max_grad_norm'])
        optimizer.step()

    model.eval()
    with torch.no_grad():
        dev_loss = 0
        for val_data in valid_iter:
            decoder_outputs, tagger_loss, others = model(val_data.word[0].to(device),
                                                         val_data.lex.to(device),
                                                         val_data.pos.to(device),
                                                         input_lengths=val_data.word[1])

            dev_loss += tagger_loss
            for step, step_output in enumerate(decoder_outputs):
                batch_size = val_data.lex.size(0)
                dev_loss += criterion(step_output.view(batch_size, -1),
                                      val_data.lex[:, step + 1].to(device))

    scheduler.step(dev_loss)
    logger.info("Epoch: %d, Train loss: %f, Dev loss: %f" % (epoch, train_loss, dev_loss))

dict_save = {"model": model.state_dict(), "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
torch.save(dict_save, config['model_name'])
