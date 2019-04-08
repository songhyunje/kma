import argparse
import io
import logging
import os

import dill
import torch
import torchtext
import yaml
from torchtext.data import Dataset

from kma.common.util import syllable_to_eojeol
from kma.dataset import POSExample
from kma.decoders.rnn_decoder import RNNDecoderPointer
from kma.encoders.rnn_encoder import RNNEncoder
from kma.models.model import KMAModel
from kma.taggers.crf_tagger import CRFTagger


def parse_option():
    option = argparse.ArgumentParser(description='Korean morphological analyzer')
    option.add_argument('--input_file', type=str, required=True)
    option.add_argument('--output_file', type=str, required=True)
    return option.parse_args()


with open(os.path.join('config', 'kma.yaml'), 'r') as f:
    config = yaml.load(f)

args = parse_option()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
logger.info('Device: %s' % device)

with open(config['vocab_name'], 'rb') as fin:
    vocab = dill.load(fin)
WORD, LEX, POS_TAG = vocab['WORD'], vocab['LEX'], vocab['POS']

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
checkpoint = torch.load(config['model_name'])
model.load_state_dict(checkpoint['model'])
logger.info(model)
model.eval()

with io.open(args.input_file, 'r', encoding='utf-8') as f:
    sents = f.readlines()

named_field = ('word', WORD)
examples = []
for sent in sents:
    syllables = [list(eojeol) for eojeol in sent.split()]
    examples.append(POSExample.fromsent(syllables, named_field))

text_dataset = Dataset(examples, [named_field])
text_iter = torchtext.data.BucketIterator(text_dataset, batch_size=5, shuffle=False)

outputs = []
with torch.no_grad():
    for t in text_iter:
        decoder_outputs, tagger_outputs, others = model.infer(t.word[0].to(device))
        for n in range(len(others['length'])):
            length = others['length'][n]
            tgt_id_seq = [others['sequence'][di][n].item() for di in range(length)]
            tag_id_seq = [tagger_outputs[n][di].item() for di in range(length)]
            result = [WORD.vocab.itos[tgt] + "/" + POS_TAG.vocab.itos[pos] for tgt, pos in
                      zip(tgt_id_seq, tag_id_seq)]
            outputs.append(result)

with io.open(args.output_file, 'w', encoding='utf-8') as f:
    for output in outputs:
        f.write(' '.join('{}/{}'.format(*mt) for mt in syllable_to_eojeol(output)) + '\n')
