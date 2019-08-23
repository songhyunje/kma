import io
import logging
import os
from itertools import chain

import torch
from torchtext.data import Batch, Dataset, Example

logger = logging.getLogger(__name__)


class SejongDataset(Dataset):
    """ Define a dataset for Korean POS tagging."""
    def __init__(self, fn, fields, **kwargs):
        """Create a Sejong dataset given path and fields"""

        if not isinstance(fields[0], (tuple, list)):
            fields = [('word', fields[0]), ('lex', fields[1]), ('pos', fields[2])]

        max_token = kwargs['max_token'] if 'max_token' in kwargs else None
        examples = []
        with io.open(fn, mode='r', encoding='utf-8') as f:
            sent = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    # Split long sentences
                    if max_token and len(sent) > max_token:
                        for i in range(0, len(sent), max_token):
                            examples.append(POSExample.fromlist(zip(*sent[i:i+max_token]), fields))
                    else:
                        examples.append(POSExample.fromlist(zip(*sent), fields))

                    sent = []
                    continue

                # 아름다운 아름답/VA+ㄴ/ETM
                word, tagged = line.split()
                lex_seq, pos_seq = zip(*[lex_pos.rsplit('/', maxsplit=1) for lex_pos in tagged.split('+')])

                flatten_word = list(word)
                flatten_lex, flatten_pos = [], []
                for lex, pos in zip(lex_seq, pos_seq):
                    for l in lex:
                        flatten_lex.append(l)
                        flatten_pos.append(pos)
                    # We know that a morpheme has one POS tag
                    # flatten_lex.append(lex[0])
                    # flatten_pos.append('B-' + pos)
                    # for l in lex[1:]:
                    #     flatten_lex.append(l)
                    #     flatten_pos.append('I-' + pos)
                    flatten_lex.append('<tok>')
                    flatten_pos.append('<tok>')

                sent.append((flatten_word, flatten_lex[:-1], flatten_pos[:-1]))

        self.examples = examples
        self.fields = dict(fields)
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    @classmethod
    def splits(cls, fields, train='train.txt', validation='valid.txt', test='test.txt', **kwargs):
        train_data = None if train is None else cls(train, fields, **kwargs)
        val_data = None if validation is None else cls(validation, fields, **kwargs)
        test_data = None if test is None else cls(test, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class POSExample(Example):
    """Class to convert a Sejong POS tagged sentence to an Example"""

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if name == 'pos':
                    val = [v + ['<blank>'] for v in val[:-1]] + [val[-1]]
                else:
                    val = [list(v) + ['<blank>'] for v in val[:-1]] + [list(val[-1])]

                val = list(chain.from_iterable(val))
                setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromsent(cls, data, named_fields):
        ex = cls()
        val = [list(v) + ['<blank>'] for v in data[:-1]] + [list(data[-1])]
        val = list(chain.from_iterable(val))
        for name, field in named_fields:
            setattr(ex, name, field.preprocess(val))

        return ex

