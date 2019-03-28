import numpy as np


def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0] * t[1], sizes)), sum(map(lambda t: t[0] * (1 - t[1]), sizes))


def unique_everseen(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def syllable_to_eojeol(syllable_seq):
    eojeol_seq = []
    tmp = []
    for tagged_syllable in syllable_seq:
        lex, pos = tagged_syllable.rsplit("/", 1)
        if lex == '<eos>' or pos == '<eos>':
            if not tmp:
                continue
            morpheme = ''.join([s for (s, p) in tmp])
            predicted_pos = tmp[0][1]
            eojeol_seq.append((morpheme, predicted_pos))
            break
        if lex == '<blank>' or lex == '<tok>':
            if not tmp:
                continue
            morpheme = ''.join([s for (s, p) in tmp])
            predicted_pos = tmp[0][1]
            eojeol_seq.append((morpheme, predicted_pos))
            tmp = []
        else:
            tmp.append((lex, pos))

    return eojeol_seq
