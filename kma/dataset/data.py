import json


class Sentence(object):
    def __init__(self, text='', morphs=None):
        self.text = text
        # self.result = []
        self.morphs = morphs if morphs else []

    def add_morph(self, m):
        self.morphs.append(m)

    def add_result(self, syllable_seq):
        self._post(syllable_seq)

    def _post(self, syllable_seq):
        tmp = []
        for seq in syllable_seq:
            lex, pos = seq.rsplit("/", 1)
            if lex == '<eos>' or lex == '<blank>' or lex == '<tok>':
                if not tmp:
                    continue

                lemma = ''.join([s for (s, p) in tmp])
                pos = tmp[0][1]

                morph = Morph(lemma, pos)
                self.add_morph(morph)

                if lex == '<eos>':
                    break
                else:
                    tmp = []
            else:
                tmp.append((lex, pos))

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, ensure_ascii=False)


class Morph(object):
    def __init__(self, lemma, pos):
        self.lemma = lemma
        self.pos = pos

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, ensure_ascii=False)
