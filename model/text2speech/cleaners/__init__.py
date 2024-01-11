import re
from functools import partial

from num2words import num2words

from . import en, es, ja, ko, zhCN

_number_re = re.compile(r'([0-9]+)')

def normalize_special(text):
    for f in ["\u200B", "\\\\N", "\{[^}]*\}", "⺀", "⺙", "⺛", "⻳", "⼀", "⿕", "〇", "〡", "〩", "〸", "〺", "〻", "㐀", "䶵", "_", "`", "·", "'̀", "'́", "'̌", "▽", "\x14", "\x17", "\\\\", "\u3000", "\ue105", "\ufeff"]:
        text = re.sub(f, '', text)
    for f, t in zip('\xa0\x01\n<>{}「」『』―—・ー─剝禕龐！％０１２３４５６７８９？ＩＸ', '   [][][][]----~はいぼ!%0123456789?1X'):
        text = re.sub(f, t, text)
    text = re.sub("…", '...', text)
    text = re.sub('。', '. ', text)
    text = re.sub('\. *$', '.', text)
    text = re.sub('、', ', ', text)
    text = re.sub(', *$', ',', text)
    text = re.sub("[`‘’]", '\'', text)
    text = re.sub("[＂“”˝]", '\"', text)
    text = re.sub("～", '~ ', text)
    return text


class Cleaner(object):
    def __call__(self, text, lang):
        return getattr(self, re.sub('-', '', lang))(text)

    def en(self, text):
        text = re.sub('￥', 'yuan', text)
        text = normalize_special(text)
        text = en.expand_numbers(text)
        text = en.expand_abbreviations(text)
        return text

    def es(self, text):
        text = re.sub('￥', 'yuan', text)
        text = normalize_special(text)
        text = es.expand_numbers(text)
        text = en.expand_abbreviations(text)
        return text

    def zhCN(self, text):
        #text = re.sub(' ', '', text)
        text = re.sub('￥', '元', text)
        text = normalize_special(text)
        text = zhCN.expand_numbers(text)
        text = en.expand_abbreviations(text)
        return text

    def ko(self, text):
        text = re.sub('￥', '위안', text)
        text = ko.normalize_manual(text)
        text = normalize_special(text)
        text = ko.expand_numbers(text)
        text = en.expand_abbreviations(text)
        return text

    def ja(self, text):
        #text = re.sub(' ', '', text)
        text = re.sub('￥', '円', text)
        text = normalize_special(text)
        text = ja.expand_numbers(text)
        text = en.expand_abbreviations(text)
        return text

    # TODO
    def de(self, text):
        text = re.sub(_number_re, lambda m: num2words(m.group(0), lang='de'), text)
        text = en.expand_abbreviations(text)
        return text

    def fr(self, text):
        text = re.sub(_number_re, lambda m: num2words(m.group(0), lang='fr'), text)
        text = en.expand_abbreviations(text)
        return text

    def ru(self, text):
        text = re.sub(_number_re, lambda m: num2words(m.group(0), lang='ru'), text)
        text = en.expand_abbreviations(text)
        return text

    def th(self, text):
        text = re.sub(_number_re, lambda m: num2words(m.group(0), lang='th'), text)
        text = en.expand_abbreviations(text)
        return text

    def vi(self, text):
        # TODO vi num2words , str_to_number function
        text = re.sub(_number_re, lambda m: num2words(m.group(0), lang='vi'), text)
        text = en.expand_abbreviations(text)
        return text


cleaner = Cleaner()
