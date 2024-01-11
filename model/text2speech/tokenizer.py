# -*- coding: utf-8 -*-
#!/usr/bin/python3

import re

import typing as tp
import jaconv
import jamo
import numpy as np
import pyopenjtalk
from cutlet import Cutlet
from pypinyin import Style, lazy_pinyin


# '々ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲンヴヶ'
# 'あ い う え お か き く け こ さ し す せ そ た ち つ て と な に ぬ ね の は ひ ふ へ ほ ま み む め も や ゆ よ ら り る れ ろ わ を ん が ぎ ぐ げ ご ざ じ ず ぜ ぞ だ ぢ づ で ど ば び ぶ べ ぼ ぱ ぴ ぷ ぺ ぽ きゃ きゅ きょ ぎゃ ぎゅ ぎょ しゃ しゅ しょ じゃ じゅ じょ ちゃ ちゅ ちょ にゃ にゅ にょ ひゃ ひゅ ひょ びゃ びゅ びょ ぴゃ ぴゅ ぴょ みゃ みゅ みょ りゃ りゅ りょ'
# chinese_chars = 'abcdefghijklmnopqrstuvwxyz12345'
# japanese_chars = 'abdeghikmnoprstuwyz'


class Dutlet(Cutlet):
    def __init__(self, system='nihon'):
        super().__init__(system)
        self.table['’'] = "'"
        self.table[' '] = " "

    def get_single_mapping(self, pk, kk, nk):
        if kk == 'っ':
            if nk:
                if self.use_tch and nk == 'ち':
                    return 't'
                elif nk in 'あいうえおっ':
                    return '-'
                else:
                    return self.table[nk][0]  # first character
            else:
                # seems like it should never happen, but 乗っ|た is two tokens
                # so leave this as is and pick it up at the word level
                return 't'
        return super().get_single_mapping(pk, kk, nk)


class Tokenizer():
    def __init__(self):
        # 간단한 문자열 전처리 작업에 활용하기 위해 미리 지정한다.
        self.unify_punct = {
            '[–—―]': '-',
            '[\[\]`´ʹʻʼʽʿ˂˃ˈˊˋ˹˺˻˼‘’‹›〈〉「」]': '\'',
            '[¨«»ʺ˝˝“”《》『』＂]': '"',
            '[（{]【': '[',
            '[）}]】': ']',
            '…': '...',
            '‧': ' ',
            '[、，]': ', ',
            '。': '. ',
            '？': '?',
            '！': '!',
            '；': ';',
            '：': ':',
            # 'ฃ': 'ข',
        }
        self.invalid_chars = r'[\|]'

        self.pad = '_'
        self.sos = '^'
        self.eos = '$'
        self.punc = '.,\'\"!?¿;:~-'
        self.space = ' '

        self.valid_chars = {}
        self.chars: tp.Dict[tp.AnyStr, tp.AnyStr] = {}
        latin_alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.chars['en'] = latin_alphabet.upper() + latin_alphabet
        self.chars['es'] = self.chars['en'] + "ÁÉÍÑÓÚáéíñóúü"
        self.chars['de'] = self.chars['en'] + "ÄÖÜßäöü"
        self.chars['fr'] = self.chars['en'] + \
            "ÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ"
        self.chars['zh-CN'] = ' 12345abcdefghijklmnopqrstuvwxyz'
        self.chars['ja'] = ' abdeghikmnoprstuvwyz'
        self.chars['ru'] = ''.join(
            map(chr, range(0x0410, 0x0450))) + '\u0401' + '\u0451'
        self.chars['ko'] = ''.join(map(
            chr, [*range(0x1100, 0x1113), *range(0x1161, 0x1176), *range(0x11A8, 0x11C3)]))
        # self.alphabet['th'] = '.123458\\abcdefhijklmnoprstuwŋɛɤɯʔʰːๆᴐ'
        self.chars['th'] = ''.join(
            map(chr, [*range(0xE01, 0xE3A), *range(0xE3F, 0xE5B)]))
        vi_alphabet = "abcdeghiklmnopqrstuvxy"
        c2n_vi = {
            'á': 'a2', 'à': 'a1', 'ả': 'a4', 'ã': 'a3', 'ạ': 'a5',
            'â': 'a6', 'ấ': 'a62', 'ầ': 'a61', 'ẩ': 'a64', 'ẫ': 'a63', 'ậ': 'a65',
            'ă': 'a7', 'ắ': 'a72', 'ằ': 'a71', 'ẳ': 'a74', 'ẵ': 'a73', 'ặ': 'a75',

            'é': 'e2', 'è': 'e1', 'ẻ': 'e4', 'ẽ': 'e3', 'ẹ': 'e5',
            'ê': 'e6', 'ế': 'e62', 'ề': 'e61', 'ể': 'e64', 'ễ': 'e63', 'ệ': 'e65',

            'í': 'i2', 'ì': 'i1', 'ỉ': 'i4', 'ĩ': 'i3', 'ị': 'i5',

            'ó': 'o2', 'ò': 'o1', 'ỏ': 'o4', 'õ': 'o3', 'ọ': 'o5',
            'ô': 'o6', 'ố': 'o62', 'ồ': 'o61', 'ổ': 'o64', 'ỗ': 'o63', 'ộ': 'o65',
            'ơ': 'o7', 'ớ': 'o72', 'ờ': 'o71', 'ở': 'o74', 'ỡ': 'o73', 'ợ': 'o75',

            'ú': 'u2', 'ù': 'u1', 'ủ': 'u4', 'ũ': 'u3', 'ụ': 'u5',
            'ư': 'u7', 'ứ': 'u72', 'ừ': 'u71', 'ử': 'u74', 'ữ': 'u73', 'ự': 'u75',

            'ý': 'y2', 'ỳ': 'y1', 'ỷ': 'y4', 'ỹ': 'y3', 'ỵ': 'y5',

            'đ': 'd7'}

        vi_accent = ''.join(sorted(c2n_vi.keys()))
        self.chars['vi'] = vi_alphabet.upper() + vi_alphabet + \
            vi_accent.upper() + vi_accent
        self.script: tp.Dict[tp.AnyStr, tp.AnyStr] = self.chars.copy()
        self.script['vi'] = '1234567' + vi_alphabet

        for lang in self.chars:
            self.valid_chars[lang] = self.space + self.punc + self.chars[lang]

        # 순서를 지정해주기 위해.
        langs = ['en', 'es', 'de', 'fr', 'zh-CN', 'ja', 'ru', 'ko', 'th', 'vi']
        self.special = self.pad + self.sos + self.eos + self.space + self.punc
        self.all_symbols = self.special + \
            ''.join([c for l in langs for c in sorted(
                set(self.script[l].lower()))])
        len_special = len(self.special)
        abc_idx: tp.List = (
            len_special + np.cumsum(list(map(lambda l: len(set(self.script[l].lower())), langs)))).tolist()
        abc_idx.insert(0, len_special)
        abc_idx = dict(zip(langs, abc_idx))

        self.abc_seq = {l: {v: i + abc_idx[l] for i, v in enumerate(
            sorted(set(self.script[l].lower())))} for l in langs}

        nkatu = Dutlet()
        self.romaji = lambda s: nkatu.map_kana(' '.join(
            [jaconv.kata2hira(node['pron']) for node in pyopenjtalk.run_frontend(s)]))
        self.pinyin = lambda chars: ' '.join(lazy_pinyin(
            chars, style=Style.TONE3, neutral_tone_with_five=True))
        self.jamo = lambda chars: ''.join(jamo.hangul_to_jamo(chars))

        self.vi_c2n = lambda chars: ''.join(
            [c2n_vi[c] if c in c2n_vi else c for c in chars])

    def __call__(self, word, lang):
        processors = []

        # 독자적 문자 활용 (순서 관계 없으므로 다른 함수에 영향받지 않도록 처음부터 진행)
        processors += [self.ko, self.th]

        # 라틴, 키릴 알파벳 계열
        if lang == 'vi' or (re.search(f"[{self.chars['vi'][44:]}]", word) and lang not in ['es', 'fr', 'de']):
            processors += [self.vi, self.es,
                           self.en, self.fr, self.de, self.ru]
        elif lang == 'es' or (re.search(f"[{self.chars['es'][52:]}]", word) and lang not in ['fr', 'de']):
            processors += [self.es, self.en,
                           self.fr, self.de, self.vi, self.ru]
        elif lang == 'de' or (re.search(f"[{self.chars['de'][52:]}]", word) and lang not in ['fr']):
            processors += [self.de, self.en,
                           self.es, self.fr, self.vi, self.ru]
        elif lang == 'fr' or re.search(f"[{self.chars['fr'][52:]}]", word):
            processors += [self.fr, self.en,
                           self.es, self.de, self.vi, self.ru]
        elif lang == 'ru' or re.search(f"[{self.chars['ru']}]", word):
            processors += [self.ru, self.en,
                           self.es, self.fr, self.de, self.vi]
        elif lang == 'en' or re.search(f"[{self.chars['en']}]", word):
            processors += [self.en, self.es,
                           self.fr, self.de, self.vi, self.ru]
        else:
            processors += [self.en, self.es,
                           self.fr, self.de, self.vi, self.ru]

        # 로마자화
        if lang == 'zh-CN':
            processors += [self.zhCN, self.ja]
        elif lang == 'ja':
            processors += [self.ja, self.zhCN]
        else:
            processors += [self.zhCN, self.ja]

        word = [word]
        for processor in processors:
            word = processor(word)

        word = list(filter(lambda c: isinstance(c, int), word))

        return word

    def word_to_int(self, word, lang, func=None):
        output = []
        for chars in word:
            if isinstance(chars, int):
                output += [chars]
            else:
                if func != None:
                    chars = func(chars)
                start = 0
                for idx, c in enumerate(chars):
                    if c in tokenizer.abc_seq[lang]:
                        if start != idx:
                            output += [chars[start:idx]]
                        output += [tokenizer.abc_seq[lang][c]]
                        start = idx + 1
                if start != len(chars):
                    output += [chars[start:]]
        return output

    def en(self, word):
        return self.word_to_int(word, "en")

    def es(self, word):
        return self.word_to_int(word, "es")

    def de(self, word):
        return self.word_to_int(word, "de")

    def fr(self, word):
        return self.word_to_int(word, "fr")

    def ru(self, word):
        return self.word_to_int(word, "ru")

    def zhCN(self, word):
        return self.word_to_int(word, "zh-CN", self.pinyin)

    def ja(self, word):
        return self.word_to_int(word, "ja", self.romaji)

    def ko(self, word):
        return self.word_to_int(word, "ko", self.jamo)

    def th(self, word):
        return self.word_to_int(word, "th")

    def vi(self, word):
        return self.word_to_int(word, "vi", self.vi_c2n)

    def parse(self, text):
        text = text.lower()
        text = self.char_map(text)

        if re.search(self.invalid_chars, text):
            print(f'{text} has invalid character.')
            return [], []

        src_words, puncs_list = [], []
        end_before = 0
        # regex용으로 쓰려면 [] 또한 escape해야한다.
        for m in re.finditer(f'[{self.space}{self.punc}]+', text):
            src_words += [text[end_before:m.start()]]
            puncs_list += [m.group(0)]
            end_before = m.end()
        src_words += [text[end_before:]]
        return src_words, puncs_list

    def char_map(self, text):
        for char in self.unify_punct:
            text = re.sub(char, self.unify_punct[char], text)
        return text


tokenizer = Tokenizer()


def tokenize(text, lang):
    # 단순히 text를 숫자 리스트로 바꿔주는 것 외에 단어 기반 리스트를 반환한다.
    src_words, puncs_list = tokenizer.parse(text)

    tokens, words = [], []
    for src_word, puncs in zip(src_words, puncs_list + [[]]):
        # 각 단어를 기준으로 변경한다.
        word = tokenizer(src_word, lang)
        words += [(src_word, [tokenizer.all_symbols[c]
                   for c in word])] if word else []
        words += [(puncs, list(puncs))] if puncs else []
        tokens += word + [tokenizer.all_symbols.find(c) for c in puncs]

    tokens = [1] + tokens + [2]
    words = [('^', ['^'])] + words + [('$', ['$'])]
    return tokens, words
