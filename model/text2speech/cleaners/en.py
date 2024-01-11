import inflect
import re
from unidecode import unidecode

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_fraction_number_re = re.compile(r'(([0-9]+)\+)?([0-9]+)/([0-9]+)')
_telephone_re = re.compile(r'(\d?\d?\d?\d)-(\d?\d\d\d)-(\d\d\d\d)')

digit_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
        '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
        '8': 'eight', '9': 'nine'}

month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'}

def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def _expand_fraction(m):
    mixed = m.group(2)
    numer = m.group(3)
    denom = m.group(4)
    print(m.group(), mixed, numer, denom)

    text = ''
    if mixed != None:
        text += expand_cardinal(mixed) + ' and '
    if numer == '1' and denom == '2':
        text += 'a half'
    elif numer == '1' and denom == '4':
        text += 'a quarter'
    else:
        numer = expand_cardinal(numer)
        denom = expand_ordinal(denom)
        text += '{} {}'.format(expand_cardinal(numer), expand_ordinal(denom))
    return text

def _expand_date(m, format, detail):
    year = m.group(1+format.find('y')) if format.find('y')>=0 else None
    year = '{} {}'.format(expand_cardinal(year[:2]), expand_cardinal(year[2:])) if year else None
    month = m.group(1+format.find('m'))
    month = month_dict[int(month)]
    day = m.group(1+format.find('d'))
    day = expand_ordinal(day)
    if detail == '1':
        return 'The {} of {}, {}'.format(day, month, year) if year else 'The {} of {}'.format(day, month)
    elif detail == '2':
        return '{} {}, {}'.format(month, day, year)

def _expand_time(m, format, detail):
    AMPM = m.groups()[-1]
    print(AMPM)
    if AMPM:
        AMPM = AMPM.upper() if AMPM.upper() in ['AM', 'PM'] else None
    hour = m.group(1+format.find('h'))
    if hour[0] == ':': hour = hour[1:]
    if '12' not in format and detail == '1' and AMPM == 'PM':
        hour = str(12+int(hour))
    hour = expand_cardinal(hour)
    minute = m.group(1+format.find('m')) if format.find('m') >=0 else None
    if minute and minute[0] == ':': minute = minute[1:]
    minute = expand_cardinal(minute) if minute else None
    ## TODO reading second?
    second = m.group(1+format.find('s')) if format.find('s') >=0 else None
    if second and second[0] == ':': second = second[1:]
    return_text = hour
    if minute: return_text += ' ' + minute
    if '24' not in format and detail == '2':
        if AMPM: return_text += ' ' + AMPM
    return return_text

def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text



# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
  (r'%', 'percent'),
  ('i.e.', 'that is'),
  ('e.g.', 'for example'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)

def expand_cardinal(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

def expand_ordinal(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_number_re, lambda x: _inflect.number_to_words(x.group()+'th'), text)
    return text

def expand_fraction(text):
    ## Fraction 
    ## TODO 5+1/2 -> five and a half
    ## 1/2 -> one half, 1/4 -> one quarter
    ## 1/3 -> one third
    text = re.sub(_comma_number_re, _remove_commas, text)
    ## TODO +쪽, 있는 것을 먼저 서치하는 게 맞나?
    text = re.sub(_fraction_number_re, _expand_fraction, text)
    return text

def expand_date(text, format='dmy', detail='1'):
    ## Date
    format = format.lower()
    format = re.sub('[^dmy]', '', format)
    format = re.sub('(d+)|(m+)|(y+)', lambda x: x.group()[0], format)
    for date_sep in ['\.', '/', '-', ' ']:
        if format[0] == 'y': date_re = '(\d\d\d\d){}(\d?\d){}(\d?\d)'.format(date_sep, date_sep)
        elif format[-1] == 'y': date_re = '(\d?\d){}(\d?\d){}(\d\d\d\d)'.format(date_sep, date_sep)
        elif len(format) <= 2: date_re = '(\d?\d){}(\d?\d)'.format(date_sep)
        else: raise ValueError('year information must be on the first or the last position.')
        text = re.sub(date_re, lambda x: _expand_date(x, format, detail), text)

    return text

def expand_time(text, format='hms12', detail='2'):
    format = format.lower()
    format = re.sub('[^hms124]', '', format)
    ## NOTE 맨 뒤에 ?가 붙는 경우는, 아예 무시해도 같은 regex가 되므로 순차적으로 긴 것부터 가져오길.
    time_re = '(\d?\d)(:\d?\d)?(:\d?\d)?([AaPp][Mm])'
    text = re.sub(time_re, lambda x: _expand_time(x, format, detail), text)
    time_re = '(\d?\d)(:\d?\d)(:\d?\d)'
    text = re.sub(time_re, lambda x: _expand_time(x, format, detail), text)
    time_re = '(\d?\d)(:\d?\d)'
    text = re.sub(time_re, lambda x: _expand_time(x, format, detail), text)
    time_re = '(\d?\d)'
    text = re.sub(time_re, lambda x: _expand_time(x, format, detail), text)
    return text

def expand_digits(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_number_re, lambda x: ''.join([digit_dict[digit]+' ' for digit in x.group()]), text)
    return text

def expand_telephone(text):
    text = re.sub(_telephone_re, lambda x: ''.join([digit_dict[digit]+' '\
            if digit in digit_dict else ' ' for digit in x.group()]), text)
    return text

def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)
