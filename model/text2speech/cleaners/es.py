import re
from num2words import num2words

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_libras_re = re.compile(r'£([0-9\,]*[0-9]+)')
_euros_re = re.compile(r'€([0-9\,]*[0-9]+)')
_pesos_re = re.compile(r'[\$₱]([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'([0-9]+)(º|ª|er|ra|do|da|to|ta|mo|ma)')
_number_re = re.compile(r'([0-9]+)')
_fraction_number_re = re.compile(r'(([0-9]+)\+)?([0-9]+)/([0-9]+)')
_telephone_re = re.compile(r'(\d?\d?\d?\d)-(\d?\d\d\d)-(\d\d\d\d)')


digit_dict = {
        '0': 'cero', '1': 'uno', '2': 'dos', '3': 'tres', '4': 'cuatro',
        '5': 'cinco', '6': 'seis', '7': 'siete', '8': 'ocho', '9': 'nueve',
}

month_dict = {1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
        5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
        9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'}

def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' punto ')


def _expand_pesos(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' pesos'    # Unexpected format
    pesos = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if pesos and cents:
        peso_unit = 'peso' if pesos == 1 else 'pesos'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (pesos, peso_unit, cents, cent_unit)
    elif pesos:
        peso_unit = 'peso' if pesos == 1 else 'pesos'
        return '%s %s' % (pesos, peso_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero pesos'

def _expand_ordinal(m):
    return num2words(float(m.group(1)), to='ordinal', lang='es')

def _expand_number(m):
    return num2words(m.group(), lang='es')

def expand_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_libras_re, r'\1 libras', text)
    text = re.sub(_euros_re, r'\1 euros', text)
    text = re.sub(_pesos_re, _expand_pesos, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

