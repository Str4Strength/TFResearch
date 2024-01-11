import re

digit_dict = {
    '0': 'ゼロ', '1': '一', '2': '二', '3': '三', '4': 'よん',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}
phone_num_to_ja = {
    '0': 'ゼロ', '1': '一', '2': '二', '3': '三', '4': 'よん',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}

unit_to_ja1 = {
    '%': 'パーセント',
    'cm': 'センチメ-トル',
    'mm': 'ミリメ-トル',
    'km': 'キロメ-トル',
    'kg': 'キログラム',
}

unit_to_ja2 = {
    'm': 'メ-トル',
}

etc_dictionary = {
}

number_checker = "([\+-] ?)?(\d[\d,]*[\.]?\d*)"
etc_metrics_checker = " *(%|kg|km|cm|mm|m)"

num_to_ja1 = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
num_to_ja10 = ['', '十', '百', '千']
num_to_ja10000 = ['', '万', '億', '兆', '京', '垓']
# NOTE 물건을 셀 때는 발음이 많이 달라지지만, pykakasi에서 일부는 해결해줌.


def sub_manual(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text


def number_to_japanese(num_str, is_phone=False, is_date=False, is_metrics=False, is_fraction=False):
    if is_phone:
        return sub_manual(num_str.group(1), phone_num_to_ja) + ' '\
            + sub_manual(num_str.group(2), phone_num_to_ja) + ' '\
            + sub_manual(num_str.group(3), phone_num_to_ja)
    elif is_fraction:
        # NOTE basically, one number_checker has two groups (+_ ) and (number)
        return re.sub('^.*$', lambda x: number_to_japanese(x), num_str.group(4)) + '分の '\
            + re.sub('^.*$', lambda x: number_to_japanese(x), num_str.group(2))
    elif is_date:
        return re.sub('^.*$', lambda x: number_to_japanese(x), num_str.group(1)) + '年 '\
            + re.sub('^.*$', lambda x: number_to_japanese(x), num_str.group(2)) + '月 '\
            + re.sub('^.*$', lambda x: number_to_japanese(x),
                     num_str.group(3)) + '日'
    elif is_metrics:
        if num_str.group(1) is not None:
            num_str, unit_str = num_str.group(
                1) + num_str.group(2), num_str.group(3)
        else:
            num_str, unit_str = num_str.group(2), num_str.group(3)
    else:
        num_str, unit_str = num_str.group(), ""
    num_str = num_str.replace(',', '')
    num_str = num_str.replace(' ', '')

    unit_str = sub_manual(unit_str, unit_to_ja1)
    unit_str = sub_manual(unit_str, unit_to_ja2)

    num = float(num_str)
    if num >= 100 and unit_str != "":
        return num_str + unit_str

    # Check whether this number is valid float or not
    check_float = num_str.split('.')
    if len(check_float) == 2:
        digit_str, float_str = check_float
    elif len(check_float) == 1:
        digit_str, float_str = check_float[0], None
    else:
        raise Exception(" [!] Wrong number format")

    if digit_str.startswith('0') and float_str is None:
        num_str = sub_manual(num_str, digit_dict)
        return num_str + unit_str

    # Delete '-' character when replace digit
    digit = int(digit_str)
    if digit_str.startswith("-"):
        digit, digit_str = abs(digit), str(abs(digit))

    # Change each digit to its corresponding Hangul
    if digit >= 1e24:  # higher than '해'
        ja_chars = re.sub('\d', lambda x: digit_dict[x.group()], digit_str)
    else:
        ja_chars = ""
        size = len(str(digit))
        char_temp = []
        for idx, digit_number in enumerate(digit_str, start=1):
            digit_number = int(digit_number)

            if digit_number != 0:
                if digit_number != 1 or size == idx or idx == 1:
                    char_temp += num_to_ja1[digit_number]
                char_temp += num_to_ja10[(size - idx) % 4]

            if (size - idx) % 4 == 0 and len(char_temp) != 0:
                ja_chars += "".join(char_temp)
                ja_chars += num_to_ja10000[int((size - idx) / 4)]
                char_temp = []

    # May not read the first letter
    if ja_chars.startswith("一") and len(ja_chars) > 1:  # and digit <= 2000:
        ja_chars = ja_chars[1:]

    # Supplementing cases where certain numbers or mathematical symbols appear
    if digit == 0:
        ja_chars += "ゼロ"
    if float_str is not None and float_str != '':
        ja_chars += "点 "
        ja_chars += re.sub('\d', lambda x: digit_dict[x.group()], float_str)
    if num_str.startswith("+"):
        ja_chars = "たす " + ja_chars
    elif num_str.startswith("-"):
        ja_chars = "ひく " + ja_chars

    return ja_chars + unit_str


def expand_numbers(text):
    # Phone number
    text = expand_telephone(text)
    # Date
    text = expand_date(text)
    # Fraction
    text = expand_fraction(text)
    # Count number
    # text = re.sub(number_checker + count_checker,
    #        lambda x: number_to_japanese(x, is_count=True, is_metrics=True), text)
    # Other english metrics number
    # Only number
    text = expand_cardinal(text)
    return text


def expand_cardinal(text):
    # Other english metrics number
    text = re.sub(number_checker + etc_metrics_checker,
                  lambda x: number_to_japanese(x, is_metrics=True), text)
    # Only number
    text = re.sub(number_checker,
                  lambda x: number_to_japanese(x), text)
    return text

# def expand_ordinal(text):
#    ## Count number
#    text = re.sub(number_checker,
#            lambda x: number_to_japanese(x, is_count=True, is_metrics=True), text)
#    return text


def expand_fraction(text):
    # Fraction
    text = re.sub(number_checker + '/' + number_checker,
                  lambda x: number_to_japanese(x, is_fraction=True), text)
    return text


def expand_date(text, date_seps=['\.', '/', '-']):
    # Date
    for date_sep in date_seps:
        text = re.sub('(\d\d\d\d){}(\d?\d){}(\d?\d)'.format(date_sep, date_sep),
                      lambda x: number_to_japanese(x, is_date=True), text)
    return text


def expand_telephone(text):
    # Phone number
    text = re.sub('(\d?\d?\d?\d)-(\d?\d\d\d)-(\d\d\d\d)',
                  lambda x: number_to_japanese(x, is_phone=True), text)
    return text


def normalize_manual(text):
    text = sub_manual(text, etc_dictionary)
    return text
