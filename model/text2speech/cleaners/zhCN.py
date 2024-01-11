import re


# TODO 숫자 0의 경우는 항상 '떨어질 령' 을 기반으로 써야함. '떨어질 영' 과 모양은 같으나, 다른 유니코드를 가짐.
digit_dict = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}
phone_num_to_zh = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}

unit_to_zh1 = {
    # '%': '퍼센트', TODO 백분지 삼십 과 같은 방식으로 읽음.
    'cm': '厘米',
    'mm': '毫米',
    'km': '公里',
    'kg': '公斤',
}

unit_to_zh2 = {
    'm': '米',
}


number_checker = "([\+-] ?)?(\d[\d,]*[\.]?\d*)"
etc_metrics_checker = " *(%|kg|km|cm|mm|m)"

num_to_zh1 = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
num_to_zh10 = ['', '十', '百', '千']
num_to_zh10000 = ['', '万', '億', '兆', '京', '垓']
# 1천 이상에서는 2를 两으로 읽음


def sub_manual(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text


def number_to_chinese(num_str: re.Match,
                      is_phone: bool = False,
                      is_date: bool = False,
                      is_metrics: bool = False,
                      is_fraction: bool = False):
    if is_phone:
        return sub_manual(num_str.group(1), phone_num_to_zh) + ' '\
            + sub_manual(num_str.group(2), phone_num_to_zh) + ' '\
            + sub_manual(num_str.group(3), phone_num_to_zh)
    elif is_fraction:
        # NOTE basically, one number_checker has two groups (+_ ) and (number)
        return re.sub('^.*$', lambda x: number_to_chinese(x), num_str.group(4)) + '分之 '\
            + re.sub('^.*$', lambda x: number_to_chinese(x), num_str.group(2))
    elif is_date:
        return re.sub('^.*$', lambda x: number_to_chinese(x), num_str.group(1)) + '年 '\
            + re.sub('^.*$', lambda x: number_to_chinese(x), num_str.group(2)) + '月 '\
            + re.sub('^.*$', lambda x: number_to_chinese(x),
                     num_str.group(3)) + '号'
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

    unit_str = sub_manual(unit_str, unit_to_zh1)
    unit_str = sub_manual(unit_str, unit_to_zh2)

    num = float(num_str)
    if num >= 100 and unit_str != "":
        if unit_str.strip() == '%':
            return '百分之' + num_str
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
        if unit_str.strip() == '%':
            return '百分之' + num_str
        return num_str + unit_str

    # Delete '-' character when replace digit
    digit = int(digit_str)
    if digit_str.startswith("-"):
        digit, digit_str = abs(digit), str(abs(digit))

    # Change each digit to its corresponding Hangul
    if digit >= 1e24:  # higher than '해'
        zh_chars = re.sub('\d', lambda x: digit_dict[x.group()], digit_str)
    else:
        zh_chars = ""
        size = len(str(digit))
        char_temp = []
        for idx, digit_number in enumerate(digit_str, start=1):
            digit_number = int(digit_number)

            if digit_number != 0:
                if digit_number != 1 or size == idx or idx == 1:
                    if size-idx > 2 and digit_number == 2:
                        char_temp += '两'
                    else:
                        char_temp += num_to_zh1[digit_number]
                char_temp += num_to_zh10[(size - idx) % 4]
            else:
                char_temp += "零"

            if (size - idx) % 4 == 0 and len(char_temp) != 0:
                zh_chars += "".join(char_temp)
                zh_chars += num_to_zh10000[int((size - idx) / 4)]
                char_temp = []

    zh_chars = re.sub("零+", "零", zh_chars)
    zh_chars = re.sub("零$", "", zh_chars)

    # May not read the first letter
    if zh_chars.startswith("一") and len(zh_chars) > 1:  # and digit >= 20000:
        zh_chars = zh_chars[1:]

    # Supplementing cases where certain numbers or mathematical symbols appear
    if digit == 0:
        zh_chars += "零"
    if float_str is not None and float_str != '':
        zh_chars += "点 "
        zh_chars += re.sub('\d', lambda x: digit_dict[x.group()], float_str)
    if num_str.startswith("+"):
        zh_chars = "加 " + zh_chars
    elif num_str.startswith("-"):
        zh_chars = "减 " + zh_chars

    if unit_str.strip() == '%':
        return '百分之' + zh_chars
    return zh_chars + unit_str


def expand_numbers(text):
    # Phone number
    text = expand_telephone(text)
    # Date
    text = expand_date(text)
    # Fraction
    text = expand_fraction(text)
    # Count number
    # text = re.sub(number_checker + count_checker,
    #        lambda x: number_to_chinese(x, is_count=True, is_metrics=True), text)
    # Other english metrics number
    # Only number
    text = expand_cardinal(text)
    return text


def expand_cardinal(text):
    # Other english metrics number
    text = re.sub(number_checker + etc_metrics_checker,
                  lambda x: number_to_chinese(x, is_metrics=True), text)
    # Only number
    text = re.sub(number_checker,
                  lambda x: number_to_chinese(x), text)
    return text

# def expand_ordinal(text):
#    ## Count number
#    text = re.sub(number_checker,
#            lambda x: number_to_chinese(x, is_count=True, is_metrics=True), text)
#    return text


def expand_fraction(text):
    # Fraction
    text = re.sub(number_checker + '/' + number_checker,
                  lambda x: number_to_chinese(x, is_fraction=True), text)
    return text


def expand_date(text, date_seps=['\.', '/', '-']):
    # Date
    for date_sep in date_seps:
        text = re.sub('(\d\d\d\d){}(\d?\d){}(\d?\d)'.format(date_sep, date_sep),
                      lambda x: number_to_chinese(x, is_date=True), text)
    return text


def expand_telephone(text):
    # Phone number
    text = re.sub('(\d?\d?\d?\d)-(\d?\d\d\d)-(\d\d\d\d)',
                  lambda x: number_to_chinese(x, is_phone=True), text)
    return text


def normalize_manual(text):
    return text
