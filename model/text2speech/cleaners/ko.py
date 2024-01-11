import re
from jamo import hangul_to_jamo

jamo_leads = "".join([chr(_) for _ in range(0x1100, 0x1113)])
jamo_vowels = "".join([chr(_) for _ in range(0x1161, 0x1176)])
jamo_tails = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

digit_dict = {
    '0': '영', '1': '일', '2': '이', '3': '삼', '4': '사',
    '5': '오', '6': '육', '7': '칠', '8': '팔', '9': '구',
}
phone_num_to_kor = {
    '0': '공', '1': '일', '2': '이', '3': '삼', '4': '사',
    '5': '오', '6': '육', '7': '칠', '8': '팔', '9': '구',
}

unit_to_kor1 = {
    '%': '퍼센트',
    'cm': '센치미터',
    'mm': '밀리미터',
    'km': '킬로미터',
    'kg': '킬로그람',
}

unit_to_kor2 = {
    'm': '미터',
}

abb_to_kor1 = {
    'ㅂ': '바',
    'ㅈ': '지',
    'ㄷ': '덜',
    'ㄱ': '고',
    'ㅅ': '샤',
    'ㅛ': '요',
    'ㅕ': '여',
    'ㅑ': '야',
    'ㅐ': '애',
    'ㅔ': '에',
    'ㅁ': '므',
    'ㄴ': '노',
    'ㅇ': '응',
    'ㄹ': '리',
    'ㅎ': '흐',
    'ㅗ': '오',
    'ㅓ': '어',
    'ㅏ': '아',
    'ㅣ': '이',
    'ㅋ': '크',
    'ㅌ': '티',
    'ㅊ': '축',
    'ㅍ': '팜',
    'ㅠ': '유',
    'ㅜ': '우',
    'ㅡ': '으',
    'ㄳ': '감사',
    'ㅄ': '브스',
    'ㄵ': '',
    'ㄼ': '',
    'ㄺ': '',
    'ㄻ': '',
    'ㄾ': '',
    'ㄿ': '',
}

number_checker = "([\+-] ?)?(\d[\d,]*[\.]?\d*)"
count_checker = " *(시|명|가지|살|마리|포기|송이|톨|통|점|개(?!월)|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판|배|번|차례|장|마디|군데|발|곳|냥|푼|돈|말|되|채|필|알|판|쌍)"
etc_metrics_checker = " *(%|kg|km|cm|mm|m)"

num_to_kor1 = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
count_to_kor1 = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
num_to_kor10 = ['', '십', '백', '천']
num_to_kor10000 = ['', '만', '억', '조', '경', '해']

count_tenth_dict = {
    "십": "열", "두십": "스무", "세십": "서른", "네십": "마흔", "다섯십": "쉰",
    "여섯십": "예순", "일곱십": "일흔", "여덟십": "여든", "아홉십": "아흔",
}


def sub_manual(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text


def number_to_korean(num_str, is_phone=False, is_date=False, is_count=False, is_metrics=False, is_fraction=False):
    if is_phone:
        return sub_manual(num_str.group(1), phone_num_to_kor) + ' '\
            + sub_manual(num_str.group(2), phone_num_to_kor) + ' '\
            + sub_manual(num_str.group(3), phone_num_to_kor)
    elif is_fraction:
        # NOTE basically, one number_checker has two groups (+_ ) and (number)
        return re.sub('^.*$', lambda x: number_to_korean(x), num_str.group(4)) + '분의 '\
            + re.sub('^.*$', lambda x: number_to_korean(x), num_str.group(2))
    elif is_date:
        return re.sub('^.*$', lambda x: number_to_korean(x), num_str.group(1)) + '년 '\
            + re.sub('^.*$', lambda x: number_to_korean(x), num_str.group(2)) + '월 '\
            + re.sub('^.*$', lambda x: number_to_korean(x),
                     num_str.group(3)) + '일'
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

    unit_str = sub_manual(unit_str, unit_to_kor1)
    unit_str = sub_manual(unit_str, unit_to_kor2)

    num = float(num_str)
    if num >= 100 and unit_str != "":
        return num_str + unit_str

    # Check whether this number is valid float or not
    check_float = num_str.split('.')
    if len(check_float) == 2:
        digit_str, float_str = check_float
    elif len(check_float) >= 3:
        raise Exception(" [!] Wrong number format")
    else:
        digit_str, float_str = check_float[0], None

    if digit_str.startswith('0') and float_str is None:
        num_str = sub_manual(num_str, digit_dict)
        return num_str + unit_str

    # ex) 3.14개, 6.53시
    if is_count and float_str is not None:
        is_count = False
        # raise Exception(" [!] `is_count` and float number does not fit each other")

    # Delete '-' character when replace digit
    digit = int(digit_str)
    if digit_str.startswith("-"):
        digit, digit_str = abs(digit), str(abs(digit))

    # Change each digit to its corresponding Hangul
    size = len(str(digit))
    if digit >= 1e24:  # higher than '해'
        kor_chars = re.sub('\d', lambda x: digit_dict[x.group()], digit_str)
    else:
        kor_chars = ""
        char_temp = []
        for idx, digit_number in enumerate(digit_str, start=1):
            digit_number = int(digit_number)

            if digit_number != 0:
                if is_count and (digit_number != 1 or size == idx or idx == 1) and (size-idx < 2):
                    char_temp += count_to_kor1[digit_number]
                elif digit_number != 1 or size == idx or idx == 1:
                    char_temp += num_to_kor1[digit_number]
                char_temp += num_to_kor10[(size - idx) % 4]

            if (size - idx) % 4 == 0 and len(char_temp) != 0:
                kor_chars += "".join(char_temp)
                kor_chars += num_to_kor10000[int((size - idx) / 4)]
                char_temp = []

    # May not read the first letter
    if is_count:
        if kor_chars.startswith("한") and len(kor_chars) > 1:
            kor_chars = kor_chars[1:]

        if any(word in kor_chars for word in count_tenth_dict):
            kor_chars = re.sub(
                '|'.join(count_tenth_dict.keys()),
                lambda x: count_tenth_dict[x.group()], kor_chars)
    elif kor_chars.startswith("일") and len(kor_chars) > 1:
        kor_chars = kor_chars[1:]

    # Supplementing cases where certain numbers or mathematical symbols appear
    if digit == 0:
        kor_chars += "영"
    if float_str is not None and float_str != '':
        kor_chars += "쩜 "
        kor_chars += re.sub('\d', lambda x: digit_dict[x.group()], float_str)
    if num_str.startswith("+"):
        kor_chars = "플러스 " + kor_chars
    elif num_str.startswith("-"):
        kor_chars = "마이너스 " + kor_chars

    return kor_chars + unit_str


def expand_numbers(text):
    # Phone number
    text = expand_telephone(text)
    # Date
    text = expand_date(text)
    # Fraction
    text = expand_fraction(text)
    # Count number
    text = re.sub(number_checker + count_checker,
                  lambda x: number_to_korean(x, is_count=True, is_metrics=True), text)
    # Other english metrics number
    # Only number
    text = expand_cardinal(text)
    return text


def expand_cardinal(text):
    # Other english metrics number
    text = re.sub(number_checker + etc_metrics_checker,
                  lambda x: number_to_korean(x, is_metrics=True), text)
    # Only number
    text = re.sub(number_checker,
                  lambda x: number_to_korean(x), text)
    return text


def expand_ordinal(text):
    # Count number
    text = re.sub(number_checker,
                  lambda x: number_to_korean(x, is_count=True, is_metrics=True), text)
    return text


def expand_fraction(text):
    # Fraction
    text = re.sub(number_checker + '/' + number_checker,
                  lambda x: number_to_korean(x, is_fraction=True), text)
    return text


def expand_date(text, date_seps=['\.', '/', '-']):
    # Date
    for date_sep in date_seps:
        text = re.sub('(\d\d\d\d){}(\d?\d){}(\d?\d)'.format(date_sep, date_sep),
                      lambda x: number_to_korean(x, is_date=True), text)
    return text


def expand_telephone(text):
    # Phone number
    text = re.sub('(\d?\d?\d?\d)-(\d?\d\d\d)-(\d\d\d\d)',
                  lambda x: number_to_korean(x, is_phone=True), text)
    return text


def normalize_manual(text):
    text = sub_manual(text, abb_to_kor1)
    return text
