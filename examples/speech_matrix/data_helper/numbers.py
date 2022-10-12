"""
Modified from https://github.com/keithito/tacotron
"""


from functools import partial
import inflect
import re
from num2words import num2words


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_space_number_re = re.compile(r"([0-9][0-9 ]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_percent_re = re.compile(r"([0-9\.\,]*[0-9]+[ ]*\%)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = {
    "en": re.compile(r"[0-9]+(st|nd|rd|th)"),
    "es": re.compile(r"[0-9]+(ero|ro|do|to)"),
    "fr": re.compile(r"[0-9]+(er|ème)"),
}
_number_re = re.compile(r"[0-9]+")

_cent_lang = {
    "en": "cent",
    "es": "centavo",
    "fr": "cent",
    "de": "cent",
    "it": "cent",
    "nl": "cent",
    "pl": "cent",
    "pt": "centavo",
    "ro": "cent",
    "hu": "cent",
    "hr": "cent",
    "cs": "cent",
    "fi": "senttiä",
    "sk": "cent",
    "sl": "cent",
    "lt": "cent",
}
_cents_lang = {
    "en": "cents",
    "es": "centavos",
    "fr": "centimes",
    "de": "cent",
    "it": "centesimi",
    "nl": "centen",
    "pl": "centy",
    "pt": "centavos",
    "ro": "cenți",
    "hu": "cent",
    "hr": "centi",
    "cs": "centů",
    "fi": "senttiä",
    "sk": "centov",
    "sl": "centov",
    "lt": "centov",
}
_dollar_lang = {
    "en": "dollar",
    "es": "dólar",
    "fr": "dollar",
    "de": "dollar",
    "it": "dollaro",
    "nl": "dollar",
    "pl": "dolar",
    "pt": "dólar",
    "ro": "dolar",
    "hu": "dollárt",
    "hr": "dolara",
    "cs": "dolarů",
    "fi": "dollaria",
    "sk": "dolárov",
    "sl": "dolarjev",
    "lt": "dolarjev",
}
_dollars_lang = {
    "en": "dollars",
    "es": "dolares",
    "fr": "dollars",
    "de": "dollar",
    "it": "dollari",
    "nl": "dollar",
    "pl": "dolarów",
    "pt": "dólares",
    "ro": "dolari",
    "hu": "dollárt",
    "hr": "dolara",
    "cs": "dolarů",
    "fi": "dollaria",
    "sk": "dolárov",
    "sl": "dolarjev",
    "lt": "dolarjev",
}
_percent_lang = {
    "en": "percent",
    "es": "por ciento",
    "fr": "pour cent",
    "de": "prozent",
    "it": "per cento",
    "nl": "procent",
    "pl": "procent",
    "pt": "por cento",
    "ro": "la sută",
    "hu": "százalék",
    "hr": "postotak",
    "cs": "procento",
    "fi": "prosentteina",
    "sk": "percentá",
    "sl": "odstotek",
    "lt": "odstotek",
}
_pound_lang = {
    "en": "pounds",
    "es": "libras",
    "fr": "livres",
    "de": "pfund",
    "it": "libbre",
    "nl": "ponden",
    "pl": "funty",
    "pt": "libras",
    "ro": "lire sterline",
    "hu": "fontot",
    "hr": "funti",
    "cs": "liber",
    "fi": "puntaa",
    "sk": "libier",
    "sl": "funtov",
    "lt": "funtov",
}
_point_lang = {
    "en": "point",
    "es": "punto",
    "fr": "virgule",
    "de": "punkt",
    "it": "punto",
    "nl": "punt",
    "pl": "punkt",
    "pt": "apontar",
    "ro": "punct",
    "hu": "pont",
    "hr": "točka",
    "cs": "směřovat",
    "fi": "kohta",
    "sk": "bod",
    "sl": "točka",
    "lt": "točka",
}


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _remove_space(m):
    clean_m = m.group(1).replace(" ", "")
    if len(clean_m) < len(m.group(1)) - 3:  # a sequence of digits, not like "20 000"
        return m.group(1)
    else:
        return clean_m


def _expand_percent(lang, m):
    return (
        m.group(0)
        .replace(" ", "")
        .replace("%", f" {_percent_lang[lang]}")
        .replace(",", ".")
    )


def _expand_decimal_point(lang, m):
    pid = m.group(1).find(".")
    d_str = []
    for d in m.group(1)[pid + 1 :]:
        d_str.append(num2words(d, lang=lang))
    d_str = " ".join(d_str)
    return m.group(1).replace(m.group(1)[pid:], f" {_point_lang[lang]} {d_str}")


def _expand_dollars(lang, m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + f" {_dollars_lang[lang]}"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = _dollar_lang[lang] if dollars == 1 else _dollars_lang[lang]
        cent_unit = _cent_lang[lang] if cents == 1 else _cents_lang[lang]
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = _dollars_lang[lang] if dollars == 1 else _dollars_lang[lang]
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = _cent_lang[lang] if cents == 1 else _cents_lang[lang]
        return "%s %s" % (cents, cent_unit)
    else:
        return f"{num2words(0, lang=lang)} {_dollars_lang[lang]}"


def _expand_ordinal(lang, m):
    if lang == "en":
        return _inflect.number_to_words(m.group(0))
    else:
        return num2words(m.group(0)[: -len(m.group(1))], to="ordinal", lang=lang)


def _expand_number(lang, m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return num2words(num, lang=lang)
        elif num > 2000 and num < 2010:
            if lang == "en":
                return "two thousand " + _inflect.number_to_words(num % 100)
            else:
                return (
                    num2words(2000, lang=lang) + " " + num2words(num % 100, lang=lang)
                )
        elif num % 100 == 0:
            if lang == "en":
                return _inflect.number_to_words(num // 100) + " hundred"
            else:
                return num2words(num, lang=lang).replace(", ", " ")
        else:
            if lang == "en":
                return _inflect.number_to_words(
                    num, andword="", zero="oh", group=2
                ).replace(", ", " ")
            else:
                return num2words(num, lang=lang).replace(", ", " ")
    else:
        if lang == "en":
            return _inflect.number_to_words(num, andword="")
        else:
            return num2words(num, lang=lang)


def normalize_numbers(text, lang="en"):
    text = re.sub(_space_number_re, _remove_space, text)
    text = re.sub(_percent_re, partial(_expand_percent, lang), text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, rf"\1 {_pound_lang[lang]}", text)
    text = re.sub(_dollars_re, partial(_expand_dollars, lang), text)
    if lang not in set(["it", "cs", "sk"]):
        text = re.sub(_decimal_number_re, partial(_expand_decimal_point, lang), text)
    if lang in _ordinal_re:
        text = re.sub(_ordinal_re[lang], partial(_expand_ordinal, lang), text)
    text = re.sub(_number_re, partial(_expand_number, lang), text)
    return text
