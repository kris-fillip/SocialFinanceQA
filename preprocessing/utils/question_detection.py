# Taken from https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
import re

from analysis import get_basic_attribute
from config.constants import QUESTION_HIT_WORDS, ALPHABETS, PREFIXES, SUFFIXES, STARTERS, ACRONYMS, WEBSITES, DIGITS, MULTIPLE_DOTS


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    text = re.sub(DIGITS + "[.]" + DIGITS, "\\1<prd>\\2", text)
    text = re.sub(MULTIPLE_DOTS, lambda match: "<prd>" *
                  len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(ACRONYMS+" "+STARTERS, "\\1<stop> \\2", text)
    text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]" +
                  ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(ALPHABETS + "[.]" + ALPHABETS +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+SUFFIXES+"[.] "+SUFFIXES, " \\1<stop> \\2", text)
    text = re.sub(" "+SUFFIXES+"[.]", " \\1<prd>", text)
    text = re.sub(" " + ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def evaluate_whether_question(obj: dict, attribute: str) -> bool:
    text = get_basic_attribute(obj, attribute)
    text = text.lower()
    # remove xpost content from titles
    text = re.sub("xpost from r/.*?($|\s)", "", text)
    text = re.sub("[\(\[].*?[\)\]]", "", text).strip()
    if len(text) == 0 or len(text.split()) < 4:
        return False
    if text[-1] == "?" or any(word in text for word in QUESTION_HIT_WORDS):
        return True
    if attribute == "selftext":
        sentences = split_into_sentences(text)
        if len(sentences) > 3:
            if sentences[-2][-1] == "?" or sentences[-3][-1] == "?":
                return True
    return False
