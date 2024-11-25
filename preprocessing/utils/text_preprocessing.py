#!/usr/bin/env python3
from nltk.tokenize.casual import _replace_html_entities
import re
import strip_markdown
import unicodedata

from config.constants import REDDIT_USER_RE, HASH_RE, URL_RE, FILTER_ELEMENTS


def preprocess_text(text: str) -> str:
    if text is None:
        text = ""
    text = _replace_html_entities(text)
    text = re.sub(REDDIT_USER_RE, ' ', text)
    text = re.sub(HASH_RE, '', text)
    text = re.sub(URL_RE, ' ', text)
    text = strip_emoji(text)
    text = remove_edits_and_updates_from_text(text)
    text = strip_markdown.strip_markdown(text)
    return text


def strip_emoji(text: str) -> str:
    '''Take out emoji. Returns doc string.
    ::param text:: tweet
    ::type doc:: str
    '''
    text = ''.join(c for c in text if unicodedata.category(c) != 'So')
    return text


def remove_edits_and_updates_from_text(text: str) -> str:
    for el in FILTER_ELEMENTS:
        if el in text:
            text_split = text.split(el)
            if len(text_split[0]) != 0:
                text = text_split[0]
    return text
