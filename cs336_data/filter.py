import gzip
import random

import fasttext
import re

from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
import nltk
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


LID_MODEL = fasttext.load_model("local-shared-data/classifiers/lid.176.bin")
NSFW_MODEL = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
TOXIC_MODEL = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
QUALITY_CLASSIFIER = fasttext.load_model("local-shared-data/classifiers/quality_classifier_unpreprocessed.bin")
nltk.download("punkt_tab", quiet=True)


def extract_text(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    try: 
        return extract_plain_text(html_bytes.decode(encoding))
    except:
        return ""


def lang_identifier(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")
    predictions = LID_MODEL.predict(text, k=1)
    label = predictions[0][0]
    confidence = predictions[1][0]
    label = label.replace("__label__", "")
    return (label, confidence)


def is_english(text: str) -> bool:
    label, confidence = lang_identifier(text)
    return label == "en" and confidence >= 0.7


def mask_emails(text: str) -> tuple[str, int]:
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"
    email_count = len(re.findall(pattern, text))
    emails_hid_text = re.sub(pattern, "|||EMAIL_ADDRESS|||", text)
    return (emails_hid_text, email_count)


def mask_phones(text: str) -> tuple[str, int]:
    pattern = r"(?:\+?([0-9]{1,4})[-. ]?)?\(?([0-9]{2,4})\)?[-. ]?([0-9]{3,4})[-. ]?([0-9]{3,6})"
    phone_count = len(re.findall(pattern, text))
    phone_hid_text = re.sub(pattern, "|||PHONE_NUMBER|||", text)
    return (phone_hid_text, phone_count)


def mask_ips(text: str) -> tuple[str, int]:
    pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    ip_count = len(re.findall(pattern, text))
    ip_hid_text = re.sub(pattern, "|||IP_ADDRESS|||", text)
    return (ip_hid_text, ip_count)


def classify_nsfw(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")
    predictions = NSFW_MODEL.predict(text, k=1)
    label = predictions[0][0]
    confidence = predictions[1][0]
    label = label.replace("__label__", "")
    return (label, confidence)


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")
    predictions = TOXIC_MODEL.predict(text, k=1)
    label = predictions[0][0]
    confidence = predictions[1][0]
    label = label.replace("__label__", "")
    return (label, confidence)


def is_toxic_or_nsfw(text: str, threshold: float = 0.3) -> bool:
    nsfw_label, nsfw_conf = classify_nsfw(text)
    tox_label, tox_conf = classify_toxic_speech(text)
    if nsfw_label == "nsfw" and nsfw_conf >= threshold:
        return True
    if tox_label == "toxic" and tox_conf >= threshold:
        return True
    return False


def gopher_quality_filter(text: str) -> bool:
    words = nltk.word_tokenize(text)
    
    if len(words) < 50 or len(words) > 100_000:
        return False
    
    total_chars = 0
    for word in words:
        total_chars += len(word)
    if total_chars / len(words) < 3 or total_chars / len(words) > 10:
        return False

    ellipsis_count = 0
    lines = text.splitlines()
    for line in lines:
        if line.endswith("..."):
            ellipsis_count += 1
    if ellipsis_count / len(lines) >= 0.3:
        return False

    alpha_word = 0
    for word in words:
        has_alpha = any(c.isalpha() for c in word)
        if has_alpha: alpha_word += 1
    if alpha_word / len(words) < 0.8:
        return False

    return True


def classify_quality(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")
    predictions = QUALITY_CLASSIFIER.predict(text, k=1)
    label = predictions[0][0]
    confidence = predictions[1][0]
    label = label.replace("__label__", "")
    if label == "negative":
        label = "cc"
    if label == "positive":
        label = "wiki"
    return (label, confidence)


if __name__ == "__main__":
    ...
