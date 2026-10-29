import random
import os
import hashlib
import fasttext
import re
import unicodedata
import nltk
import mmh3
import shutil

from pathlib import Path
from nltk.util import ngrams
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from collections import defaultdict
import numpy as np
from itertools import combinations
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


def exact_deduplicate(input_files: list[os.PathLike], output_dir: os.PathLike):
    line_freqs = defaultdict(int)
    for file in input_files:
        with open(file, "r") as file:
            lines = file.read().splitlines()
            for line in lines:
                line_freqs[hashlib.sha256(line.encode("utf-8")).digest()] += 1
    for file in input_files:
        output_path = output_dir / file.name
        with open(file, "r") as file:
            with open(output_path, "a") as new_file:
                lines = file.read().splitlines()
                for line in lines:
                    if line_freqs[hashlib.sha256(line.encode("utf-8")).digest()] <= 1:
                            new_file.write(line + "\n")


def normalize(text: str) -> str:
    # 1. Lowercase
    text = text.lower()

    # 2. NFD Unicode normalization
    text = unicodedata.normalize("NFD", text)

    # 3. Remove accents (combining marks)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    # 4. Remove punctuation
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("P")
    )

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_minhash(ngram_set: set, num_hashes: int) -> np.array:
    signature = []

    for seed in range(num_hashes):
        hash_values = [
            mmh3.hash(" ".join(gram), seed=seed, signed=False)
            for gram in ngram_set
        ]
        signature.append(min(hash_values))

    return np.array(signature, dtype=np.uint32)


def fuzzy_deduplicate(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams_count: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    # 1. Normalize documents, generate n-grams, and compute MinHash bands
    doc_bands = {}
    doc_ngrams = {}

    for doc_id, file in enumerate(input_files):
        with open(file, "r") as file:
            normalized_text = normalize(file.read())
            words = normalized_text.split()
            ngram_set = set(ngrams(words, n=ngrams_count))

            doc_ngrams[doc_id] = ngram_set

            min_hashes = compute_minhash(ngram_set, num_hashes=num_hashes)
            bands = np.split(min_hashes, num_bands)
            doc_bands[doc_id] = bands

    # 2. Create LSH buckets containing documents with matching bands
    buckets = defaultdict(list)

    for doc_id, bands in doc_bands.items():
        for band_idx, band in enumerate(bands):
            key = (band_idx, tuple(band))
            buckets[key].append(doc_id)

    # 3. Generate candidate pairs from documents sharing LSH buckets
    candidate_pairs = set()

    for doc_ids in buckets.values():
        candidate_pairs.update(combinations(sorted(doc_ids), 2))

    # 4. Verify candidate pairs using exact Jaccard similarity
    duplicate_pairs = []

    for doc_a, doc_b in candidate_pairs:
        a_ngram = doc_ngrams[doc_a]
        b_ngram = doc_ngrams[doc_b]

        union = a_ngram | b_ngram
        similarity = len(a_ngram & b_ngram) / len(union) if union else 0.0

        if similarity >= jaccard_threshold:
            duplicate_pairs.append((doc_a, doc_b))

    # 5. Group connected duplicate pairs into clusters
    clusters = []

    for a, b in duplicate_pairs:
        merged = {a, b}

        i = 0
        while i < len(clusters):
            if merged & clusters[i]:
                merged.update(clusters.pop(i))
                i = 0
            else:
                i += 1

        clusters.append(merged)

    # 6. Randomly retain one document from each duplicate cluster
    kept_ids = set(range(len(input_files)))

    for cluster in clusters:
        selected = random.choice(list(cluster))

        for doc_id in cluster:
            if doc_id != selected:
                kept_ids.remove(doc_id)

    # 7. Write retained documents to the output directory
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for doc_id in kept_ids:
        source = Path(input_files[doc_id])
        shutil.copyfile(source, output_directory / source.name)


if __name__ == "__main__":
    input_files = [
        "tests/fixtures/documents_with_fuzzy_duplicates/pytorch_license.txt",
        "tests/fixtures/documents_with_fuzzy_duplicates/rails_mit_license.txt",
        "tests/fixtures/documents_with_fuzzy_duplicates/react_mit_license.txt",
    ]
    fuzzy_deduplicate(
        input_files,
        num_hashes=10,
        num_bands=5,
        ngrams_count=2,
        jaccard_threshold=0.5,
        output_directory=Path("local-shared-data/")
    )
