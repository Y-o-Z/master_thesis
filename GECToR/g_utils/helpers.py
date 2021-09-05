"""
Copyright (c) 2020 Grammarly - GECToR (https://github.com/grammarly/gector)  [adapted by Yoel Zweig]
"""

import os
import pickle
# OO filesystem paths
from pathlib import Path

# Path(__file__) -> current file
VOCAB_DIR = Path(__file__).resolve().parent.parent / "data"
# e.g. => PosixPath('/home/yoel/Desktop/master thesis/tempora/prototype/data')
START_TOKEN = "$START"
UNK = "@@UNKNOWN@@"
PAD = "@@PADDING@@"
SEQ_DELIMITERS = {"tokens": " ", "labels": "SEPL|||SEPR", "operations": "SEPL__SEPR"}


def get_verb_form_dicts():
    path_to_dict = os.path.join(VOCAB_DIR, "verb-form-vocab.txt")
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as file:
        for line in file:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"  # format, remove leading, trailing whitespace
            if decode_key not in decode:
                encode[words] = tags.strip()
                decode[decode_key] = word2
    return encode, decode


ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()


def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)


def encode_verb_form(original_word, corrected_word):
    decoding_request = original_word + "_" + corrected_word
    decoding_response = ENCODE_VERB_DICT.get(decoding_request, "")
    if original_word and decoding_response:
        answer = decoding_response
    else:
        answer = None
    return answer


def get_target_sent_by_edits(source_tokens, edits):
    # [:] (or .copy()) for a (shallow) copy
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] if len(target_tokens) > target_pos >= 0 else ''
        if label == "":  # remove token
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:  # append token
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]  # insert word
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)


def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens

    target_line = " ".join(tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()


def convert_using_case(token, smart_action):
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):  # not included [!!!]
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"  #
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


def convert_using_plural(token, smart_action):
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):  # nothing reverse about it ..
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":
            return source_token
        # deal with case
        if transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        if transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        if transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        if transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # else raise exception
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token


def read_parallel_lines(fn1, fn2):
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    assert len(lines1) == len(lines2)
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    return out_lines1, out_lines2


def read_lines(fn, skip_strip=False):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]


def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def g_write_pickle(data, name, folder=""):
    with open(folder + name, "wb") as file:
        pickle.dump(data, file)


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
