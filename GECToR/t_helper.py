# HELPER file

import random
import pickle  # => pickletools
import string
import os

import pkg_resources
from symspellpy import SymSpell, Verbosity

#  import jamspell

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)  # default values
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# extend dictionary
# all GECToR labels  TODO: problematic labels? => include ..
with open("data/output_vocabulary/labels.txt") as file:
    gector_labels = file.readlines()
# gector_label_count = 0
for label in gector_labels:
    if label.startswith("$APPEND") or label.startswith("$REPLACE"):
        word = label.strip().split("_")[-1]
        if word not in sym_spell.words:
            sym_spell.create_dictionary_entry(word, 1)
            # gector_label_count += 1
# print(f"{gector_label_count} labels added")
# all GECTOR verb forms
with open("data/verb-form-vocab.txt") as file:
    gector_verb_forms = file.readlines()
# verb_form_count = 0
different_verb_forms = set()
for verb_form in gector_verb_forms:
    v1, v2 = verb_form.strip().split(":")[0].split("_")
    different_verb_forms.add(v1)
    different_verb_forms.add(v2)
for verb_form in different_verb_forms:
    if verb_form not in sym_spell.words:
        sym_spell.create_dictionary_entry(verb_form, 1)
        # verb_form_count += 1
# print(f"{verb_form_count} verb forms added")

#  corrector = jamspell.TSpellCorrector()
#  corrector.LoadLangModel("/home/yoel/Desktop/master thesis/tempora/prototype/z_storage/jamspell/en.bin")


def human_readable_time(in_time):
    """
    seconds → days, hours, minutes, seconds
    """
    in_time = int(in_time)
    seconds_in_one_minute = 60
    seconds_in_one_hour = 60 * seconds_in_one_minute
    seconds_in_one_day = 24 * seconds_in_one_hour

    days = in_time // seconds_in_one_day
    in_time = in_time % seconds_in_one_day
    hours = in_time // seconds_in_one_hour
    in_time = in_time % seconds_in_one_hour
    minutes = in_time // seconds_in_one_minute
    in_time = in_time % seconds_in_one_minute
    seconds = in_time

    # ternary → relating to three
    d = "day" if days == 1 else "days"
    h = "hour" if hours == 1 else "hours"
    m = "minute" if minutes == 1 else "minutes"
    s = "second" if seconds == 1 else "seconds"
    hrt = f"{days} {d}, {hours} {h}, {minutes} {m}, {seconds} {s}"

    return hrt


def read_pickle(name="z_student_dict"):
    with open(name, "rb") as file:
        return pickle.load(file)


def write_pickle(data, name="z_student_dict"):
    with open(name, "wb") as file:
        pickle.dump(data, file)


def read_lines(name, skip_strip_test=False):
    if not os.path.exists(name):
        return []
    with open(name, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip() or skip_strip_test]


def mean(lst):
    return sum(lst) / len(lst) if lst else 0


def mae(l1, l2):
    # mean absolute error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def binomial(n, p):
    count = 0
    for _ in range(n):
        if random.random() <= p:
            count += 1
    return count


def symspell_word_correction(word, max_edit_distance=1):
    best_guess = str(sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=max_edit_distance, transfer_casing=True,
                     ignore_token=f"[{string.punctuation + string.digits}]", include_unknown=True)[0])
    occurrences = []  # => if a comma exists in the word itself
    for index in range(len(best_guess)):
        if best_guess[index] == ",":
            occurrences.append(index)
    pos = occurrences[-2]
    return best_guess[:pos]


def symspell_sentence_correction(sentence, skip_indices=None, end_fix=False):
    skip_indices = [] if skip_indices is None else skip_indices
    tokens = sentence.split()
    if end_fix and tokens[-1][-1] in [".", "!", "?"]:  # ?--
        if len(tokens[-1]) > 1:
            tokens.append(tokens[-1][-1])
            tokens[-2] = tokens[-2][:-1]
    for index in range(len(tokens)):
        if index in skip_indices or tokens[index] in sym_spell.words:
            continue
        else:
            correction = symspell_word_correction(tokens[index], max_edit_distance=1)
            if tokens[index] != correction:  # (not) include "unknown"
                tokens[index] = correction
            else:
                tokens[index] = symspell_word_correction(tokens[index], max_edit_distance=2)  # include unknown
    if end_fix and tokens[-1][-1] in [".", "!", "?"]:  # ?--
        tokens[-2] = "".join([tokens[-2], tokens[-1]])
        del tokens[-1]
    return " ".join(tokens)


"""
def jamspell_correction(word_or_sentence):
    return corrector.FixFragment(word_or_sentence)
"""


def get_g_transformations():
    with open("data/output_vocabulary/g_labels.txt") as f:
        lines = f.readlines()
        g_transformations = {}
    for line in lines:
        g_transformations[line.rstrip()] = len(g_transformations)
    return g_transformations


def is_append(s): return s.startswith("$APPEND")


def is_replace(s): return s.startswith("$REPLACE")


def is_delete(s): return s == "$DELETE"


def is_merge_space(s): return s == "$MERGE_SPACE"


def is_merge_hyphen(s): return s == "$MERGE_HYPHEN"


def is_split_hyphen(s): return s == "$TRANSFORM_SPLIT_HYPHEN"
