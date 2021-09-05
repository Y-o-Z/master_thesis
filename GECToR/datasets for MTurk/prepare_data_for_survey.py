import random

from t_helper import read_pickle, write_pickle, get_g_transformations, symspell_sentence_correction


RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def test_sentences(sentences, is_triple=False):
    if not is_triple:
        for n, s in enumerate(sentences):
            print(str(n) + ": ", s)
            _ = input()
    else:
        for n, s in enumerate(sentences):
            print(str(n) + ": ", s[0])
            print(str(n) + ": ", s[1])
            print(str(n) + ": ", s[2])
            print("*" * 30)
            _ = input()


def choose_sentences(sentences):
    chosen_sentences = []
    chosen_indices = []
    while len(chosen_sentences) < 5:
        index = random.randint(0, len(sentences) - 1)
        if index in chosen_indices:
            continue
        print(sentences[index][2])
        is_not_0 = int(input())  # 0 => continue, else "don't"
        if is_not_0:
            chosen_sentences.append(sentences[index])
            chosen_indices.append(index)
    return chosen_sentences, chosen_indices


def remove_spaces(sentence):  # n't, 's, punctuation
    index = 0
    while index < len(sentence):
        if sentence[index] == "'" and index > 1:
            if sentence[index - 2:index + 3] == " n't ":
                sentence = sentence[:index - 2] + sentence[index - 1:]
            elif sentence[index - 1:index + 3] == " 's ":
                sentence = sentence[:index - 1] + sentence[index:]
            else:
                index += 1
        elif sentence[index] in [".", ",", "!"] and index > 0 and sentence[index - 1] == " ":
            sentence = sentence[:index - 1] + sentence[index:]
        else:
            index += 1
    return sentence


def parse_sentences_from_file(filename):
    sentences_by_class = []
    for index in range(29):
        list_of_triplets = read_pickle(filename + "_" + str(index))
        class_sentences = []
        for triplet in list_of_triplets:
            # from t_student_session
            i1, j1 = triplet[2]["preds_indices"]
            i2, j2 = triplet[2]["one_error_preds_indices"]
            # symspellpy
            # triplet[0] = symspell_sentence_correction(triplet[0], skip_indices=list(range(i1, j1 + 1)))
            # triplet[1] = symspell_sentence_correction(triplet[1], skip_indices=list(range(i2, j2 + 1)))
            correct_sentence_tokens = triplet[0].split()
            error_sentence_tokens = triplet[1].split()
            feedback = ""
            feedback += " ".join(correct_sentence_tokens[:i1])
            space = " " if i2 != 0 else ""
            feedback += f"{space}{RED}{{"  # {{ → escape {
            feedback += " ".join(error_sentence_tokens[i2:j2 + 1])
            feedback += f"}}{RESET} → {BLUE}{{"
            feedback += " ".join(correct_sentence_tokens[i1:j1 + 1])
            feedback += f"}}{RESET} "
            feedback += " ".join(correct_sentence_tokens[j1 + 1:])
            correct_sentence_tokens = " ".join(correct_sentence_tokens)
            error_sentence_tokens = " ".join(error_sentence_tokens)
            class_sentences.append([correct_sentence_tokens, error_sentence_tokens, feedback])
        sentences_by_class.append(class_sentences)
    return sentences_by_class


"""
conll = parse_sentences_from_file("datasets for MTurk/conll_by_error_class/list_conll")
fce = parse_sentences_from_file("datasets for MTurk/fce_by_error_class/list_fce")

sentences_by_class = []
total_number_of_sentences = 0  # 1157
for c in range(29):
    sentences_by_class.append(conll[c])
    sentences_by_class[c].extend(fce[c])
    total_number_of_sentences += len(sentences_by_class[c])

# n't, 's, punctuation
for c in range(29):
    for s in range(len(sentences_by_class[c])):
        sentences_by_class[c][s][0] = remove_spaces(sentences_by_class[c][s][0])
        sentences_by_class[c][s][1] = remove_spaces(sentences_by_class[c][s][1])
        sentences_by_class[c][s][2] = remove_spaces(sentences_by_class[c][s][2])

g = get_g_transformations()

for i, class_sentences in enumerate(sentences_by_class):
    print(list(g)[i] + ":", len(class_sentences))

write_pickle(sentences_by_class, "datasets for MTurk/error_types_combined_data/sentences_by_class")
"""
sentences_by_class = read_pickle("datasets for MTurk/error_types_combined_data/sentences_by_class")

# 65 "good" sentences per class for our survey (5 + 12 * 5)
num_per_class = 65

# CONLL-2014
# ERROR TYPE 1 ("casing upper")  # + check?  (+ balance)
casing_errors = sentences_by_class[0]
keep_indices = []
for index in range(len(casing_errors)):
    if casing_errors[index][1].count(".") + casing_errors[index][1].count("!") == 1 and \
            not casing_errors[index][2].startswith("\x1b"):
        keep_indices.append(index)
casing_errors = [triplet for i, triplet in enumerate(casing_errors) if i in keep_indices]
# + ERROR TYPE 1 ("casing lower")  # + check?
casing_2_errors = sentences_by_class[2]
random.shuffle(casing_2_errors)
casing_2_errors = casing_2_errors[:num_per_class - len(casing_errors)]
casing_errors.extend(casing_2_errors)
random.shuffle(casing_errors)

casing_errors_test = [casing_errors[index] for index in range(5)]  # check!
casing_errors_train = [casing_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(casing_errors_test, "datasets for MTurk/error_types_combined_data/casing_errors_test")
# write_pickle(casing_errors_train, "datasets for MTurk/error_types_combined_data/casing_errors_train")

# ERROR TYPE 2 ("singular")  # + check?
singular_errors = sentences_by_class[7]
random.shuffle(singular_errors)
singular_errors = singular_errors[:num_per_class]

singular_errors_test = [singular_errors[index] for index in range(5)]  # check!
singular_errors_train = [singular_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(singular_errors_test, "datasets for MTurk/error_types_combined_data/singular_errors_test")
# write_pickle(singular_errors_train, "datasets for MTurk/error_types_combined_data/singular_errors_train")

# ERROR TYPE 3 ("plural")  # + check?
plural_errors = sentences_by_class[8]
random.shuffle(plural_errors)
plural_errors = plural_errors[:num_per_class]

plural_errors_test = [plural_errors[index] for index in range(5)]  # check!
plural_errors_train = [plural_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(plural_errors_test, "datasets for MTurk/error_types_combined_data/plural_errors_test")
# write_pickle(plural_errors_train, "datasets for MTurk/error_types_combined_data/plural_errors_train")

# ERROR TYPE 4 ("vb => vbz")  # + check?
vb_vbz_errors = sentences_by_class[9]
random.shuffle(vb_vbz_errors)
vb_vbz_errors = vb_vbz_errors[:num_per_class]

vb_vbz_errors_test = [vb_vbz_errors[index] for index in range(5)]  # check!
vb_vbz_errors_train = [vb_vbz_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(vb_vbz_errors_test, "datasets for MTurk/error_types_combined_data/vb_vbz_errors_test")
# write_pickle(vb_vbz_errors_train, "datasets for MTurk/error_types_combined_data/vb_vbz_errors_train")

# ERROR TYPE 5 ("vbz => vb")  # + check?
vbz_vb_errors = sentences_by_class[13]
random.shuffle(vbz_vb_errors)
vbz_vb_errors = vbz_vb_errors[:num_per_class]

vbz_vb_errors_test = [vbz_vb_errors[index] for index in range(5)]  # check!
vbz_vb_errors_train = [vbz_vb_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(vbz_vb_errors_test, "datasets for MTurk/error_types_combined_data/vbz_vb_errors_test")
# write_pickle(vbz_vb_errors_train, "datasets for MTurk/error_types_combined_data/vbz_vb_errors_train")

# ERROR TYPE 6 ("vb <=> vbn")  # + check?
vb_vbn_errors = []
for c in [10, 17]:
    vb_vbn_errors.extend(sentences_by_class[c])
random.shuffle(vb_vbn_errors)
vb_vbn_errors = vb_vbn_errors[:num_per_class]

vb_vbn_errors_test = [vb_vbn_errors[index] for index in range(5)]  # check!
vb_vbn_errors_train = [vb_vbn_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(vb_vbn_errors_test, "datasets for MTurk/error_types_combined_data/vb_vbn_errors_test")
# write_pickle(vb_vbn_errors_train, "datasets for MTurk/error_types_combined_data/vb_vbn_errors_train")

# ERROR TYPE 7 ("vb <=> vbg") # + check?
vb_vbg_errors = []
for c in [12, 25]:
    vb_vbg_errors.extend(sentences_by_class[c])
random.shuffle(vb_vbg_errors)
vb_vbg_errors = vb_vbg_errors[:num_per_class]

vb_vbg_errors_test = [vb_vbg_errors[index] for index in range(5)]  # check!
vb_vbg_errors_train = [vb_vbg_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(vb_vbg_errors_test, "datasets for MTurk/error_types_combined_data/vb_vbg_errors_test")
# write_pickle(vb_vbg_errors_train, "datasets for MTurk/error_types_combined_data/vb_vbg_errors_train")

# ERROR TYPE 8 ("other verb forms") # + check?
other_verb_errors = []
for c in [11, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28]:
    other_verb_errors.extend(sentences_by_class[c])
random.shuffle(other_verb_errors)
other_verb_errors = other_verb_errors[:num_per_class]

other_verb_errors_test = [other_verb_errors[index] for index in range(5)]  # check!
other_verb_errors_train = [other_verb_errors[index] for index in range(5, num_per_class)]  # + check?

# write_pickle(other_verb_errors_test, "datasets for MTurk/error_types_combined_data/other_verb_errors_test")
# write_pickle(other_verb_errors_train, "datasets for MTurk/error_types_combined_data/other_verb_errors_train")

# ----------
"""
# quality? => inspect 20 sentences per category
# $CASE_CAPITAL (too many double sentences => ((+) not) starting with capital letter error)
test = sentences_by_class[0]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)

# $CASE_LOWER
test = sentences_by_class[2]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)

# $NOUN_NUMBER_SINGULAR
test = sentences_by_class[7]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)

# $NOUN_NUMBER_PLURAL
test = sentences_by_class[8]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)

# $VB_VBZ
test = sentences_by_class[9]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)

# $VB_VBN
test = sentences_by_class[10]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)

# $VB_VBG
test = sentences_by_class[12]
random.shuffle(test)
test_sentences(test[:20], is_triple=True)
"""
