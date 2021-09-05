# clean text files (conll_2014, ..) for better GECToR performance

from string import punctuation
for c in [".", ",", "!", '"', "'", "-"]:
    punctuation = punctuation.replace(c, "")


# folder = "/home/yoel/Desktop/master thesis/tempora/prototype/datasets/CoNLL-2014/"
folder = "/home/yoel/Desktop/master thesis/tempora/prototype/datasets/fce_v2.1.bea19/"


def find_occurrences(sentence, word):
    return [i for i, w in enumerate(sentence) if w == word]


# "conll_errors.txt"
with open(folder + "error_dev.txt") as file:
    error_lines = file.readlines()
with open(folder + "error_test.txt") as file:
    error_2_lines = file.readlines()
error_lines.extend(error_2_lines)

# "conll_corrected.txt"
with open(folder + "correct_dev.txt") as file:
    correct_lines = file.readlines()
with open(folder + "correct_test.txt") as file:
    correct_2_lines = file.readlines()
correct_lines.extend(correct_2_lines)

print(f"before: num lines: {len(error_lines)}")

assert len(error_lines) == len(correct_lines)

index = 0
while index < len(error_lines):
    error_lines[index] = error_lines[index].strip()
    correct_lines[index] = correct_lines[index].strip()
    # remove sentences w/o normal termination
    if error_lines[index][-1] not in [".", "!"] or correct_lines[index][-1] not in [".", "!"]:
        del error_lines[index]
        del correct_lines[index]
        continue
    # remove "multi-sentences"
    if error_lines[index].count(".") + error_lines[index].count("!") > 1 or \
            correct_lines[index].count(".") + correct_lines[index].count("!") > 1:
        del error_lines[index]
        del correct_lines[index]
        continue
    # remove sentences that are too short (..21) or too long (120..)
    if not 21 < len(error_lines[index]) < 120 or not 21 < len(correct_lines[index]) < 120:
        del error_lines[index]
        del correct_lines[index]
        continue
    # remove sentences with special punctuation symbols
    fail = False
    for c in punctuation:
        if error_lines[index].count(c) or correct_lines[index].count(c):  # != 0
            fail = True
            break
    if fail:
        del error_lines[index]
        del correct_lines[index]
        continue
    # remove sentences (w)if only capital letters
    if error_lines[index] == error_lines[index].upper() or correct_lines[index] == correct_lines[index].upper():
        del error_lines[index]
        del correct_lines[index]
        continue
    # remove apostrophes (single quotes → sentences) except for " 's ", " n't "
    if error_lines[index].count("'") or correct_lines[index].count("'"):  # != 0
        occurrences = find_occurrences(error_lines[index], "'")
        for occurrence in occurrences:
            if occurrence < 2 or (error_lines[index][occurrence - 1: occurrence + 3] != " 's " and
                                  error_lines[index][occurrence - 2: occurrence + 3] != " n't "):
                fail = True
                break
        occurrences = find_occurrences(correct_lines[index], "'")
        for occurrence in occurrences:
            if occurrence < 2 or (correct_lines[index][occurrence - 1: occurrence + 3] != " 's " and
                                  correct_lines[index][occurrence - 2: occurrence + 3] != " n't "):
                fail = True
                break
        if fail:
            del error_lines[index]
            del correct_lines[index]
            continue
    # remove double quotes
    if error_lines[index].count('"') or correct_lines[index].count('"'):  # != 0
        occurrences = find_occurrences(error_lines[index], '"')
        drift = 0  # keep track of index drift due to slicing
        for occurrence in occurrences:
            error_lines[index] = error_lines[index][:occurrence + drift] + error_lines[index][occurrence + drift + 2:]
            drift -= 2
        occurrences = find_occurrences(correct_lines[index], '"')
        drift = 0  # keep track of index drift due to slicing
        for occurrence in occurrences:
            correct_lines[index] = correct_lines[index][:occurrence + drift] + \
                                   correct_lines[index][occurrence + drift + 2:]
            drift -= 2

    index += 1  # if and else

print(f"after: num lines: {len(error_lines)}")

assert len(error_lines) == len(correct_lines)

# "conll_errors_clean.txt"
with open(folder + "fce_errors_clean.txt", "w") as file:
    file.write("\n".join(error_lines))

# "conll_corrected_clean.txt"
with open(folder + "fce_corrected_clean.txt", "w") as file:
    file.write("\n".join(correct_lines))
