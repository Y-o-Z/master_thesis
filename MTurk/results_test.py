import boto3
import csv
import xmltodict
import datetime  # is used (?)
import copy

from helper import read_pickle, write_pickle, get_mturker_dictionary_entry, NUM_CLASSES, NUM_SENTENCES_PER_CLASS_TEST

import matplotlib.pyplot as plt
from scipy import stats


data = []
with open("new_user_credentials.csv") as file:
    csv_reader = csv.reader(file, delimiter=",")
    for row in csv_reader:
        data.append(row)

MTURK_SANDBOX = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

mturk = boto3.client("mturk",
                     aws_access_key_id=data[1][2],
                     aws_secret_access_key=data[1][3],
                     region_name="us-east-1",
                     # endpoint_url=MTURK_SANDBOX  # leave out endpoint_url to connect to live MTurk
                     )

# ---

# [0] → correct, [1] → error, [2] → solution(, [3] → identifier)
class_test_casing = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/casing_errors_test")
class_test_singular = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/singular_errors_test")
class_test_plural = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/plural_errors_test")
class_test_vb_vbz = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vb_vbz_errors_test")
class_test_vbz_vb = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vbz_vb_errors_test")
class_test_vb__vbn = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vb_vbn_errors_test")
class_test_vb__vbg = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vb_vbg_errors_test")
class_test_other = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/other_verb_errors_test")

number_to_class = {0: "casing", 1: "singular", 2: "plural", 3: "vb_vbz", 4: "vbz_vb", 5: "vb__vbn", 6: "vb__vbg",
                   7: "other"}

sentences_test = []
for i, suffix in enumerate(number_to_class.values()):
    tmp = []  # => silence warning
    exec(f"tmp = class_test_{suffix}")
    for index in range(NUM_SENTENCES_PER_CLASS_TEST):
        tmp[index].append(f"{i}_{index}")  # class, sentence number [3]
    sentences_test.extend(tmp)


def overview(assignments):
    for assignment in assignments:
        print("-" * 10)
        print(f"HITId: {assignment['HITId']}, AssignmentId: {assignment['AssignmentId']}, "
              f"WorkerId: {assignment['WorkerId']}")
        print(f"AcceptTime: {assignment['AcceptTime']}, SubmitTime: {assignment['SubmitTime']}")
        print(score_answers(assignment))
        # => if no comment: => first exercise
        print(f"comment: {xmltodict.parse(assignment['Answer'])['QuestionFormAnswers']['Answer'][0]['FreeText']}")
        print("-" * 10)
        _ = input()


def display_answers(assignment, add_solution=False):  # not in survey order, but alphabetically
    xml_doc = xmltodict.parse(assignment["Answer"])
    print("-" * 10)
    if isinstance(xml_doc["QuestionFormAnswers"]["Answer"], list):
        # multiple fields in HIT layout
        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
            print("For input field: " + answer_field["QuestionIdentifier"])
            print("Submitted answer: " + answer_field["FreeText"])
            if add_solution and answer_field["QuestionIdentifier"].startswith("exercise"):
                _, i, j = map(lambda x: int(x) if x != "exercise" else -1,
                              answer_field["QuestionIdentifier"].split("_"))  # => [..]
                ldict = {}
                exec(f"tmp = class_test_{number_to_class[i]}", globals(), ldict)
                tmp = ldict["tmp"]
                print(f"Suggestion: {tmp[j][2]}")
    else:
        # single field in HIT layout
        print("For input field (1): " + xml_doc["QuestionFormAnswers"]["Answer"]["QuestionIdentifier"])
        print("Submitted answer (1): " + xml_doc["QuestionFormAnswers"]["Answer"]["FreeText"])
    print(f"AssignmentId = {assignment['AssignmentId']}, WorkerId = {assignment['WorkerId']}")
    print("-" * 10)


def display_all_answers_for_exercise(assignments, exercise):
    _, i, j = map(lambda x: int(x) if x != "exercise" else -1, exercise.split("_"))  # => [..]
    print("-" * 10)
    for assignment in assignments:
        xml_doc = xmltodict.parse(assignment["Answer"])
        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
            e = answer_field["QuestionIdentifier"]
            if e == exercise:
                print(answer_field["FreeText"])
    ldict = {}
    exec(f"tmp = class_test_{number_to_class[i]}", globals(), ldict)
    tmp = ldict["tmp"]
    print(tmp[j][2])  # solution


def score_answers(assignment):  # NUM_SENTENCES_PER_CLASS_TEST=5 sentences per class
    xml_doc = xmltodict.parse(assignment["Answer"])
    correct_answers = [0 for _ in range(NUM_CLASSES)]
    for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
        if answer_field["QuestionIdentifier"].startswith("exercise"):
            exercise = answer_field["QuestionIdentifier"]
            # => [int(x) if x != "exercise" else -1 for x in exercise.split("_")]
            _, i, j = map(lambda x: int(x) if x != "exercise" else -1, exercise.split("_"))  # class, sentence
            ldict = {}
            exec(f"solution = class_test_{number_to_class[i]}[{j}][0]", globals(), ldict)  # global, local?
            solution = ldict["solution"]  # "import" from local
            if solution == answer_field["FreeText"]:  # %
                correct_answers[i] += 1 / NUM_SENTENCES_PER_CLASS_TEST
    return correct_answers


def score_exercises(assignments):
    exercise_scores = {}
    for assignment in assignments:
        xml_doc = xmltodict.parse(assignment["Answer"])
        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
            exercise = answer_field["QuestionIdentifier"]
            if not exercise.startswith("exercise"):
                continue
            # => [int(x) if x != "exercise" else -1 for x in exercise.split("_")]
            _, i, j = map(lambda x: int(x) if x != "exercise" else -1, exercise.split("_"))  # class, sentence
            ldict = {}
            exec(f"solution = class_test_{number_to_class[i]}[{j}][0]", globals(), ldict)  # global, local?
            solution = ldict["solution"]  # "import" from local
            if solution == answer_field["FreeText"]:  # num_correct
                if exercise in exercise_scores:
                    exercise_scores[exercise] += 1
                else:
                    exercise_scores[exercise] = 1
    return exercise_scores


def evaluate_assignments(assignments, automatic_approve=False, m="Thank you for participating and I wish you a lot "
                                                                 "of interesting HITs in your future Turker-days."):
    for assignment in assignments:
        worker_id = assignment["WorkerId"]
        score = score_answers(assignment)
        if worker_id not in mturker_dictionary:
            mturker_dictionary[worker_id] = get_mturker_dictionary_entry()
        for c in range(NUM_CLASSES):
            mturker_dictionary[worker_id][number_to_class[c]][0] += score[c]
            mturker_dictionary[worker_id][number_to_class[c]][1] += 1 - score[c]
            mturker_dictionary[worker_id][number_to_class[c]][3] = assignment["SubmitTime"].timestamp()
        if automatic_approve:
            r = mturk.approve_assignment(AssignmentId=assignment["AssignmentId"], RequesterFeedback=m)
            print(r)


def create_dictionary_from_hits(hits, bad_qualifications=None, ignore_workers=None, verify_workers=None):
    bad_qualifications = [] if bad_qualifications is None else bad_qualifications
    ignore_workers = [] if ignore_workers is None else ignore_workers
    verify_workers = [] if verify_workers is None else verify_workers
    bad_workers = set()
    for qualification in bad_qualifications:
        tmp = mturk.list_workers_with_qualification_type(QualificationTypeId=qualification)["Qualifications"]
        for t in tmp:
            bad_workers.add(t["WorkerId"])
    for worker in ignore_workers:
        bad_workers.add(worker)
    my_dict = {}
    for hit in hits:
        assignments = mturk.list_assignments_for_hit(HITId=hit)["Assignments"]
        for assignment in assignments:
            if assignment["AssignmentId"] in doubles_approved:  # (=> recruitment)
                continue
            if (assignment["WorkerId"] not in bad_workers and not verify_workers or
                assignment["WorkerId"] in verify_workers) and assignment["AssignmentStatus"] == "Approved":
                if assignment["WorkerId"] not in my_dict:
                    my_dict[assignment["WorkerId"]] = score_answers(assignment)
                else:
                    scores = score_answers(assignment)
                    for c in range(NUM_CLASSES):
                        my_dict[assignment["WorkerId"]][c] += scores[c]
    return my_dict


mturker_dictionary = read_pickle("mturker_dictionary")
test_group = read_pickle("test_group")
control_group = read_pickle("control_group")
base_group = read_pickle("base_group")
all_workers = read_pickle("all_workers")
blacklist = read_pickle("blacklist")
that_one_Turker_that_did_not_finish = "A1D9FECXIKCO4O"
blacklist.append(that_one_Turker_that_did_not_finish)
doubles_approved = read_pickle("doubles_approved")

final_test_group = [w for w in test_group if w not in blacklist]
final_control_group = [w for w in control_group if w not in blacklist]
final_base_group = [w for w in base_group if w not in blacklist]

reviewable_hits = mturk.list_reviewable_hits(MaxResults=100); print(f"NumResults: {reviewable_hits['NumResults']}")
# reviewable_hits = mturk.list_hits(MaxResults=100); print(f"NumResults: {reviewable_hits['NumResults']}")  # ..

hit_ids = []
for data in reviewable_hits["HITs"]:
    if data["RequesterAnnotation"].startswith("FINAL"):  # ..
        hit_ids.append(data["HITId"])
print(len(hit_ids))

# => in ("Submitted", "Approved", "Rejected")
assignments = []
for hit_id in hit_ids:
    tmp = mturk.list_assignments_for_hit(HITId=hit_id)
    for assignment in tmp["Assignments"]:
        if assignment["AssignmentStatus"] == "Approved":  # ..
            assignments.append(assignment)
    if not tmp["Assignments"]:  # empty
        print(hit_id)
print(len(assignments))

"""
evaluate_assignments(assignments)
write_pickle(mturker_dictionary, "mturker_dictionary")
# """

"""
assignments[i] =>
{
    'AssignmentId': 'string',
    'WorkerId': 'string',
    'HITId': 'string',
    'AssignmentStatus': 'Submitted'|'Approved'|'Rejected',
    'AutoApprovalTime': datetime(2015, 1, 1),
    'AcceptTime': datetime(2015, 1, 1),
    'SubmitTime': datetime(2015, 1, 1),
    'ApprovalTime': datetime(2015, 1, 1),
    'RejectionTime': datetime(2015, 1, 1),
    'Deadline': datetime(2015, 1, 1),
    'Answer': 'string',
} 
"""

# ------------------------------

# => test, control, base
"""
initial_hits = read_pickle("initial_hits")  # 17 (initial)
assignments = []
already_present = []  # no doubles (=> recruitment)
for hit in initial_hits:
    tmp = mturk.list_assignments_for_hit(HITId=hit)
    for assignment in tmp["Assignments"]:
        if assignment["WorkerId"] in all_workers and not assignment["WorkerId"] in already_present:
            assignments.append(assignment)
            already_present.append(assignment["WorkerId"])
worker_scores = {w: 0 for w in all_workers}
for assignment in assignments:  # 69
    scores = score_answers(assignment)
    worker_scores[assignment["WorkerId"]] = sum(scores)
sorted_worker_scores = dict(sorted(worker_scores.items(), key=lambda x: x[1]))
my_test_group = []
my_control_group = []
my_base_group = []
counter = 0
for w in sorted_worker_scores:
    if counter == 0:
        my_test_group.append(w)
        counter += 1
    elif counter == 1:
        my_control_group.append(w)
        counter += 1
    else:
        my_base_group.append(w)
        counter = 0
# + write_pickle(..)
"""

# ------------------------------

# Visualization
# (initial/final) percentage of correct answers per error class
initial_hits = read_pickle("initial_hits")
final_hits = read_pickle("final_hits")

my_dict_initial_all = create_dictionary_from_hits(initial_hits, ignore_workers=blacklist)
my_dict_initial_all = dict(sorted(my_dict_initial_all.items(), key=lambda x: x[0]))
# ---
my_dict_initial_test = create_dictionary_from_hits(initial_hits, verify_workers=final_test_group)
my_dict_initial_test = dict(sorted(my_dict_initial_test.items(), key=lambda x: x[0]))
my_dict_initial_control = create_dictionary_from_hits(initial_hits, verify_workers=final_control_group)
my_dict_initial_control = dict(sorted(my_dict_initial_control.items(), key=lambda x: x[0]))
my_dict_initial_base = create_dictionary_from_hits(initial_hits, verify_workers=final_base_group)
my_dict_initial_base = dict(sorted(my_dict_initial_base.items(), key=lambda x: x[0]))
# ---
my_dict_final_all = create_dictionary_from_hits(final_hits, ignore_workers=blacklist)
my_dict_final_all = dict(sorted(my_dict_final_all.items(), key=lambda x: x[0]))
# ---
my_dict_final_test = create_dictionary_from_hits(final_hits, verify_workers=final_test_group)
my_dict_final_test = dict(sorted(my_dict_final_test.items(), key=lambda x: x[0]))
my_dict_final_control = create_dictionary_from_hits(final_hits, verify_workers=final_control_group)
my_dict_final_control = dict(sorted(my_dict_final_control.items(), key=lambda x: x[0]))
my_dict_final_base = create_dictionary_from_hits(final_hits, verify_workers=final_base_group)
my_dict_final_base = dict(sorted(my_dict_final_base.items(), key=lambda x: x[0]))

"""
for w in my_dict_initial_all:
    print(f"worker {w}: initial points: {sum(my_dict_initial_all[w]) * 5:.2f}, final points: "
          f"{sum(my_dict_final_all[w]) * 5:.2f}, difference (f - i): "
          f"{(sum(my_dict_initial_all[w]) - sum(my_dict_final_all[w])) * 5:.2f}")
# """

my_dicts = [my_dict_initial_test, my_dict_initial_control, my_dict_initial_base]
# my_dicts = [my_dict_final_all]
names = ["initial_test", "final_test", "initial_control", "final_control", "initial_base", "final_base"]
# names = ["final_all"]

# """
add_dict = my_dict_initial_all
counts_add = [0 for _ in range(NUM_CLASSES)]
for k in add_dict:
    for c in range(NUM_CLASSES):
        counts_add[c] += add_dict[k][c]
counts_add = [100 * c / len(add_dict) for c in counts_add]
# """

for index in range(len(my_dicts)):
    num_entries = len(my_dicts[index])
    counts = [0 for _ in range(NUM_CLASSES)]  # => "how many percentage points out of 100"
    sd = [0 for _ in range(NUM_CLASSES)]

    for k in my_dicts[index]:
        for c in range(NUM_CLASSES):
            counts[c] += my_dicts[index][k][c]
    counts = [c / num_entries for c in counts]
    for k in my_dicts[index]:
        for c in range(NUM_CLASSES):
            sd[c] += abs(counts[c] - my_dicts[index][k][c])
    sd = [d / num_entries for d in sd]
    for c in range(NUM_CLASSES):  # %, same keys
        counts[c] *= 100
        sd[c] *= 100
    print(f"overall average for {names[index]}: {sum(counts) / NUM_CLASSES}")

    plt.style.use("seaborn")
    plt.figure(figsize=(12, 5))
    plt.errorbar(list(number_to_class.values()), counts, yerr=sd, marker="^", linestyle="None", ecolor="orange",
                 capsize=5, markeredgewidth=0.9)
    plt.errorbar(list(number_to_class.values()), counts_add, marker="^", color="grey", linestyle="None")
    plt.xticks(rotation=45)
    plt.ylim([0, 100])
    plt.yticks(list(range(0, 101, 10)))
    plt.ylabel("% correct answers per error group")
    # plt.title(names[index])
    plt.savefig("final_test_all.png", dpi=500, bbox_inches="tight")
    plt.show()

# points per student  # ..
"""
points = [0 for _ in range(len(mturker_dictionary))]
for i, k in enumerate(mturker_dictionary):
    for c in number_to_class.values():
        points[i] += mturker_dictionary[k][c][0] * NUM_SENTENCES_PER_CLASS_TEST
points_dict = {}
for p in points:
    if p in points_dict:
        points_dict[p] += 1
    else:
        points_dict[p] = 1
mean = sum([c * points_dict[c] for c in points_dict]) / sum(points_dict.values())
values = []
for c in points_dict:
    for _ in range(points_dict[c]):
    values.append(c)
sd = sum([abs(v - mean) for v in values]) / sum(points_dict.values())
# """

# statistical tests

# q1 & q2: is leitner better than random selection? do hints improve student learning?
# => tailed t-test for independent samples
test = copy.deepcopy(my_dict_final_test)
control = copy.deepcopy(my_dict_final_control)
base = copy.deepcopy(my_dict_final_base)

# test = copy.deepcopy(my_dict_initial_test)
# control = copy.deepcopy(my_dict_initial_control)
# base = copy.deepcopy(my_dict_initial_base)


"""
test_group = read_pickle("test_group")
control_group = read_pickle("control_group")

test_scores = []
control_scores = []
for w in final:
    if w in test_group:
        test_scores.append(final[w])
        index = test_group.index(w)
        control_scores.append(final[control_group[index]])  # corresponding

scores_test = [sum([s * NUM_SENTENCES_PER_CLASS_TEST for s in scores]) for scores in test_scores]
mean_test = sum(scores_test) / len(scores_test)
sd_test = sum([abs(score - mean_test) for score in scores_test]) / len(scores_test)

scores_control = [sum([s * NUM_SENTENCES_PER_CLASS_TEST for s in scores]) for scores in control_scores]
mean_control = sum(scores_control) / len(scores_control)
sd_control = sum([abs(score - mean_control) for score in scores_control]) / len(scores_control)

results = []
iterations = 1000
for _ in range(iterations):
    test_scores = stats.norm.rvs(loc=1, scale=2, size=3)  # scale => standard deviation
    control_scores = stats.norm.rvs(loc=1, scale=2, size=3)

    r1, r2 = stats.ttest_ind(test_scores, control_scores, equal_var=True, alternative="greater")
    results.append(r2)
print(sum(results) / iterations)
# """

# test: 1,2,3,11,14,19,20 not present
# control: 1,2,4,5 not present
# base: 1,3 not present
for index in [0, 1, 3, 4]:
    if test_group[index] in test:
        test.pop(test_group[index])
for index in [0, 1, 2, 10, 13, 18, 19]:
    if control_group[index] in control:
        control.pop(control_group[index])
assert len(test) == len(control)

for index in [0, 2]:
    if control_group[index] in control:
        control.pop(control_group[index])
for index in [0, 1, 3, 4]:
    if base_group[index] in base:
        base.pop(base_group[index])
assert len(control) == len(base)

scores_test = [NUM_SENTENCES_PER_CLASS_TEST * sum(score) for worker, score in test.items()]
mean_test = sum(scores_test) / len(scores_test); print(mean_test)
sd_test = sum([abs(score - mean_test) for score in scores_test]) / len(scores_test)

scores_control = [NUM_SENTENCES_PER_CLASS_TEST * sum(score) for worker, score in control.items()]
mean_control = sum(scores_control) / len(scores_control); print(mean_control)
sd_control = sum([abs(score - mean_control) for score in scores_control]) / len(scores_control)

scores_base = [NUM_SENTENCES_PER_CLASS_TEST * sum(score) for worker, score in base.items()]
mean_base = sum(scores_base) / len(scores_base); print(mean_base)
sd_base = sum([abs(score - mean_base) for score in scores_base]) / len(scores_base)

r1, r2 = stats.ttest_ind(scores_test, scores_control, equal_var=False, alternative="greater")

r3, r4 = stats.ttest_ind(scores_control, scores_base, equal_var=True, alternative="greater")

# ----------

# q3: did the students learn?
# => paired t-test (before, after)
initial = my_dict_initial_all
final = my_dict_final_all

scores_initial = [NUM_SENTENCES_PER_CLASS_TEST * sum(score) for _, score in initial.items()]
mean_initial = sum(scores_initial) / len(scores_initial)
sd_initial = sum([abs(score - mean_initial) for score in scores_initial]) / len(scores_initial)

scores_final = [NUM_SENTENCES_PER_CLASS_TEST * sum(score) for _, score in final.items()]
mean_final = sum(scores_final) / len(scores_final)
sd_final = sum([abs(score - mean_final) for score in scores_final]) / len(scores_final)

"""
results = []
iterations = 1000
for _ in range(iterations):
    scores_before = stats.norm.rvs(loc=mean_initial, scale=sd_initial, size=4)
    scores_after = stats.norm.rvs(loc=mean_final, scale=sd_final, size=4)

    r3, r4 = stats.ttest_rel(scores_after, scores_before, alternative="greater")
    results.append(r4)
print(sum(results) / iterations)
# """

r5, r6 = stats.ttest_rel(scores_final, scores_initial, alternative="greater")  #  << 0.01

# per error class
for error_class in range(8):
    scores_initial_class = [NUM_SENTENCES_PER_CLASS_TEST * score[error_class] for _, score in initial.items()]
    scores_final_class = [NUM_SENTENCES_PER_CLASS_TEST * score[error_class] for _, score in final.items()]
    r7, r8 = stats.ttest_rel(scores_final_class, scores_initial_class, alternative="greater")  # < 0.05 for all
    print(f"{number_to_class[error_class]}: {r8}")



