import boto3
import csv
import xmltodict
import datetime  # is used

from helper import read_pickle, write_pickle, NUM_CLASSES, NUM_SENTENCES_PER_CLASS_TRAIN, NUM_SENTENCES_PER_CLASS_TEST

import matplotlib.pyplot as plt


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
class_train_casing = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/casing_errors_train")
class_train_singular = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/singular_errors_train")
class_train_plural = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/plural_errors_train")
class_train_vb_vbz = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vb_vbz_errors_train")
class_train_vbz_vb = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vbz_vb_errors_train")
class_train_vb__vbn = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vb_vbn_errors_train")
class_train_vb__vbg = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/vb_vbg_errors_train")
class_train_other = read_pickle("../prototype/datasets for MTurk/error_types_combined_data/other_verb_errors_train")

number_to_class = {0: "casing", 1: "singular", 2: "plural", 3: "vb_vbz", 4: "vbz_vb", 5: "vb__vbn", 6: "vb__vbg",
                   7: "other"}

sentences_train = []
for i, suffix in enumerate(number_to_class.values()):
    tmp = []  # => silence warning
    exec(f"tmp = class_train_{suffix}")
    for index in range(NUM_SENTENCES_PER_CLASS_TRAIN):
        tmp[index].append(f"{i}_{index}")  # class, sentence number [3]
    sentences_train.extend(tmp)

# solution "styling"
"""
for index in range(NUM_CLASSES * NUM_SENTENCES_PER_CLASS_TRAIN):
    sentences_train[index][2] = sentences_train[index][2].replace("\x1b[91m{", "<span style='color:red'>")
    sentences_train[index][2] = sentences_train[index][2].replace("\x1b[94m{", "<span style='color:blue'>")
    sentences_train[index][2] = sentences_train[index][2].replace("}\x1b[0m", "</span>")
# """


def overview_train(assignments):
    for assignment in assignments:
        print("-" * 10)
        print(f"HITId: {assignment['HITId']}, AssignmentId: {assignment['AssignmentId']}, "
              f"WorkerId: {assignment['WorkerId']}")
        print(f"AcceptTime: {assignment['AcceptTime']}, SubmitTime: {assignment['SubmitTime']}")
        s, e = score_answers_train(assignment)
        print(s)
        print(f"classes = {e}")
        # => if no comment: \n\t first exercise
        print(f"comment: {xmltodict.parse(assignment['Answer'])['QuestionFormAnswers']['Answer'][0]['FreeText']}")
        print("-" * 10)
        _ = input()


def display_answers_train(assignment, one_by_one=False, add_solution=False):  # not in survey order, but alphabetically
    xml_doc = xmltodict.parse(assignment["Answer"])
    print("-" * 10)
    if isinstance(xml_doc["QuestionFormAnswers"]["Answer"], list):
        # Multiple fields in HIT layout
        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
            print("For input field: " + answer_field["QuestionIdentifier"])
            print("Submitted answer: " + answer_field["FreeText"])
            if add_solution and answer_field["QuestionIdentifier"].startswith("exercise"):
                _, i, j = map(lambda x: int(x) if x != "exercise" else -1,
                              answer_field["QuestionIdentifier"].split("_"))
                ldict = {}
                exec(f"tmp = class_train_{number_to_class[i]}", globals(), ldict)
                tmp = ldict["tmp"]
                print(f"Solution: {tmp[j][2]}")
            if one_by_one:
                _ = input()
    else:
        # One field in HIT layout
        print("For input field (1): " + xml_doc["QuestionFormAnswers"]["Answer"]["QuestionIdentifier"])
        print("Submitted answer (1): " + xml_doc["QuestionFormAnswers"]["Answer"]["FreeText"])
    print(f"AssignmentId = {assignment['AssignmentId']}, WorkerId = {assignment['WorkerId']}")
    print("-" * 10)


def display_all_answers_for_exercise_train(assignments, exercise):
    _, i, j = map(lambda x: int(x) if x != "exercise" else -1, exercise.split("_"))
    print("-" * 10)
    for assignment in assignments:
        xml_doc = xmltodict.parse(assignment["Answer"])
        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
            e = answer_field["QuestionIdentifier"]
            if e == exercise:
                print(answer_field["FreeText"])
    ldict = {}
    exec(f"tmp = class_train_{number_to_class[i]}", globals(), ldict)
    tmp = ldict["tmp"]
    print(tmp[j][2])


def score_answers_train(assignment):
    xml_doc = xmltodict.parse(assignment["Answer"])
    # num_class = (len(xml_doc["QuestionFormAnswers"]["Answer"]) - 2) // NUM_SENTENCES_PER_CLASS_TEST
    correct_answers = [0 for _ in range(NUM_CLASSES)]
    exercises = {}
    for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
        if answer_field["QuestionIdentifier"].startswith("exercise"):
            exercise = answer_field["QuestionIdentifier"]
            # => [int(x) if x != "exercise" else -1 for x in exercise.split("_")]
            _, i, j = map(lambda x: int(x) if x != "exercise" else -1, exercise.split("_"))  # class, sentence
            if i in exercises:
                exercises[i].append(j)
            else:
                exercises[i] = [j]
            ldict = {}
            exec(f"solution = class_train_{number_to_class[i]}[{j}][2]", globals(), ldict)  # global, local?
            solution = ldict["solution"]  # "import" from local
            error_word = solution.split("{")[1].split("}")[0]
            correct_word = solution.split("{")[2].split("}")[0]
            char_after = solution.split("{")[2].split("}")[1][4]
            if error_word + char_after not in answer_field["FreeText"] and \
                    correct_word + char_after in answer_field["FreeText"]:
                correct_answers[i] += 1 / NUM_SENTENCES_PER_CLASS_TEST
    return correct_answers, exercises


"""
def score_exercises_train(assignments):  # less useful ...
    exercise_scores = {}
    for assignment in assignments:
        xml_doc = xmltodict.parse(assignment["Answer"])
        for answer_field in xml_doc["QuestionFormAnswers"]["Answer"]:
            exercise = answer_field["QuestionIdentifier"]
            if not exercise.startswith("exercise"):
                continue
            _, i, j = map(lambda x: int(x) if x != "exercise" else -1, exercise.split("_"))  # class, sentence
            ldict = {}
            exec(f"solution = class_train_{number_to_class[i]}[{j}][2]", globals(), ldict)  # global, local?
            solution = ldict["solution"]  # "import" from local
            error_word = solution.split("{")[1].split("}")[0]
            correct_word = solution.split("{")[2].split("}")[0]
            char_after = solution.split("{")[2].split("}")[1][4]
            if error_word + char_after not in answer_field["FreeText"] and \
                    correct_word + char_after in answer_field["FreeText"]:
                if exercise in exercise_scores:
                    exercise_scores[exercise] += 1
                else:
                    exercise_scores[exercise] = 1
    return exercise_scores, len(assignments) * NUM_SENTENCES_PER_CLASS_TEST
# """


def evaluate_assignments_train(assignments, automatic_approve=False, automatic_update_status=False,
                               m="Thank you for participating. See you soon!"):
    for assignment in assignments:
        worker_id = assignment["WorkerId"]
        score, exercises = score_answers_train(assignment)
        for c in range(NUM_CLASSES):
            if c in exercises:
                mturker_dictionary[worker_id][number_to_class[c]][0] += score[c]
                mturker_dictionary[worker_id][number_to_class[c]][1] += 1 - score[c]
                mturker_dictionary[worker_id][number_to_class[c]][3] = assignment["SubmitTime"].timestamp()
                for j in exercises[c]:
                    mturker_dictionary[worker_id][number_to_class[c]][2].append(j)
        if automatic_approve:
            r = mturk.approve_assignment(AssignmentId=assignment["AssignmentId"], RequesterFeedback=m)
            # print(r)
            if automatic_update_status:
                mturk.update_hit_review_status(HITId=assignment["HITId"])


def create_dictionary_from_hits_train(hits, bad_qualifications=None, ignore_workers=None):
    bad_qualifications = [] if bad_qualifications is None else bad_qualifications
    ignore_workers = [] if ignore_workers is None else ignore_workers
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
            if assignment["WorkerId"] not in bad_workers and assignment["AssignmentStatus"] == "Approved":
                if assignment["WorkerId"] not in my_dict:
                    scores = score_answers_train(assignment)
                    exercises = list(scores[1])
                    my_dict[assignment["WorkerId"]] = score_answers_train(assignment)[0]
                    my_dict[assignment["WorkerId"]].append(exercises)
                else:
                    scores = score_answers_train(assignment)
                    exercises = list(scores[1])
                    for c in range(NUM_CLASSES):
                        my_dict[assignment["WorkerId"]][c] += scores[c][0]
                    my_dict[assignment["WorkerId"]][-1].extend(exercises)
    return my_dict


mturker_dictionary = read_pickle("mturker_dictionary")
not_done_dictionary = read_pickle("not_done_dictionary")

reviewable_hits = mturk.list_reviewable_hits(MaxResults=100); print(f"NumResults: {reviewable_hits['NumResults']}")

my_qualifications = mturk.list_qualification_types(MustBeRequestable=False, MustBeOwnedByCaller=True, MaxResults=100)
my_qualifications = my_qualifications["QualificationTypes"]

hit_ids = []
should_be_workers = []
for data in reviewable_hits["HITs"]:
    if data["RequesterAnnotation"].startswith("FINAL"):
        hit_ids.append(data["HITId"])
        # hit = mturk.get_hit(HITId=data["HITId"])
        # for q in my_qualifications:
        #     if q["QualificationTypeId"] == hit["HIT"]["QualificationRequirements"][0]["QualificationTypeId"]:
        #         should_be_workers.append(q["Name"])
        #         break
print(len(hit_ids))

# + Submitted, Approved, Rejected
are_workers = []
assignments = []
for hit_id in hit_ids:
    tmp = mturk.list_assignments_for_hit(HITId=hit_id)
    for assignment in tmp["Assignments"]:
        if assignment["AssignmentStatus"] == "Submitted":
            assignments.append(assignment)
            are_workers.append(assignment["WorkerId"])
    if not tmp["Assignments"] and False:  # empty
        # print(hit_id)
        pass
print(len(assignments))

"""
for w in should_be_workers:
    if w not in are_workers:
        if w not in not_done_dictionary:
            not_done_dictionary[w] = 1
        else:
            not_done_dictionary[w] += 1
# write_pickle(not_done_dictionary, "not_done_dictionary")
# """

# 0) overview
# x) ...
# -1) evaluate

# ------------------------------

# Visualization
# percentage of correct answers per error class
# list of hits (submitted, i => round (1))
h1 = ['3KTZHH2ONIFT4SQCNC2JLKVKFAU8MC', '308KJXFUJR6B0B9BT7N93PW7Z6QATJ', '3P6ENY9P79W01K1E51YX64H21K0IHL',
      '31JUPBOORN49YA8RN014L7GVF548LA']
h2 = ['3VZYA8PITOYDEW03098IB3EGAHE05D', '3BO3NEOQM0HLRXT0ZVO5PIUYOPWAIO', '3S1WOPCJFGTKCLUIXIBRX2ZPMJJEJZ',
      '3L60IFZKF3I05PQUZJGC6SGN177HHD', '36KM3FWE3RCS10EMO2NF9NSC1N907S']
h3 = ['3GL25Y6843UJJTRKVKEB1JCD6LUMXT', '3N3WJQXELSQYCV627JS3BG3ST742LX', '36D1BWBEHN1IIDON7VTIBTIHTCU2MZ',
      '3GITHABACYLO0V9NJW8IO87HY242NZ', '3Y3N5A7N4G98JYHU0G0DKXJM8ROMYZ', '3NSCTNUR2ZN9F9YLSB5B09QTJ5W5AJ',
      '39RRBHZ0AU1REBL8AU3NL0B15E4ZVK', '3GONHBMNHVY8OORZW22WY4SOELSMZQ']
h4 = ['33K3E8REWWV4Y4PG1J4SW1BIMIE8X4', '3ZZAYRN1I6R02F3C3QSR4KGCORTTO8', '32TZXEA1OLKVP2HLINT8KGRCFH3411',
      '3MJ28H2Y1E8YHDSM2UAT14AY8O7O5C', '3QI9WAYOGQB9Y9KTEU17DFYZ18M6S8', '3L2OEKSTW9ATY9FQ6FQUFFIROO88YA',
      '3HO4MYYR12OPDMCX3ZW7SGGNM576UO']
h5 = ['36JW4WBR06KGRTZO4TSC374ALX3FHW', '3D4BBDG7ZHWUU98FY6D9R7IX5FT3C2', '3HYV4299H0WVS4YZ6EE08CQB6DY8ES',
      '3IH9TRB0FBZPX9G03CVCEH1VK0ZI1R', '30Y6N4AHYPWWI3ZV9S9GTB3VWTLDRG', '3SSN80MU8COOT5RHA81VM6K9HJ3KXW']

hits = [h1, h2, h3, h4, h5]

test_group = read_pickle("test_group")
control_group = read_pickle("control_group")

my_dicts = []
for hit_list in hits:
    my_dicts.append(create_dictionary_from_hits_train(hit_list, bad_qualifications=["3BERVBF4P827RP0FK9ANECYHIUQYP7"],
                                                      ignore_workers=test_group))
counts = [[0 for _ in range(NUM_CLASSES)] for _ in range(len(hits))]  # how many ...
for index in range(len(counts)):
    num_entries = len(my_dicts[index])
    class_counter = [0 for _ in range(NUM_CLASSES)]
    for k in my_dicts[index]:
        for c in range(NUM_CLASSES):
            counts[index][c] += my_dicts[index][k][c]
        for c in my_dicts[index][k][-1]:
            class_counter[c] += 1
    counts[index] = [counts[index][c] / class_counter[c] * 100 if class_counter[c] != 0 else -1
                     for c in range(NUM_CLASSES)]

avgs = []
for index in range(len(counts)):
    num_classes = 0
    my_sum = 0
    for c in counts[index]:
        if c != -1:
            my_sum += c
            num_classes += 1
    print(f"iteration {index}: overall average: {my_sum / num_classes}")
    avgs.append(my_sum / num_classes)

plt.style.use("seaborn")
plt.figure(figsize=(12, 5))
legends = []
for index in range(len(counts)):
    plt.errorbar(list(number_to_class.values()), counts[index], marker="^", linestyle="None")
    legends.append(f"training iteration {index + 1}: avg = {avgs[index]:.2f}")
plt.xticks(rotation=45)
plt.ylim([-3, 103])
plt.yticks(list(range(0, 101, 10)))
plt.ylabel("percentage of correct answers per error class")
plt.legend(legends, loc="lower left")
plt.savefig("training_progress_control.png")
plt.show()
