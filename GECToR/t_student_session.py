import random
import time

from t_helper import binomial, read_pickle

NUM_EXERCISES_PER_SESSION = 10  # same error class

RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


class StudentSession:
    def __init__(self, error_class, student):
        self.error_class = error_class
        self.student = student

    def __repr__(self):
        return f"StudentSession({self.error_class}, {self.student})"

    def run_simulated(self, lag_time=-1):  # lag time in days
        # => patchwork function
        index = self.student.history[self.error_class][0]
        # session_count = self.student.history[self.error_class][1]

        # print(f"running session {session_count} for error class {self.error_class} [simulated]", end="\n\n")

        if index == -1:
            # no past session for self.error_class
            new_session = {"error_class": self.error_class,
                           "#c_total": 0, "#i_total": 0,
                           "#c_session": 0, "#i_session": 0,
                           "start_time": -1, "stop_time": -1,
                           "prev_stop_time": -1}
        else:
            # read last session for self.error_class
            new_session = self.student.history[index].copy()  # shallow copy
            new_session["prev_stop_time"] = new_session["stop_time"]  # only if index != -1

        # learning → (- lag time + c_total + i_total - error_class (simulate difficulty) + bias + iq)
        if new_session["prev_stop_time"] == -1:  # first session
            lag_time = 30  # base case (MAX_HALF_LIFE)
            new_session["start_time"] = time.time()
        elif lag_time == -1:  # no parameter present
            lag_time = (time.time() - new_session["prev_stop_time"]) / (24 * 3600)  # seconds to days
            new_session["start_time"] = time.time()
        else:
            new_session["start_time"] = new_session["prev_stop_time"] + (lag_time * 24 * 3600)  # days to seconds

        p = 2 ** (-lag_time / 2 ** (0.1 * new_session["#c_total"] + 0.05 * new_session["#i_total"]
                                    - 0.2 * self.error_class + 0.5 + 0.1 * self.student.iq))

        num_correct = binomial(NUM_EXERCISES_PER_SESSION, p)
        num_incorrect = NUM_EXERCISES_PER_SESSION - num_correct

        new_session["#c_total"] += num_correct
        new_session["#i_total"] += num_incorrect
        new_session["#c_session"] = num_correct
        new_session["#i_session"] = num_incorrect
        new_session["stop_time"] = new_session["start_time"] + 1

        self.student.history.append(new_session)
        self.student.history[self.error_class][0] = len(self.student.history) - 1  # change pointer to newest session
        self.student.history[self.error_class][1] += 1  # increment session count

    def run(self):
        index = self.student.history[self.error_class][0]
        # session_count = self.student.history[self.error_class][1]

        # print(f"running session {session_count} for exercise {self.error_class}", end="\n\n")

        if index == -1:
            # no past session for self.error_class
            new_session = {"error_class": self.error_class,
                           "#c_total": 0, "#i_total": 0,
                           "#c_session": 0, "#i_session": 0,
                           "start_time": -1, "stop_time": -1,
                           "prev_stop_time": -1}
        else:
            # read last session for self.error_class
            new_session = self.student.history[index].copy()
            new_session["#c_session"] = 0
            new_session["#i_session"] = 0
            new_session["prev_stop_time"] = self.student.history[index]["stop_time"]  # only if index != -1

        # triplet → preds, one_error_preds, error_dicts{error_class, preds_indices, one_error_preds_indices}
        triplets = get_sentences_from_error_class(self.error_class, NUM_EXERCISES_PER_SESSION)

        new_session["start_time"] = time.time()

        for triplet in triplets:
            # print(f"{i + 1} / {NUM_EXERCISES_PER_SESSION}:")
            run_triplet(triplet, new_session, log=False)

        new_session["stop_time"] = time.time()

        self.student.history.append(new_session)
        self.student.history[self.error_class][0] = len(self.student.history) - 1  # change pointer to newest session
        self.student.history[self.error_class][1] += 1  # increment session count


def run_triplet(triplet, session, log=False):
    # triplet → preds, one_error_preds, error_dicts {error_class, preds_indices, one_error_preds_indices}
    print("Rewrite the following sentence correctly:")
    answer = input(triplet[1] + "\n")
    if answer == triplet[0]:
        if False:  # num_corrections == 0:
            print("Very good!", end="\n\n")  # ..
        else:
            s = ""  # "s" if num_corrections > 1 else ""
            print(f"Very good! You caught the error{s}.", end="\n\n")
        session["#c_total"] += 1
        session["#c_session"] += 1
    else:
        if False:  # num_corrections == 0:
            print("The sentence was already correct. You'll get it next time \\(^^)/", end="\n\n")  # ..
        else:
            print("That wasn't quite right. You'll get it next time \\(^^)/", end="\n\n")
            # print feedback
            correct_sentence_tokens = triplet[0].split()
            error_sentence_tokens = triplet[1].split()
            feedback = ""
            i1, j1 = triplet[2]["preds_indices"]
            i2, j2 = triplet[2]["one_error_preds_indices"]
            feedback += " ".join(correct_sentence_tokens[:i1])
            feedback += f" {RED}{{"  # {{ → escape {
            feedback += " ".join(error_sentence_tokens[i2:j2 + 1])
            feedback += f"}}{RESET} → {BLUE}{{"
            feedback += " ".join(correct_sentence_tokens[i1:j1 + 1])
            feedback += f"}}{RESET} "
            feedback += " ".join(correct_sentence_tokens[j1 + 1:])
            print("## SOLUTION ##", feedback, "#" * 14, sep="\n", end="\n\n")
        session["#i_total"] += 1
        session["#i_session"] += 1
    if log:
        print("#" * 7, f"correct_sentence: {triplet[0]}", f"error_sentence: {triplet[1]}", f"dict: {triplet[2]}",
              "#" * 7, sep="\n", end="\n\n")
    input("\n")


def get_sentences_from_error_class(exercise, num_sentences):
    name = f"datasets for MTurk/fce_by_error_class/list_fce_{exercise}"  # path
    sentences = read_pickle(name)
    # print(f"number of sentences in error class {exercise} = {len(sentences)}")
    return random.sample(sentences, min(len(sentences), num_sentences))  # no duplicate picks
