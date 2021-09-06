import pickle
import math
import time


NUM_CLASSES = 8
NUM_SENTENCES_PER_CLASS_TEST = 5
NUM_SENTENCES_PER_CLASS_TRAIN = 60

MIN_HALF_LIFE = 0.98  # (days)
MAX_HALF_LIFE = 7  # (days)
LN2 = math.log(2.)


def read_pickle(name="z_student_dict"):
    with open(name, "rb") as file:
        return pickle.load(file)


def write_pickle(data, name="z_student_dict"):
    with open(name, "wb") as file:
        pickle.dump(data, file)


def get_mturker_dictionary_entry():
    # [0] => #c_total, [1] => #i_total, [2] => [seen already indices], [3] => time (seconds) last practice session
    return {"casing": [0, 0, [], 0], "singular": [0, 0, [], 0], "plural": [0, 0, [], 0], "vb_vbz": [0, 0, [], 0],
            "vbz_vb": [0, 0, [], 0], "vb__vbn": [0, 0, [], 0], "vb__vbg": [0, 0, [], 0], "other": [0, 0, [], 0]}


class SpacedRepetitionModel:
    """
    Spaced repetition model
      - 'leitner'
    """
    def __init__(self, method="Leitner"):
        self.method = method

    def __repr__(self):
        return f"SpacedRepetitionModel({self.method})"

    def predict(self, student_dict, error_class, base=2.):
        fv = [student_dict[error_class][0], student_dict[error_class][1]]  # [0] => #c_total, [1] => #i_total
        lag_time = time.time() - student_dict[error_class][3]  # seconds since last practice session
        lag_time = lag_time / (24 * 3600)  # seconds to days
        if self.method == 'Leitner':
            try:
                h = hl_clip((base ** (fv[0] - fv[1]) * MIN_HALF_LIFE))  # bias => half-life
                p = p_clip(base ** (-lag_time / h))
                return p, h
            except OverflowError:
                return 0.9999, MAX_HALF_LIFE
        else:
            raise Exception("unknown method")


def hl_clip(h):
    # bound half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def p_clip(p):
    # bound min/max model predictions
    return min(max(p, 0.0001), .9999)
