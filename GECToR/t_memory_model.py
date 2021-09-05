"""
Copyright (c) 2016 Duolingo Inc. Half-Life Regression. MIT Licence.  [adapted by Yoel Zweig]
"""

import math
import random
import time
# import sys

from collections import defaultdict

from t_students import NUM_ERROR_CLASSES
from t_student_session import StudentSession
from t_helper import mae

MIN_HALF_LIFE = 1.0 / (24 * 6)  # 10 minutes (in days)
MAX_HALF_LIFE = 90              # 3 months (in days)
LN2 = math.log(2.)


class SpacedRepetitionModel:
    """
    Spaced repetition model
      - 'hlr' (half-life regression) with lexeme tags
      - 'leitner'
    """
    def __init__(self, method="leitner", initial_weights=None, lrate=.001, hl_wt=.01, l2_wt=.1, log=False):
        self.method = method
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate  # learning rate
        self.hlwt = hl_wt  # h loss weight
        self.l2wt = l2_wt  # l2 loss weight
        self.log = log

    def __repr__(self):
        return f"SpacedRepetitionModel({self.method}, log={self.log})"

    def halflife(self, fv, base):
        try:
            # base(2)^{\Theta \cdot fv} (weights[k] = 0 for first run (defaultdict))
            e = sum([self.weights[k] * x_k for k, x_k in enumerate(fv)])
            return hl_clip(base ** e)
        except OverflowError:
            # sys.stderr.write("Exception in halflife()")
            return MAX_HALF_LIFE

    def predict(self, lag_time, student, error_class, base=2., session=None):  # session for training, lag_time in days
        if session is None:
            latest_session = student.history[student.history[error_class][0]]
            # fv = feature vector (#all past correct responses, #all past incorrect responses, bias=1, lexeme tags)
            fv = [latest_session["#c_total"], latest_session["#i_total"], 1]
        else:
            fv = [session["#c_total"], session["#i_total"], 1]
        for e_c in range(NUM_ERROR_CLASSES):
            if e_c != error_class:
                fv.append(0)
            else:
                fv.append(1)
        if self.method == 'hlr':
            try:
                # base(2)^{-\Delta / h}
                h = self.halflife(fv, base)
                p = p_clip(base ** (-lag_time / h))
                return p, h
            except OverflowError:
                # sys.stderr.write("Exception in predict(), 'hlr'")
                return 0.9999, MAX_HALF_LIFE
        elif self.method == 'leitner':
            try:
                h = hl_clip((base ** (fv[0] - fv[1]) * MIN_HALF_LIFE))  # initial (equilibrium) step 10 minutes
                p = p_clip(base ** (-lag_time / h))
                return p, h
            except OverflowError:
                # sys.stderr.write("Exception in predict(), 'leitner'")
                return 0.9999, MAX_HALF_LIFE
        else:
            raise Exception("unknown method")

    def train_update(self, lag_time, session):  # lag_time in days
        # fv = feature vector with (#all past correct responses, #all past incorrect responses, bias=1, lexeme tags)
        fv = [session["#c_total"], session["#i_total"], 1]
        for e_c in range(NUM_ERROR_CLASSES):
            if e_c != session["error_class"]:
                fv.append(0)
            else:
                fv.append(1)
        if self.method == "hlr":
            # % correct answers
            session_p = p_clip(session["#c_session"] / (session["#c_session"] + session["#i_session"]))
            session_h = hl_clip(-lag_time / math.log(session_p, 2))  # estimate "true half-life" from p
            p, h = self.predict(lag_time, -1, session["error_class"], session=session)  # -1 placeholder for student
            # p-loss-term dw
            dlp_dw = 2 * (p - session_p) * (LN2 ** 2) * p * (lag_time / h)
            # h-loss-term dw
            dlh_dw = 2 * (h - session_h) * LN2 * h
            for k, x_k in enumerate(fv):
                rate = (1 / (1 + session_p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                self.weights[k] -= rate * dlp_dw * x_k
                self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                self.weights[k] -= rate * self.l2wt * self.weights[k]
                # increment feature count for learning rate
                self.fcounts[k] += 1
            return session_p, p

    def train(self, dictionary):
        sessions = []
        for student in dictionary.values():  # all student histories
            sessions.extend(student.history[29:])
        random.shuffle(sessions)

        if self.log:
            print(f"Number of sessions for training: {len(sessions)}", end="\n\n")
        true_p = []
        predicted_p = []
        for i, session in enumerate(sessions):
            if not session["prev_stop_time"] == -1:  # else skip
                lag_time = session["start_time"] - session["prev_stop_time"]
                lag_time = lag_time / (24 * 3600)  # seconds to days
                session_p, p = self.train_update(lag_time, session)  # p is predicted
                if self.log:
                    true_p.append(session_p)
                    predicted_p.append(p)
                    if not i % 50 and i:  # i % 50 == 0 and i != 0
                        print(f"{i} of {len(sessions)} sessions processed...")
                        print(f"MAE(p) for last 50 sessions: {mae(true_p, predicted_p)}", end="\n\n")
                        true_p.clear()
                        predicted_p.clear()
        if self.log:
            print("\nTraining finished.", end="\n\n")

    def mean_absolute_error(self, dictionary):  # mean absolute error over all histories
        sessions = []
        for student in dictionary.values():  # all student histories
            sessions.extend(student.history[29:])

        true_p = []
        predicted_p = []
        for session in sessions:
            if not session["prev_stop_time"] == -1:  # else skip
                session_p = session["#c_session"] / (session["#c_session"] + session["#i_session"])
                lag_time = session["start_time"] - session["prev_stop_time"]
                lag_time = lag_time / (24 * 3600)  # seconds to days
                p, _ = self.predict(lag_time, None, session["error_class"], session=session)  # None → student
                true_p.append(session_p)
                predicted_p.append(p)

        return mae(true_p, predicted_p)  # mean absolute error

    def time_for_next_session(self, student):
        error_class_order = list(range(NUM_ERROR_CLASSES))
        random.shuffle(error_class_order)
        for error_class in error_class_order:
            if student.history[error_class][0] == -1:  # no session yet
                if self.log:
                    print(f"first session for error class {error_class}")
                return 0  # now

        if self.log:
            print("all error classes have at least one session")

        # choose error_class with lowest prob
        prob, t_minus = 1, -1
        for error_class in error_class_order:
            lag_time = time.time() - student.history[student.history[error_class][0]]["stop_time"]
            lag_time = lag_time / (24 * 3600)  # seconds to days
            p, h = self.predict(lag_time, student, error_class)
            if self.log:
                print(f"error_class={str(error_class) + ',':<6}", f"_p={str(round(p, 6)) + ',':<12}",
                      f"_h={str(round(h, 6)) + ',':<12}", f"lag_time={lag_time}", sep="")
            if p <= 0.5:
                return 0  # now
            elif p < prob:
                prob = p
                t_minus = (h - lag_time) * 24 * 3600  # days to seconds
        return t_minus

    def pick_next_error_class(self, student):
        error_class_order = list(range(NUM_ERROR_CLASSES))
        random.shuffle(error_class_order)
        for error_class in error_class_order:
            if student.history[error_class][0] == -1:  # no session yet
                return error_class

        # choose error_class with lowest prob
        prob, lowest_prob_error_class = 1, -1
        for error_class in range(NUM_ERROR_CLASSES):
            lag_time = time.time() - student.history[student.history[error_class][0]]["stop_time"]
            lag_time = lag_time / (24 * 3600)  # seconds to days
            p, h = self.predict(lag_time, student, error_class)
            if p < prob:
                prob = p
                lowest_prob_error_class = error_class
        return lowest_prob_error_class

    def build_study_session(self, student, error_class=None):
        error_class = error_class if error_class else self.pick_next_error_class(student)
        return StudentSession(error_class, student)


def hl_clip(h):
    # bound half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def p_clip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def print_session_prediction(model, session):
    if not session["prev_stop_time"] == -1:  # else skip
        session_p = session["#c_session"] / (session["#c_session"] + session["#i_session"])
        lag_time_seconds = session["start_time"] - session["prev_stop_time"]
        lag_time_days = lag_time_seconds / (24 * 3600)  # seconds to days
        p, _ = model.predict(lag_time_days, None, session["error_class"], session=session)  # None → student
        print("-"*10)
        print(f"p = {p}, true_p = {session_p}, error_class: {session['error_class']}, "
              f"#c_total: {session['#c_total']}, #i_total: {session['#i_total']}, "
              f"lag time days: {lag_time_days}, lag time seconds: {lag_time_seconds}")
        print("-"*10)
    else:
        print('No prediction if session["prev_stop_time"] == -1')
