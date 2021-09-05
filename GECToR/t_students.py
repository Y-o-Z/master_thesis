NUM_ERROR_CLASSES = 29


class Student:
    def __init__(self, uid, name, history=None, iq=0):  # no mutable default arguments, 0..5
        self.uid = uid
        self.name = name
        self.history = history if history else [[-1, 0] for _ in range(NUM_ERROR_CLASSES)]
        self.iq = min(max(iq, 0), 5)

    def __repr__(self):
        return f"Student({self.uid}, {self.name}, [history], {self.iq})"

    def print_history(self, only_sessions=False):
        if not only_sessions:
            print(self.history)
        else:
            print(self.history[29:])

    def reset(self):
        self.history = [[-1, 0] for _ in range(NUM_ERROR_CLASSES)]


def add_student(student_dict, uid, name, history=None, iq=0):
    if uid not in student_dict:
        student_dict[uid] = Student(uid, name, history, iq)
