import argparse
import sys

from t_memory_model import SpacedRepetitionModel
from t_helper import human_readable_time, read_pickle, write_pickle


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Automated English Grammar Learning')

    methods = ["hlr", "leitner"]
    argparser.add_argument("--model", "-m",
                           nargs="?",  # one argument or default
                           choices=methods,
                           default='leitner',
                           help="memory models [hlr, leitner]")
    argparser.add_argument("--log", "-l",
                           action="store_true",
                           help="print additional information")
    argparser.add_argument("--simul", "-s",
                           action="store_true",
                           help="simulate student learning")
    # uid is a positional argument
    argparser.add_argument("uid",
                           type=int,
                           help="user identifier")

    args = argparser.parse_args()

    # load student data  (c for correct, i for incorrect)
    # student_dict[uid=key] → [error_classes → [index_last_session, session_count]],
    # session1, session2, ... → {error_class, #c_total, #i_total, #c_session, #i_session,
    #                           start_time, stop_time, prev_stop_time}
    student_dict = read_pickle()

    if args.uid in student_dict:
        student = student_dict[args.uid]
    else:
        sys.stderr.write(f"user id = {args.uid}\n")
        sys.exit("user id not recognized")

    if args.log:
        print(f"\nmethod = {args.model}")  # % ...
        print(f"user id = {args.uid}")

    model = SpacedRepetitionModel(args.model, log=args.log)

    print(f"\nWelcome user {student.name} ({student.uid}).", end="\n\n")
    _ = input("Press 'Enter' to start.\n")

    # shell training interface (run with)
    # 'discontinued' => MTurk survey

    while True:
        session_time = model.time_for_next_session(student)
        if session_time != 0:
            print("Please come back later.")
            print(f"Next exercise session is ready in {human_readable_time(session_time)}.")
            break

        study_session = model.build_study_session(student)
        if not args.simul:
            study_session.run()
        else:
            study_session.run_simulated()
        val = input("If you would like to continue, please type 'yes': ")
        if val == "log":
            print(student.history[-1])
            val = input("'yes'? ")
        if val == "simul":
            if not args.simul:
                print("Switching to simulation mode", end="\n\n")
                args.simul = True
            else:
                print("Already in simulation mode", end="\n\n")
            val = input("'yes'? ")
        elif val == "normal":
            if args.simul:
                print("Switching to normal mode", end="\n\n")
                args.simul = False
            else:
                print("Already in normal mode", end="\n\n")
            val = input("'yes'? ")
        if val.lower() not in ["yes", "'yes'", '"yes"', "y"]:
            break

    # store student data
    write_pickle(student_dict)


# ------------------------------
"""

# simulating student data for HLR evaluation => see 'memory_models.ipynb' in folder => review/memory models/

import random

from t_students import add_student, NUM_ERROR_CLASSES

training_dict = {}  # read_pickle("...")
testing_dict = {}  # read_pickle("...")

# 300 * 6 simulated students, 300 per iq option (0..5) for training (2900 (100 * 29) sessions per student)
# 30 * 6 simulated students, 30 per iq option (0..5) for testing (2900 (100 * 29) sessions per student)

for i in range(300):
    for j in range(6):  # iq (noise variable)
        add_student(training_dict, 6 * i + j, str(6 * i + j), None, j)

for i in range(30):
    for j in range(6):  # iq (noise variable)
        add_student(testing_dict, 6 * i + j, str(6 * i + j), None, j)


def simulate_sessions(_model, _student):
    for _ in range(100):  # 100 sessions per student per error_class
        for _error_class in range(NUM_ERROR_CLASSES):
            _study_session = _model.build_study_session(_student, _error_class)
            if random.random() <= 0.8:  # short time
                lag_time = random.random()
            else:  # long time
                lag_time = random.randint(1, 30)
            _study_session.run_simulated(lag_time=lag_time)


model_leitner = SpacedRepetitionModel()
model_hlr = SpacedRepetitionModel(method="hlr")
for index in range(len(training_dict)):
    simulate_sessions(model_hlr, training_dict[index])
for index in range(len(testing_dict)):
    simulate_sessions(model_hlr, testing_dict[index])

# write_pickle(training_dict, "z_training_dict")
# write_pickle(testing_dict, "z_testing_dict")

print(model_leitner.mean_absolute_error(testing_dict))  # 0.0655 (continually high predictions towards the end?!)

print(model_hlr.mean_absolute_error(testing_dict))  # 0.3995 (pre-training)
model_hlr.train(training_dict)
print(model_hlr.mean_absolute_error(testing_dict))  # 0.0482, (post-training) (it 1)

# write_pickle(model_hlr, name="z_s_trained_hlr")

"""
