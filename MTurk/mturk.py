# HTTPStatusCode 200 => success  (=> initial, main, final)

import boto3  # https://blog.mturk.com/
import csv
import datetime  # is used

from new_create_survey import create_training_survey
from helper import read_pickle


# HIT: human intelligence task → HTML → XML (customizable)
# assignment: one worker on one hit

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
                     endpoint_url=MTURK_SANDBOX  # leave out endpoint_url to connect to live MTurk
                     )

if "sandbox" in str(mturk._endpoint):
    print(f"I have ${mturk.get_account_balance()['AvailableBalance']} in my Sandbox account .")
else:
    print(f"I have ${mturk.get_account_balance()['AvailableBalance']} in my real account .")

# ---

# initial_survey = open("initial_survey.xml").read()
final_survey = open("final_survey.xml").read()

# initial_name = "Language Learning [Initial HIT - LOI approx. 15 minutes]"
final_name = "Language Learning [Final HIT - LOI approx. 15 minutes]"
name = "Language Learning [Main HIT(s) - LOI approx. 5 minutes]"

# ---

# REQUIREMENTS
# Israel: => IL, India: => IN (..), + not already participating, + not blacklisted, ..
"""
requirements_initial = [{
    'QualificationTypeId': '00000000000000000071',  # location
    'Comparator': 'In',
    'LocaleValues': [{
        'Country': 'IN'
    }],
    "RequiredToPreview": True  # => "ActionsGuarded": ...
}, {
    'QualificationTypeId': '35OXBT565DPUVO44D0CRB0MUP3WV28',  # already participating
    'Comparator': 'DoesNotExist',
    'RequiredToPreview': True
}, {
    'QualificationTypeId': '3LQV637WQB434BPZVLXU019Z2GJ6BV',  # blacklist
    'Comparator': 'DoesNotExist',
    'RequiredToPreview': True
}]
# """

# """
not_english_requirements = [{
    'QualificationTypeId': '00000000000000000071',  # location
    'Comparator': 'NotIn',
    'LocaleValues': [{
        'Country': 'US',
    }, {
        'Country': 'CA',
    }, {
        'Country': 'GB'
    }, {
        'Country': 'AU'
    }, {
        'Country': 'NZ'
    }],
    "RequiredToPreview": True  # => "ActionsGuarded": ...
}, {
    'QualificationTypeId': '35OXBT565DPUVO44D0CRB0MUP3WV28',  # already participating
    'Comparator': 'DoesNotExist',
    'RequiredToPreview': True
}, {
    'QualificationTypeId': '3Y0FER39339HGTDP2LJDZH59MERXK0',  # already participating 2
    'Comparator': 'DoesNotExist',
    'RequiredToPreview': True
}, {
    'QualificationTypeId': '3LQV637WQB434BPZVLXU019Z2GJ6BV',  # blacklist
    'Comparator': 'DoesNotExist',
    'RequiredToPreview': True
}, {
    'QualificationTypeId': '00000000000000000040',  # number of HITs approved
    'Comparator': 'GreaterThan',
    'IntegerValues': [101],
    'RequiredToPreview': True
}, {
    'QualificationTypeId': '000000000000000000L0',  # percentage of assignments approved
    'Comparator': 'GreaterThanOrEqualTo',
    'IntegerValues': [95],
    'RequiredToPreview': True
}]
# """

"""
requirements_final_1 = [{
    'QualificationTypeId': '35OXBT565DPUVO44D0CRB0MUP3WV28',  # already participating
    'Comparator': 'Exists',
    'RequiredToPreview': True
}, {
    'QualificationTypeId': '3BERVBF4P827RP0FK9ANECYHIUQYP7',  # "expelled"
    'Comparator': 'DoesNotExist',
    'RequiredToPreview': True
}]
# """

# creating qualifications  !! + MaxResults=100 (++) !!
# + list_hits_for_qualification_type, list_workers_with_qualification_type, disassociate_qualification_from_worker, ..
"""
mturk.create_qualification_type(Name="...", Keywords="Me", Description="Ich", QualificationTypeStatus="Active")
mturk.list_qualification_types(MustBeRequestable=False, MustBeOwnedByCaller=True)
mturk.delete_qualification_type(QualificationTypeId="...")
mturk.associate_qualification_with_worker(QualificationTypeId="...", WorkerId="A3MPEJAS5SSFE6", (\ ?)
                                          IntegerValue=1, SendNotification=False)  # => me
# """

# ---

mturker_dictionary = read_pickle("mturker_dictionary")
test_group = read_pickle("test_group")
control_group = read_pickle("control_group")
base_group = read_pickle("base_group")
# all_workers = read_pickle("all_workers")

# r = ""

documentation = ""
local_blacklist = []  # expelled
lb = mturk.list_workers_with_qualification_type(QualificationTypeId="3A7LZ792EO9656B5AMLOSHTOXU0HQJ", MaxResults=100)
for w in lb["Qualifications"]:
    local_blacklist.append(w["WorkerId"])

# """ => initial, MAIN, final
message = "A new HIT has just been published for you. Please remember that you will only be invited to the final " \
          "language test if you complete your HITs on time. Thank you and all the best!"
my_qualifications = mturk.list_qualification_types(MustBeRequestable=False, MustBeOwnedByCaller=True, MaxResults=100)
my_qualifications = my_qualifications["QualificationTypes"]
for i, worker in enumerate(test_group):
    skip_worker_1 = False
    skip_worker_2 = False
    skip_worker_3 = False
    if worker in local_blacklist:
        print(f"skip {worker} (test)")
        skip_worker_1 = True
    if not skip_worker_1:
        qualification_id = ""
        for q in my_qualifications:
            if q["Name"] == worker:
                qualification_id = q["QualificationTypeId"]
        requirements_local = [{
            'QualificationTypeId': qualification_id,  # personal qualification
            'Comparator': 'Exists',
            'RequiredToPreview': True
        }]
        assert qualification_id != ""
        doc, num = create_training_survey(mturker_dictionary[worker])
        if doc == "":
            print(f"\n'continued' for workers {worker} (test), {control_group[i]} (control), {base_group[i]} (base)")
            continue
        """ DONE
        r = new_hit = mturk.create_hit(
            Title=name,
            Description="Find and correct all grammatical errors.",
            Keywords='learning, english, sentence, grammar, error, correction, continuation, study',
            Reward='0.5',  # 0.01 = 1 cent (minimum)
            MaxAssignments=1,  # 100, see above (cheaper if <= 9 (+ 20% vs + 40%))
            LifetimeInSeconds=72000,  # 20 hours
            AssignmentDurationInSeconds=3600,  # 1 hour
            AutoApprovalDelayInSeconds=172800,  # 2 days
            RequesterAnnotation="main_hit(s)_2_9",
            QualificationRequirements=requirements_local,
            # UniqueRequestToken="123",  # must be unique (24h, also for deleted surveys)
            Question=doc  # initial_survey
        )
        mturk.notify_workers(Subject="New HIT published", MessageText=message, WorkerIds=[worker])
        # """
        # documentation += f"{r['HIT']['HITId']}: {worker}, "
        print(f"\ntest worker = {worker}")
    # ----------
    worker_2 = control_group[i]
    if worker_2 in local_blacklist:
        print(f"skip {worker_2} (control)")
        skip_worker_2 = True
    if not skip_worker_2:
        qualification_id = ""
        for q in my_qualifications:
            if q["Name"] == worker_2:
                qualification_id = q["QualificationTypeId"]
        requirements_local = [{
            'QualificationTypeId': qualification_id,  # personal qualification
            'Comparator': 'Exists',
            'RequiredToPreview': True
        }]
        assert qualification_id != ""
        if skip_worker_1:
            num = 3
        doc_2, _ = create_training_survey(mturker_dictionary[worker_2], is_random=True, num_error_classes=num)
        """ DONE
        r = new_hit = mturk.create_hit(
            Title=name,
            Description="Find and correct all grammatical errors.",
            Keywords='learning, english, sentence, grammar, error, correction, continuation, study',
            Reward='0.5',  # 0.01 = 1 cent (minimum)
            MaxAssignments=1,  # 100, see above (cheaper if <= 9 (+ 20% vs + 40%))
            LifetimeInSeconds=72000,  # 20 hours
            AssignmentDurationInSeconds=3600,  # 1 hour
            AutoApprovalDelayInSeconds=172800,  # 2 days
            RequesterAnnotation="main_hit(s)_2_9",
            QualificationRequirements=requirements_local,
            # UniqueRequestToken="123",  # must be unique (24h, also for deleted surveys)
            Question=doc_2  # initial_survey
        )
        mturk.notify_workers(Subject="New HIT published", MessageText=message, WorkerIds=[worker_2])
        # """
        # documentation += f"{r['HIT']['HITId']}: {worker_2}, "
        print(f"\ncontrol worker = {worker_2}")
        # ----------
    worker_3 = base_group[i]
    if worker_3 in local_blacklist:
        print(f"skip {worker_3} (base)")
        skip_worker_3 = True
    if not skip_worker_3:
        qualification_id = ""
        for q in my_qualifications:
            if q["Name"] == worker_3:
                qualification_id = q["QualificationTypeId"]
        requirements_local = [{
            'QualificationTypeId': qualification_id,  # personal qualification
            'Comparator': 'Exists',
            'RequiredToPreview': True
        }]
        assert qualification_id != ""
        if skip_worker_1:
            num = 3
        doc_3, _ = create_training_survey(mturker_dictionary[worker_3], is_random=True, num_error_classes=num,
                                          no_hints=True)
        """ DONE
        r = new_hit = mturk.create_hit(
            Title=name,
            Description="Find and correct all grammatical errors.",
            Keywords='learning, english, sentence, grammar, error, correction, continuation, study',
            Reward='0.5',  # 0.01 = 1 cent (minimum)
            MaxAssignments=1,  # 100, see above (cheaper if <= 9 (+ 20% vs + 40%))
            LifetimeInSeconds=72000,  # 20 hours
            AssignmentDurationInSeconds=3600,  # 1 hour
            AutoApprovalDelayInSeconds=172800,  # 2 days
            RequesterAnnotation="main_hit(s)_2_9",
            QualificationRequirements=requirements_local,
            # UniqueRequestToken="123",  # must be unique (24h, also for deleted surveys)
            Question=doc_3  # initial_survey
        )
        mturk.notify_workers(Subject="New HIT published", MessageText=message, WorkerIds=[worker_3])
        # """
        # documentation += f"{r['HIT']['HITId']}: {worker_3}, "
        print(f"\nbase worker = {worker_3}")
# """


documentation = ""
all_workers = read_pickle("all_workers")
local_blacklist = []  # expelled
lb = mturk.list_workers_with_qualification_type(QualificationTypeId="3A7LZ792EO9656B5AMLOSHTOXU0HQJ", MaxResults=100)
for w in lb["Qualifications"]:
    local_blacklist.append(w["WorkerId"])

message = "The final HIT has just been published for you. Please do it carefully. All the best and have a great day!"
my_qualifications = mturk.list_qualification_types(MustBeRequestable=False, MustBeOwnedByCaller=True, MaxResults=100)
my_qualifications = my_qualifications["QualificationTypes"]

# """
for worker in all_workers[:]:
    if worker in local_blacklist:
        print(f"skip {worker}")
        continue
    qualification_id = ""
    for q in my_qualifications:
        if q["Name"] == worker:
            qualification_id = q["QualificationTypeId"]
    requirements_local = [{
        'QualificationTypeId': qualification_id,  # personal qualification
        'Comparator': 'Exists',
        'RequiredToPreview': True
    }]
    # """
    r = new_hit = mturk.create_hit(
        Title=final_name,
        Description="Find and correct all grammatical errors.",
        Keywords='learning, english, sentence, grammar, error, correction, continuation, study',
        Reward='4',  # 0.01 = 1 cent (minimum)
        MaxAssignments=1,  # 100, see above (cheaper if <= 9 (+ 20% vs + 40%))
        LifetimeInSeconds=86400,  # 1 day
        AssignmentDurationInSeconds=7200,  # 2 hours
        AutoApprovalDelayInSeconds=172800,  # 2 days
        RequesterAnnotation="FINAL",
        QualificationRequirements=requirements_local,
        # UniqueRequestToken="123",  # must be unique (24h, also for deleted surveys)
        Question=final_survey
    )
    # """
    documentation += f"{r['HIT']['HITId']}: {worker}, "
    mturk.notify_workers(Subject="Final HIT published", MessageText=message, WorkerIds=[worker])
# """

print("A new HIT has been created. You can view it here:")
# sandbox
# print(f"https://workersandbox.mturk.com/mturk/preview?groupId={new_hit['HIT']['HITGroupId']}", end="\n\n")
# marketplace
# print(f"https://worker.mturk.com/mturk/preview?groupId={new_hit['HIT']['HITGroupId']}", end="\n\n")

# ---

# + (useful) boto3 python commands
# list_hits
my_hits = mturk.list_hits(MaxResults=100); print(f"NumResults: {my_hits['NumResults']}")
# print(my_hits)  # ["HITs"][i] (newest first)
print(f"Title: {my_hits['HITs'][0]['Title']}, \nHITID: {my_hits['HITs'][0]['HITId']}, "
      f"Status: {my_hits['HITs'][0]['HITStatus']}, Reward: {my_hits['HITs'][0]['Reward']}$")

# list_reviewable_hits (vs "reviewing")
my_hits_2 = mturk.list_reviewable_hits(MaxResults=100); print(f"NumResults: {my_hits_2['NumResults']}")


# get_hit
hit = mturk.get_hit(HITId="...")
print(hit["HIT"]["HITStatus"])
# print(hit["HIT"]["HITReviewStatus"])  # => internal auditing

# delete_hit (+ update_expiration_for_hit (+ update_hit_review_status))
"""
for index in range(my_hits["NumResults"]):
    r1 = mturk.update_expiration_for_hit(HITId=my_hits["HITs"][index]["HITId"], ExpireAt=datetime.datetime(2015, 1, 1))
    r2 = mturk.delete_hit(HITId=my_hits["HITs"][index]["HITId"])
    print(r1, r2, sep="\n")
# """

# reject_assignment
"""
for index in range(my_hits["NumResults"]):
    assignments = mturk.list_assignments_for_hit(HITId=my_hits["HITs"][index]["HITId"])
    for assignment in assignments["Assignments"]:
        r = mturk.reject_assignment(AssignmentId=assignment["AssignmentId"], RequesterFeedback="NEF!")
        print(r)
"""

# approve_assignment
"""
mturk.approve_assignment(AssignmentId="...", RequesterFeedback="Thanks!")
# """

# notify_workers
"""
response = mturk.notify_workers(Subject="123", MessageText="456", WorkerIds=['789'])
# """

# ..
