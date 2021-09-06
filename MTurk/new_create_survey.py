import random

from helper import read_pickle, SpacedRepetitionModel, NUM_CLASSES, NUM_SENTENCES_PER_CLASS_TRAIN, \
    NUM_SENTENCES_PER_CLASS_TEST


# [0] → correct, [1] → error, [2] → solution(, [3] → identifier)
folder = "../prototype/datasets for MTurk/error_types_combined_data/"

class_casing = read_pickle(folder + "casing_errors_train")
class_singular = read_pickle(folder + "singular_errors_train")
class_plural = read_pickle(folder + "plural_errors_train")
class_vb_vbz = read_pickle(folder + "vb_vbz_errors_train")
class_vbz_vb = read_pickle(folder + "vbz_vb_errors_train")
class_vb__vbn = read_pickle(folder + "vb_vbn_errors_train")
class_vb__vbg = read_pickle(folder + "vb_vbg_errors_train")
class_other = read_pickle(folder + "other_verb_errors_train")

number_to_class = {0: "casing", 1: "singular", 2: "plural", 3: "vb_vbz", 4: "vbz_vb", 5: "vb__vbn", 6: "vb__vbg",
                   7: "other"}

sentences = []
for i, suffix in enumerate(number_to_class.values()):
    tmp = []  # => silence warning
    exec(f"tmp = class_{suffix}")
    for index in range(NUM_SENTENCES_PER_CLASS_TRAIN):
        tmp[index].append(f"{i}_{index}")  # class, sentence number [3]
    sentences.extend(tmp)

# styling: bash => html
for index in range(NUM_CLASSES * NUM_SENTENCES_PER_CLASS_TRAIN):
    sentences[index][2] = sentences[index][2].replace("\x1b[91m{", "<span style='color:red'>")
    sentences[index][2] = sentences[index][2].replace("\x1b[94m{", "<span style='color:blue'>")
    sentences[index][2] = sentences[index][2].replace("}\x1b[0m", "</span>")

model = SpacedRepetitionModel()

file = open("feedback_II", "r")
feedback = file.readlines()
feedback = "".join(feedback).split("%")

feedback_for_class = {}
for i, c in enumerate(number_to_class.values()):
    feedback_for_class[c] = feedback[i]
    feedback_for_class[c] = feedback_for_class[c].replace(f"&{i + 1}", f"class='help_{i}' style='display:none'")


def create_training_survey(student_dict, is_random=False, num_error_classes=0, no_hints=False):
    classes_to_train = []
    if not is_random:
        for key in student_dict:
            p, _ = model.predict(student_dict, key)
            if p <= 0.5:
                classes_to_train.append([key, p])
        classes_to_train = sorted(classes_to_train, key=lambda x: x[1])[:3]  # ascending, max 3
        classes_to_train = [c[0] for c in classes_to_train]  # drop p
    else:
        classes_to_train = random.sample(list(range(NUM_CLASSES)), k=num_error_classes)
        classes_to_train = [number_to_class[c] for c in classes_to_train]

    if not len(classes_to_train):  # == 0
        return "", 0

    num_sentences = len(classes_to_train) * NUM_SENTENCES_PER_CLASS_TEST
    num_pages = num_sentences + 3  # information, instructions, ..., submission

    # sentences from "classes_to_train" that have not yet been seen by the student
    total_survey_sentences = []
    survey_sentences = []
    for class_to_train in classes_to_train:  # correct class and not already seen
        total_survey_sentences.append([s for s in sentences if
                                       number_to_class[int(s[3].split("_")[0])] == class_to_train and
                                       int(s[3].split("_")[1]) not in
                                       student_dict[number_to_class[int(s[3].split("_")[0])]][2]])
    for index in range(len(total_survey_sentences)):  # pick (5) sentences at random from each class
        survey_sentences.extend(random.sample(total_survey_sentences[index], k=NUM_SENTENCES_PER_CLASS_TEST))
    random.shuffle(survey_sentences)

    c_0 = "class='0' style='display:block'"
    c_0_mt = "class = '0', style='display:block;margin-top:25px'"
    c_1 = "class='1' style='display:none'"
    c_1_mt = "class='1' style='display:none;margin-top:25px'"

    html = f"""
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
    <crowd-form answer-format="flatten-objects">
        
    <h2 {c_0}>Hi, welcome to today's language learning HIT.</h2>
        <p {c_0}>In order to enhance your learning experience, we provide you with a virtual assistant. 
        After trying on your own to solve an exercise, please don't forget to check the assistant's suggestion.<br>
        Nobody is perfect, so the assistant can also get it wrong from time to time &#x1F916</p>
        <p {c_0_mt}></p>
        <p {c_0}>As usual, you will be required to find and correct grammatical errors. 
        Most sentences will contain only a single error, but multiple or no errors are possible as well.</p>
        <p {c_0_mt}></p>
        <p {c_0}> Please solve all the grammatical exercises without relying on outside help and don't 
        forget to have fun!</p>
        <p {c_0_mt}></p>
        <p {c_0}><b>Thank you.</b></p>
        <p {c_0_mt}></p>
        <p {c_0}>I confirm to have read and understood the provided information: 
        <input type="checkbox" onchange="change_color()" id="checkbox"></p>
        <p {c_0_mt}></p>
        <p {c_0}>In case you wish to change the color of the survey background:</p>
        <p {c_0}><input type="checkbox" onchange="background_color('c1')" id="c1"> outer space</p>
        <p {c_0}><input type="checkbox" onchange="background_color('c2')" id="c2"> deep space sparkle</p>
        <p {c_0}><input type="checkbox" onchange="background_color('c3')" id="c3"> smoke</p> 
        <p {c_0}><input type="checkbox" onchange="random_background_color('c4')" id="c4"> random</p>
    
    <h2 {c_1}>Instructions</h2>
        <p {c_1}><b>Sentences may contain zero, one, or multiple errors. It's your task to find and  
        correct them.</b></p>
        <p {c_1_mt}></p>
        <p {c_1}><span style="background-color:yellow"><b>Note:</b></span> In order to move on to the next exercise, 
        you have to take a look at the assistant's suggestion. Be sure to finish an exercise before looking at 
        the suggestion, as you will not be allowed to change your answer afterwards. There is also a 
        short time lock that stops Turkers from rushing through the exercises.<br>
        It is important that you try to come up with your own solution. 
        Otherwise, you will not profit from these exercises.</p>
        <ul {c_1}>
            <li>Grammatical errors include, among other things, faulty verb forms, singular and plural errors, 
            and capitalization errors.<br>
            Bad punctuation or stylistically bad writing should not be counted as an error.</li>
            <li>If there are multiple ways to fix a sentence, choose the one that is closest to the original. 
            Also, try  to correct faulty words instead of adding or removing words.<br>
            If you are unsure, just go with you best guess or take a look at the hint. 
            After all, it's all about learning.</li>
        </ul>
        
        <h3 {c_1_mt}>Examples of good answers:</h3>
        <p {c_1_mt}></p>
        <p {c_1}>Sentence: If you don't <span style="color:red;font-weight:bold">agrees</span>, I will have to act.</p>
        <p {c_1}>Answer: If you don't <span style="color:blue;font-weight:bold">agree</span>, I will have to act.</p>
        <p {c_1}>Explanation: The correct conjugation of the verb agree for the second person singular is 
        <span style="color:blue;font-weight:bold">agree</span>. The form 
        <span style="color:red;font-weight:bold">agrees</span> would be correct for the third person singular.</p>
        <p {c_1_mt}></p>
        <p {c_1}>Sentence: She got married and had two <span style="color:red;font-weight:bold">childs</span>.</p>
        <p {c_1}>Answer: She got married and had two <span style="color:blue;font-weight:bold">children</span>.</p>
        <p {c_1}>Explanation: Child has the irregular plural form 
        <span style="color:blue;font-weight:bold">children</span>. The form 
        <span style="color:red;font-weight:bold">childs</span> does not exist.</p>
        <p {c_1_mt}></p>
        <p {c_1}>Sentence: Our <span style="color:red;font-weight:bold">Lives</span> are getting better and better.</p>
        <p {c_1}>Answer: Our <span style="color:blue;font-weight:bold">lives</span> are getting better and better.</p>
        <p {c_1}>Explanation: Nouns, if they are not at the beginning of a sentence, are generally not capitalized. 
        Therefore <span style="color:blue;font-weight:bold">lives</span> is correct and 
        <span style="color:red;font-weight:bold">Lives</span> is not.</p>
    """

    for i, sentence in enumerate(survey_sentences):
        html += f"<h3 class='{i + 2}' style='display:none'>Exercise {i + 1} of {num_sentences}</h3>"
        html += f"<p class='{i + 2}' style='display:none'><b>Fix the following sentence directly in the input box:" \
                f"</b><span style='margin-left:150px'></span><button onclick='save_and_show(\"{sentence[3]}\")' " \
                f"id='button_{sentence[3]}'>Store your answer and show suggestion</button></p>"
        html += f"<p id='answer_{sentence[3]}' class='{i + 2} second' style='display:none'></p>"
        html += f"<p id='solution_{sentence[3]}' class='{i + 2} second' style='display:none'><b>Suggestion:</b> " \
                f"{sentence[2]}</p>"
        html += f"<crowd-input name='exercise_{sentence[3]}' value=\"{sentence[1]}\" id='{sentence[3]}' required " \
                f"class='{i + 2} first' style='display:none'></crowd-input>"
        if not no_hints:
            html += f"<p id='help_{sentence[3]}' class='{i + 2}' style='display:none;margin-top:20px'>" \
                    f"<button onclick='help(\"{sentence[3]}\")'>Give me a hint</button></p>"

    for f in feedback_for_class.values():
        html += f

    html += f"<h2 class='{num_sentences + 2}' style='display:none'>Thank you for your participation and " \
            f"see you soon!</h2>"

    html += f"<h3 class='{num_sentences + 2}' style='display:none'>Feel free to leave a comment below.</h3>"
    html += f"<crowd-input name='comments' placeholder='...' class='{num_sentences + 2}' " \
            f"style='display:none;margin-bottom:25px'></crowd-input>"

    html += f"<h3 class='{num_sentences + 2}' style='display:none'>Please type FINISH to end the survey.</h3>"
    html += f"<crowd-input name='finish' placeholder='FINISH' required class='{num_sentences + 2}' " \
            f"style='display:none'></crowd-input>"

    html += f"<button class='{num_sentences + 2}' style='background-color:darkorange;border-color:darkorange;" \
            f"color:white;font-weight:bold;font-size:14pt;padding:5px 15px 5px 15px;display:none'>Submit</button>"

    html += f"<button onclick='previous({num_pages})' id='b1' style='color:grey;margin-top:25px;margin-right:10px'>" \
            f"Previous</button><button onclick='next({num_pages})' id='b2' style='color:grey'>Next</button>"

    html += """
    <script type="text/javascript">
    var color = 0;
    var current_page = 0;
    var max_page = 2;
    var timer_page = 2;
    var timer_page_2 = 2;
    function background_color(id) {
        if (id == "c1") {
            if (document.getElementById(id).checked) {
                document.body.style.background = "#414a4c";
                document.getElementById("c2").checked = false;
                document.getElementById("c3").checked = false;
                document.getElementById("c4").checked = false;
            } else {
                document.body.style.background = "white";
            }
        } else if (id == "c2") {
            if (document.getElementById(id).checked) {
                document.body.style.background = "#4a646c";
                document.getElementById("c1").checked = false;
                document.getElementById("c3").checked = false;
                document.getElementById("c4").checked = false;
            } else {
                document.body.style.background = "white";
            }
        } else {
            if (document.getElementById(id).checked) {
                document.body.style.background = " #738276";
                document.getElementById("c1").checked = false;
                document.getElementById("c2").checked = false;
                document.getElementById("c4").checked = false;
            } else {
                document.body.style.background = "white";
            }
        }
    }
    function random_background_color(id) {
        if (document.getElementById(id).checked) {
            let random_number = parseInt(Math.random() * 3, 10);
            if (random_number === 0) {
                document.body.style.background = "#414a4c";
            } else if (random_number === 1) {
                document.body.style.background = "#4a646c";
            } else {
                document.body.style.background = " #738276";
            }
            document.getElementById("c1").checked = false;
            document.getElementById("c2").checked = false;
            document.getElementById("c3").checked = false;
        } else {
            document.body.style.background = "white";
        }
    }
    function change_color() {
        if (!color) {
            document.getElementById("b2").style.color = "black";
            color++;
        } else if (color) {
            document.getElementById("b2").style.color = "grey";
            color--;
        }
    }
    function previous(num_pages) {
        if (current_page > 0) {
            let element;
            let elements = document.getElementsByClassName(current_page);
            for (let i = 0; i < elements.length; i++) {
                elements[i].style.display = "none";
            }
            current_page--;
            elements = document.getElementsByClassName(current_page);
            for (let j = 0; j < elements.length; j++) {
                if (!elements[j].classList.contains("first")) {
                    elements[j].style.display = "block";
                }
            }
            element = document.getElementById("b2");
            element.style.color = "black";
            if (current_page == 0) {
                element = document.getElementById("b1");
                element.style.color = "grey";
            }
    """
    if not no_hints:
        html += """
            for (let k = 0; k < 8; k++) {
                id_class =  "help_" + k;
                let helper_elements = document.getElementsByClassName(id_class);
                for (let l = 0; l < helper_elements.length; l++) {
                    helper_elements[l].style.display = "none";
                }
            }
        """
    html += """
        }
    }
    function next(num_pages) {
        if (current_page < num_pages - 1) {
            if (current_page == 0 && !document.getElementById("checkbox").checked) {
                return;
            }
            if (current_page == max_page || current_page == timer_page) {
                return;
            }
            let elements = document.getElementsByClassName(current_page);
            for (let i = 0; i < elements.length; i++) {
                elements[i].style.display = "none";
            }
            current_page++;
            if (current_page == timer_page_2 && current_page < num_pages - 1) {
                timer_page_2++;
                timer();
            }
            let element;
            elements = document.getElementsByClassName(current_page);
            for (let j = 0; j < elements.length; j++) {
                if (current_page == max_page) {
                    if (!elements[j].classList.contains("second")) {
                        elements[j].style.display = "block";
                    }
                } else {
                    if (!elements[j].classList.contains("first")) {
                        elements[j].style.display = "block";
                    }
                }
            }
            element = document.getElementById("b1");
            element.style.color = "black";
            if (current_page == num_pages - 1 || current_page == max_page) {
                element = document.getElementById("b2");
                element.style.color = "grey";
            }
    """
    if not no_hints:
        html += """
            for (let k = 0; k < 8; k++) {
                id_class =  "help_" + k;
                let helper_elements = document.getElementsByClassName(id_class);
                for (let l = 0; l < helper_elements.length; l++) {
                    helper_elements[l].style.display = "none";
                }
            }
        """
    html += """
        }
    }
    function save_and_show(id) {
        if (current_page == max_page) {
            let id_answer = "answer_" + id;
            let id_solution = "solution_" + id;
            let id_button = "button_" + id;
            document.getElementById(id_answer).innerHTML = "<b>Your answer:</b> " + document.getElementById(id).value;
            document.getElementById(id_answer).style.display = "block";
            document.getElementById(id_solution).style.display = "block";
            document.getElementById(id).style.display = "none";
            max_page++;
            if (max_page == timer_page) {
                document.getElementById("b2").style.color = "black";
            }
            document.getElementById(id_button).style.color = "grey";
        }
    }
    function help(id) {
        let my_arr = id.split("_");
        let my_class = my_arr[0];
        let id_class = "help_" + my_class;
        let elements = document.getElementsByClassName(id_class);
        for (let j = 0; j < elements.length; j++) {
            elements[j].style.display = "block";
        }
    }
    function timer() {
        let countdown = setInterval(decrement_s, 1000);
        let s = 7;
        function decrement_s() {
            s--;
            if (s == 0) {
                timer_page++;
                clearInterval(countdown);
                if (max_page == timer_page) {
                    document.getElementById("b2").style.color = "black";
                }
            }
        }
    }
    </script>
    """

    html += "</crowd-form>"

    html = html.replace("\n", "")
    html = html.replace("    ", "")

    xml = """
    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
        <HTMLContent>
            <![CDATA[<!DOCTYPE html>""" + html + """]]>
        </HTMLContent><FrameHeight>600</FrameHeight>
    </HTMLQuestion>
    """

    xml = xml.replace("\n", "")
    xml = xml.replace("    ", "")

    return xml, len(classes_to_train)
