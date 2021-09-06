import random  # (+ color smoke => star dust)

from helper import read_pickle, NUM_CLASSES, NUM_SENTENCES_PER_CLASS_TEST


# [0] → correct, [1] → error, [2] → solution(, [3] → identifier)
folder = "../prototype/datasets for MTurk/error_types_combined_data/"

class_casing = read_pickle(folder + "casing_errors_test")
class_singular = read_pickle(folder + "singular_errors_test")
class_plural = read_pickle(folder + "plural_errors_test")
class_vb_vbz = read_pickle(folder + "vb_vbz_errors_test")
class_vbz_vb = read_pickle(folder + "vbz_vb_errors_test")
class_vb__vbn = read_pickle(folder + "vb_vbn_errors_test")
class_vb__vbg = read_pickle(folder + "vb_vbg_errors_test")
class_other = read_pickle(folder + "other_verb_errors_test")

number_to_class = {0: "casing", 1: "singular", 2: "plural", 3: "vb_vbz", 4: "vbz_vb", 5: "vb__vbn", 6: "vb__vbg",
                   7: "other"}

sentences = []
for i, suffix in enumerate(number_to_class.values()):
    tmp = []  # => silence warning
    exec(f"tmp = class_{suffix}")
    for index in range(NUM_SENTENCES_PER_CLASS_TEST):
        tmp[index].append(f"{i}_{index}")  # class, sentence number [3]
    sentences.extend(tmp)

num_sentences = NUM_CLASSES * NUM_SENTENCES_PER_CLASS_TEST
num_pages = num_sentences + 3  # information, instructions, ..., submission
random.shuffle(sentences)

# adapt text for final round

c_0 = "class='0' style='display:block'"
c_0_mt = "class = '0', style='display:block;margin-top:25px'"
c_1 = "class='1' style='display:none'"
c_1_mt = "class='1' style='display:none;margin-top:25px'"

# INITIAL
"""
html = f"" "
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form answer-format="flatten-objects">

    <h2 {c_0}>Hi, please read the following information carefully before continuing.</h2>
        <p {c_0}>We are researching people's language abilities, via grammatical exercises.</p>
        <p {c_0}>This survey takes place over the course of 1-2 weeks and consists of multiple HITs that you will be 
        required to participate in over that period of time.<br>
        We will track your progress as you practice. The exercises that you will solve are designed to especially help 
        beginners and intermediate speakers improve their English proficiency.</p>
        <p {c_0_mt}></p>
        <p {c_0}>The general layout of this survey looks as follows:</p>
        <p {c_0}><b>Initial HIT <span style="background-color:yellow">($1 reward)</span>:</b> You will be asked 
        to take an initial language test that lasts approximately 15 minutes.</p>
        <p {c_0}><b>Main HITs <span style="background-color:yellow">($0.50 reward)</span>:</b> You will be invited, 
        at most once per day, to solve a HIT consisting of a small number of grammatical exercises. The duration of 
        such a HIT is around 5 minutes.<br>
        It's important that you finish each HIT within the allotted lifetime (time from publication) of 20 hours.</p>
        <p {c_0}><b>Final HIT <span style="background-color:yellow">($4 reward)</span>:</b> If you complete the initial 
        HIT and the main HITs, then you will once again be asked to take a language test. The test is similar in scope 
        to the test at the beginning of this survey.</p>
        <p {c_0_mt}></p>
        <p {c_0}>Note: It's possible that you complete the first HIT without being invited to subsequent HITs.</p>
        <p {c_0_mt}></p>
        <p {c_0}>It's important for us that people finish the whole survey. We also require that you solve both 
        language tests and all the grammatical exercises without relying on outside help.<br>
        Making mistakes is a natural part of learning, so nobody will be penalized for getting exercises wrong. 
        We do however expect you to make an effort.</p>
        <p {c_0}>If you have already participated in this survey, please do not join again, as we will not be able 
        to approve your assignment.</p>
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
        <p {c_1}>Each sentence contains a single grammatical error. It is your task to find and correct this error.</p>
        <ul {c_1}>
            <li>Grammatical errors include, among other things, faulty verb forms, singular and plural errors, 
            and capitalization errors.<br>
            Bad punctuation or stylistically bad writing should not be counted as an error.</li>
            <li>If there are multiple ways to fix an error, choose the one that is closest to the original.<br>
            Also, try to correct faulty words instead of adding or removing words.</li>
        </ul>
        <p {c_1_mt}></p>
        <p {c_1}>There is a time lock that stops Turkers from rushing through the exercises, so please take your time 
        and really try to find the errors.</p>
        
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

# FINAL
# """
html = f"""
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form answer-format="flatten-objects">

    <h2 {c_0}>Hi, please read the following information carefully before continuing.</h2>
        <p {c_0}>This is the final HIT of the survey. Like in the initial HIT, you are once again asked to take a 
        language test. We will use the results to measure your progress.</p>
        <p {c_0_mt}></p>
        <p {c_0}>Please solve all the grammatical exercises without relying on outside help and don't forget 
        to have fun! Making mistakes isn't a problem, but, as usual, we do expect you to make an effort.</p>
        <p {c_0_mt}></p>
        <p {c_0}><b>Thank you.</b></p>
        <p {c_0_mt}></p>
        <p {c_0}>I confirm to have read and understood the provided information: 
        <input type="checkbox" onchange="change_color()" id="checkbox"></p>
        <p {c_0_mt}>
        <p {c_0}>In case you wish to change the color of the survey background:</p>
        <p {c_0}><input type="checkbox" onchange="background_color('c1')" id="c1"> outer space</p>
        <p {c_0}><input type="checkbox" onchange="background_color('c2')" id="c2"> deep space sparkle</p>
        <p {c_0}><input type="checkbox" onchange="background_color('c3')" id="c3"> smoke</p> 
        <p {c_0}><input type="checkbox" onchange="random_background_color('c4')" id="c4"> random</p>

    <h2 {c_1}>Instructions</h2>
        <p {c_1}><span style="background-color:yellow">Each sentence contains a single grammatical error. 
        It is your task to find and correct this error.</span></p>
        <ul {c_1}>
            <li>Grammatical errors include, among other things, faulty verb forms, singular and plural errors, 
            and capitalization errors.<br>
            Bad punctuation or stylistically bad writing should not be counted as an error.</li>
            <li>If there are multiple ways to fix an error, choose the one that is closest to the original.<br>
            Also, try to correct faulty words instead of adding or removing words.</li>
        </ul>
        <p {c_1_mt}></p>
        <p {c_1}>There is a time lock that stops Turkers from rushing through the exercises, so please take your time 
        and really try to find the errors.</p>

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


# \"escape\" apostrophe (! (or replace \' ? (++)))
for i, sentence in enumerate(sentences):
    html += f"<h3 class='{i + 2}' style='display:none'>Exercise {i + 1} of {num_sentences}</h3>"
    # html += f"<p id='storage_{sentence[3]}' style='display:none'>{sentence[1]}</p>"
    html += f"<p class='{i + 2}' style='display:none'><b>Fix the following sentence directly in the input box:</b></p>"
    # + f"<span style='margin-left:150px'></span><button onclick='reset(\"{sentence[3]}\")'>" \
    # + f"Reset current sentence</button></p>"
    html += f"<crowd-input name='exercise_{sentence[3]}' value=\"{sentence[1]}\" id='{sentence[3]}' required " \
            f"class='{i + 2}' style='display:none'></crowd-input>"

# TODO: initial / final
# Thank you for your participation and see you soon!
html += f"<h2 class='{num_sentences + 2}' style='display:none'>Thank you for your participation " \
        f"throughout this survey! All the best!</h2>"

# """
html += f"<p class='{num_sentences + 2}' style='display:none'>I hope you had some fun and maybe even learned " \
        f"something new. It might interest you that all exercises were generated automatically from various datasets " \
        f"of student sentences using a grammatical error correction model (https://github.com/grammarly/gector). " \
        f"This is also the reason why some of the sentences contained errors and why the solutions were at times " \
        f"incorrect.</p>"
# """

html += f"<h3 class='{num_sentences + 2}' style='display:none'>Feel free to leave a comment below.</h3>"
html += f"<crowd-input name='comments' placeholder='...' class='{num_sentences + 2}' " \
        f"style='display:none;margin-bottom:25px'></crowd-input>"

html += f"<h3 class='{num_sentences + 2}' style='display:none'>Please type FINISH to end the survey.</h3>"
html += f"<crowd-input name='finish' placeholder='FINISH' required class='{num_sentences + 2}' " \
        f"style='display:none'></crowd-input>"

html += f"<button class='{num_sentences + 2}' style='background-color:darkorange;border-color:darkorange;color:white;" \
        f"font-weight:bold;font-size:14pt;padding:5px 15px 5px 15px;display:none'>Submit</button>"

html += f"<button onclick='previous({num_pages})' id='b1' style='color:grey;margin-top:25px;margin-right:10px'>" \
        f"Previous</button><button onclick='next({num_pages})' id='b2' style='color:grey'>Next</button>"

html += """
<script type="text/javascript">
var color = 0;
var current_page = 0;
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
            elements[j].style.display = "block";
        }
        element = document.getElementById("b2");
        element.style.color = "black";
        if (current_page == 0) {
            element = document.getElementById("b1");
            element.style.color = "grey";
        }
    }
}
function next(num_pages) {
    if (current_page < num_pages - 1) {
        if (current_page == 0 && !document.getElementById("checkbox").checked) {
            return;
        }
        if (current_page == timer_page) {
            return;
        }
        let elements = document.getElementsByClassName(current_page);
        for (let i = 0; i < elements.length; i++) {
            elements[i].style.display = "none";
        }
        current_page++;
        let element;
        if (current_page == timer_page_2 && current_page < num_pages - 1) {
            element = document.getElementById("b2");
            element.style.color = "grey";
            timer_page_2++;
            timer();
        }
        elements = document.getElementsByClassName(current_page);
        for (let j = 0; j < elements.length; j++) {
            elements[j].style.display = "block";
        }
        element = document.getElementById("b1");
        element.style.color = "black";
        if (((current_page == timer_page_2 - 1 && current_page == timer_page) || current_page == num_pages - 1) && 
        current_page >= 2) {
            element = document.getElementById("b2");
            element.style.color = "grey";
        }
    }
}
function timer() {
    let s = 5;
    let countdown = setInterval(decrement_s, 1000);
    function decrement_s() {
        s--;
        if (s == 0) {
            timer_page++;
            clearInterval(countdown);
            document.getElementById("b2").style.color = "black";
        }
    }
}
</script>
"""

"""
function reset(id) {
    document.getElementById(id).value = document.getElementById("storage_" + id).innerHTML;
}
# """

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

"""
with open("initial_survey.xml", "w") as file:
    file.write(xml)
# """

# """
with open("final_survey.xml", "w") as file:
    file.write(xml)
# """
