import requests

from collections import defaultdict

sample_questions = [
    # academic_info
    "What courses are required for the undergraduate curriculum?",
    "How many credit hours are needed for the bachelor's degree?",
    "Which program offers a specialization in artificial intelligence?",
    "What topics are covered in the English 1102 syllabus?",
    "How is the first-year writing course structured?",
    "Can you explain the focus areas for the computer science major?",
    "What does the portfolio exhibit for graduate students involve?",
    "Are there capstone projects required for the master's program?",
    "How is academic advising handled for new enrollments?",
    "When is the placement exam scheduled for freshmen?",
    "What subjects are emphasized in the first-year classes?",
    "Are interdisciplinary studies part of the undergraduate options?",
    "What is the structure of upper-level study tracks?",
    "What are the thesis expectations for doctoral students?",
    "Is it possible to double major in engineering and math?",
    "Where can I find research opportunities as an undergrad?",
    "How can I personalize my academic journey?",
    "Is the curriculum flexible for transfer students?",
    "What are common study paths within liberal arts?",
    "Can I minor in entrepreneurship alongside my major?",

    # faculty_and_awards
    "Who is the newly appointed department chair this semester?",
    "Which professor recently earned a national award?",
    "What achievements were recognized at the faculty awards banquet?",
    "Has any research group received special recognition?",
    "Were any students honored on the dean's list this year?",
    "Is there a lab team that won an innovation challenge?",
    "What major publications have faculty contributed to recently?",
    "Who among the faculty was honored for excellence in teaching?",
    "Did any student club win awards in the past year?",
    "What scholarships have students recently received?",
    "How do professors get involved in national associations?",
    "Which faculty members are leading significant grants?",
    "What academic journals feature our department's work?",
    "Are there faculty leading major international conferences?",
    "Which students received commendations for research excellence?",
    "What honors are available for outstanding thesis projects?",
    "Is there a yearly recognition event for top graduates?",
    "How are research fellows celebrated in the department?",
    "Are there grants given for public service projects?",
    "Have any faculty members authored award-winning books?",

    # event_announcement
    "When is the next departmental symposium scheduled?",
    "What workshops will be held during orientation week?",
    "Are there any hackathons planned for computer science students?",
    "Will there be a colloquium on literary studies this semester?",
    "When is the commencement ceremony for the graduating class?",
    "Is there an open house for prospective students this spring?",
    "What lectures are scheduled on the topic of artificial intelligence?",
    "Are there any guest speakers visiting the campus this month?",
    "When is the research presentation day for graduate students?",
    "Is there a career fair happening this semester?",
    "Will there be a seminar on emerging technologies?",
    "How do students register for leadership conferences?",
    "Are interdisciplinary panels hosted regularly?",
    "When is the summer study abroad information session?",
    "Is there an entrepreneurship showcase this fall?",
    "What events highlight undergraduate research opportunities?",
    "Are writing retreats available for graduate students?",
    "What kinds of networking events are organized for students?",
    "How often are alumni guest talks arranged?",
    "Is there a departmental celebration planned for the anniversary?",

    # financial_info
    "What are the tuition rates for the fall semester?",
    "Are there scholarships available for first-generation students?",
    "How can I apply for financial aid?",
    "What payment plans are offered for tuition and fees?",
    "Is there funding support for international students?",
    "What loan options are available for graduate studies?",
    "Where can students apply for research funding grants?",
    "Is alumni donation critical for maintaining scholarships?",
    "How much does the study abroad program typically cost?",
    "Are there emergency assistance funds for students?",
    "What external scholarships are commonly awarded here?",
    "Can tuition costs be deferred under certain circumstances?",
    "What financial resources are offered to veterans?",
    "Is housing included in the cost of attendance estimates?",
    "Are there grants available for service learning programs?",
    "What fellowships are open to doctoral students?",
    "Is there a fundraiser event to support student clubs?",
    "How are departmental grants distributed among research groups?",
    "Where can I find the latest information about education financing?",
    "Do faculty sponsors sometimes assist students financially?",

    # ask_general_question
    "Where can I find a quiet place to study on campus?",
    "Is there a club for students interested in astronomy?",
    "How can I join the intramural sports program?",
    "What are the hours for the main library?",
    "Where do students usually hang out between classes?",
    "Are tutoring services free for undergraduates?",
    "How do I report a lost ID card?",
    "Is there a student organization for volunteer work?",
    "Can alumni access career services after graduation?",
    "Where is the nearest place to get coffee on campus?",
    "What is the deadline for financial aid applications?",
    "How do I register for next semester's courses?",
    "When is the last day to drop a course without penalty?",
    "What scholarship options are available for transfer students?",
    "Are there any workshops to improve academic writing skills?",
    "Where do I go for help with technical issues on campus?",
    "Is there a fee for parking permits?",
    "Can I work part-time while taking full classes?",
    "Is summer housing available on campus?",
    "How do I contact the office of student life?"
]

expected_intents = (
    ["academic_info"] * 20 +
    ["faculty_and_awards"] * 20 +
    ["event_announcement"] * 20 +
    ["financial_info"] * 20 +
    ["ask_general_question"] * 20
)

rasa_url = "http://localhost:5005/model/parse"

# Tracking
correct = 0
total = len(sample_questions)

intent_totals = defaultdict(int)
intent_correct = defaultdict(int)

# Evaluate each question
for question, expected_intent in zip(sample_questions, expected_intents):
    payload = {"text": question}
    try:
        response = requests.post(rasa_url, json=payload)
        result = response.json()
        predicted_intent = result['intent']['name']
        intent_totals[expected_intent] += 1

        status = "Correct" if predicted_intent == expected_intent else f"Incorrect (Expected: {expected_intent})"
        if predicted_intent == expected_intent:
            correct += 1
            intent_correct[expected_intent] += 1

        print(f"Q: {question}")
        print(f"Predicted Intent: {predicted_intent} → {status}\n")

    except Exception as e:
        print(f"Error with question: {question}")
        print(f"Error: {e}\n")

# Summary
accuracy = (correct / total) * 100
print("="*50)
print(f"Total Samples Tested: {total}")
print(f"Overall Accuracy: {accuracy:.2f}%\n")

print("Accuracy by Intent:")
for intent in sorted(intent_totals.keys()):
    total_i = intent_totals[intent]
    correct_i = intent_correct.get(intent, 0)
    acc = (correct_i / total_i) * 100
    print(f"  {intent}: {acc:.2f}% ({correct_i}/{total_i})")
