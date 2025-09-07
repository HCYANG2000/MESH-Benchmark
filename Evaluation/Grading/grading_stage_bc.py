# import libaries to read json file
import json
import argparse
from contextlib import redirect_stdout
import numpy as np
import re

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path", required=True)
    return parser.parse_args()

def answer_parser(raw_pred: str):
    """
    Parse a free-form model output and extract a single binary choice A/B (0/1).
    Accepted cues (case-insensitive):
      - Leading token within first 8 chars: "A", "A.", "A:" (and B)
      - Parenthetical anywhere: "(A)" or "(B)"
      - Explicit anywhere: "Answer: A" or "Answer: B"
      - Whole-word YES => A, NO => B
    Ambiguity handling:
      - If more than one distinct choice cue (A vs B) is detected anywhere, return None.
    Returns: 0 for A/YES, 1 for B/NO, or None if illegal/unknown.
    """
    global illegal_predictions  # maintain existing counter behavior

    if not isinstance(raw_pred, str):
        illegal_predictions += 1
        return None

    text = raw_pred.strip()
    if not text:
        illegal_predictions += 1
        return None

    upper = text.upper()
    prefix = upper[:8]

    occurrences = {}  # {'A': earliest_index, 'B': earliest_index}

    def record(letter: str, idx: int):
        # keep earliest index per letter
        if letter not in occurrences or idx < occurrences[letter]:
            occurrences[letter] = idx

    # A/B cues
    for L in ['A', 'B']:
        # Leading forms within first 8 chars
        if prefix.startswith(L) and (len(prefix) == 1 or not prefix[1].isalpha()):
            record(L, 0)
        if prefix.startswith(L + '.') or prefix.startswith(L + ':'):
            record(L, 0)

        # Parenthetical anywhere
        par_idx = upper.find(f'({L})')
        if par_idx != -1:
            record(L, par_idx)

        # "Answer: L" anywhere
        m = re.search(r'ANSWER:\s*' + L + r'\b', upper)
        if m:
            record(L, m.start())

    # YES/NO as whole words anywhere (YES -> A, NO -> B)
    for m in re.finditer(r'\bYES\b', upper):
        record('A', m.start())
    for m in re.finditer(r'\bNO\b', upper):
        record('B', m.start())

    if not occurrences:
        illegal_predictions += 1
        return None

    # Disambiguate: accept only if exactly one distinct letter matched
    if len(occurrences) == 1:
        letter = next(iter(occurrences.keys()))
        return 0 if letter == 'A' else 1

    # Multiple distinct cues detected -> ambiguous/illegal
    illegal_predictions += 1
    return None

if __name__ == '__main__':
    args = parse_args()
    path = args.path

    file = path.split('/')[-1]

    path = '/'.join(path.split('/')[:-1]) + '/'

    # Read the qa.json file
    with open(path + file) as f:
        data = json.load(f)

    # Initialize the total accuracy, total number of questions for each type
    total_accuracy = 0
    total_questions = 0
    ee_accuracy = 0
    ee_questions = 0
    eov_accuracy = 0
    eov_questions = 0
    pov_accuracy = 0
    pov_questions = 0
    se_accuracy = 0
    se_questions = 0
    miv_accuracy = 0
    miv_questions = 0
    positive_predictions = 0
    negative_predictions = 0

    # Collect the wrong predictions and its corresponding question
    wrong_predictions = []

    # Count the number of illegal predictions
    illegal_predictions = 0

    # Loop through the data
    for i in range(0, len(data), 1):
        qid = data[i]['id']
        # Get the question type
        question_type = data[i]['label']
        # Get the answer
        answer = data[i]['answer']
        # Get the predicted answer
        predicted_answer = answer_parser(data[i]['pred'])
        if predicted_answer is None:
            print("please check the prediction")
            illegal_predictions += 1
            print(data[i]['pred'])
            continue 
        # If the answer is correct
        if answer == predicted_answer:
            # Increment the total accuracy
            total_accuracy += 1
            # Increment the total number of questions
            total_questions += 1

            # If the question type is ee
            if question_type == 'correct':
                ee_accuracy += 1
                ee_questions += 1
                
            # If the question type is EOV
            elif question_type == 'wrong_out_video_action':
                eov_accuracy += 1
                eov_questions += 1

            # If the question type is POV
            elif question_type == 'wrong_out_vid_person':
                pov_accuracy += 1
                pov_questions += 1

            # If the question type is SE
            elif question_type == 'wrong_similar_action':
                se_accuracy += 1
                se_questions += 1

            # If the question type is MIV
            elif question_type == 'wrong_in_video_action' or question_type == 'wrong_in_vid_person':
                miv_accuracy += 1
                miv_questions += 1
            
        # If the answer is incorrect
        else:
            # Increment the total number of questions
            total_questions += 1
            # Collect the wrong predictions and its corresponding question
            wrong_predictions.append({"answer": answer, "predicted_answer": predicted_answer, "qid": qid})

            # If the question type is ee
            if question_type == 'correct':
                ee_questions += 1

            # If the question type is EOV
            elif question_type == 'wrong_out_video_action':
                eov_questions += 1

            # If the question type is POV
            elif question_type == 'wrong_out_vid_person':
                pov_questions += 1

            # If the question type is SE
            elif question_type == 'wrong_similar_action':
                se_questions += 1

            # If the question type is MIV
            elif question_type == 'wrong_in_video_action':
                miv_questions += 1

            # If the question type is PIV
            elif question_type == 'wrong_in_vid_person' or question_type == 'wrong_in_vid_person':
                miv_questions += 1

    # save the metrics into a txt file 
    with open(path + 'grade.txt', 'w') as f:
        with redirect_stdout(f):
            print('--'*20)
            # Print the total number of questions
            print("Total Questions: ", total_questions)
            # Print the total accuracy
            print("Total Accuracy: ", total_accuracy / total_questions)
            # Print the weighted accuracy
            negative_count = 0
            for i in [eov_questions, pov_questions, se_questions, miv_questions]:
                if i > 0:
                    negative_count += 1
            negative_weight = 0.5 / negative_count
            print("Weighted Accuracy: ", ee_accuracy / ee_questions * 0.5 + sum([i / j * negative_weight if j > 0 else 0 for i, j in zip([eov_accuracy, pov_accuracy, se_accuracy, miv_accuracy], [eov_questions, pov_questions, se_questions, miv_questions])]))
            # Calculate the jenssen-shannon divergence
            current_distribution = np.array([positive_predictions / total_questions, negative_predictions / total_questions])
            current_distribution = np.maximum(current_distribution, 1e-10)
            desired_distribution = np.array([ee_questions / total_questions, sum([eov_questions, pov_questions, se_questions, miv_questions]) / total_questions])
            intermediate_distribution = (current_distribution + desired_distribution) / 2
            jensen_shannon_divergence = 0.5 * np.sum(current_distribution * np.log(current_distribution / intermediate_distribution)) + 0.5 * np.sum(desired_distribution * np.log(desired_distribution / intermediate_distribution))
            print("Jensen-Shannon Divergence: ", jensen_shannon_divergence)
            # Print the number of illegal predictions
            print("Illegal Predictions: ", illegal_predictions)
            if ee_questions > 0:
                # Print the number of questions for each type
                print("EE Questions: ", ee_questions)
                # Print the accuracy for each type
                print("EE Accuracy: ", ee_accuracy / ee_questions)

            if eov_questions > 0:
                print("EOV Questions: ", eov_questions)
                print("EOV Accuracy: ", eov_accuracy / eov_questions)

            if pov_questions > 0:
                print("POV Questions: ", pov_questions)
                print("POV Accuracy: ", pov_accuracy / pov_questions)

            if se_questions > 0:
                print("SE Questions: ", se_questions)
                print("SE Accuracy: ", se_accuracy / se_questions)

            if miv_questions > 0:
                print("MIV Questions: ", miv_questions)
                print("MIV Accuracy: ", miv_accuracy / miv_questions)
                

    # save the wrong predictions in a json with proper formatting
    with open(path + 'wrong_predictions.json', 'w') as f:
        json.dump(wrong_predictions, f, indent=4)

