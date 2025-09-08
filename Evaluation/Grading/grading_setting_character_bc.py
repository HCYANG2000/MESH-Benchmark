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

    # Disambiguate: if multiple records, pick earliest
    occurrences = sorted(occurrences.items(), key=lambda x: x[1])
    return 0 if occurrences[0][0] == 'A' else 1
    
if __name__ == '__main__':
    args = parse_args()
    path = args.path

    file = path.split('/')[-1]

    path = '/'.join(path.split('/')[:-1]) + '/'

    # Read the qa.json file
    with open(path + file) as f:
        data = json.load(f)

    # Initialize the total accuracy, total number of questions, object accuracy, total number of questions about obj, character accuracy, total number of questions about char
    total_accuracy = 0
    total_questions = 0
    # Initialize the number of positive and negative questions, the accuracy of positive and negative questions
    positive_questions = 0
    positive_accuracy = 0
    negative_questions = 0
    negative_accuracy = 0

    # Collect the wrong predictions and its corresponding question
    wrong_predictions = []

    # Count the number of illegal predictions
    illegal_predictions = 0

    # mimic the evaluation result of small dataset
    video_interval = 1
    video_counter = -1

    # Loop through the data
    for i in range(0, len(data), 1):
        # Get qid
        qid = data[i]['id']
        # Get the answer
        answer = data[i]['answer']
        # Use full prediction text for richer parsing
        predicted_answer = answer_parser(data[i]['pred'])

        # If the answer is correct
        if answer == predicted_answer:
            # Increment the total accuracy
            total_accuracy += 1
            # Increment the total number of questions
            total_questions += 1

            # if the answer is positive
            if answer == 0:
                # Increment the positive accuracy
                positive_accuracy += 1
                # Increment the total number of positive questions
                positive_questions += 1
            # if the answer is negative
            elif answer == 1:
                # Increment the negative accuracy
                negative_accuracy += 1
                # Increment the total number of negative questions
                negative_questions += 1
            
        # If the answer is incorrect
        else:
            # Increment the total number of questions
            total_questions += 1
            # Collect the wrong predictions and its corresponding question
            wrong_predictions.append({"answer": answer, "predicted_answer": predicted_answer, "qid": qid})

            # if the answer is positive
            if answer == 0:
                # Increment the total number of positive questions
                positive_questions += 1
            # if the answer is negative
            elif answer == 1:
                # Increment the total number of negative questions
                negative_questions += 1

    # save the metrics into a txt file 
    with open(path + 'grade.txt', 'w') as f:
        with redirect_stdout(f):
            print('--'*20)
            # Print the total number of questions
            print("Total Questions: ", total_questions)
            # Print the total accuracy
            print("Total Accuracy: ", total_accuracy / total_questions)
            # Print the number of illegal predictions
            print("Illegal Predictions: ", illegal_predictions)
                

            print('--'*20)
            # Print the total number of positive questions
            print("Positive Questions: ", positive_questions)
            # Print the positive accuracy
            print("Positive Accuracy: ", positive_accuracy / positive_questions)
            # Print the total number of negative questions
            print("Negative Questions: ", negative_questions)
            # Print the negative accuracy
            print("Negative Accuracy: ", negative_accuracy / negative_questions)
            # Print the positive rate, which is number of positive predictions divided by the total number of predictions
            print("Positive Rate: ", (positive_accuracy + negative_questions - negative_accuracy) / total_questions) 
            # Calculate the jensen-shannon divergence between the current distribution and the uniform distribution
            current_distribution = np.array([(positive_accuracy + negative_questions - negative_accuracy) / total_questions, 1- (positive_accuracy + negative_questions - negative_accuracy) / total_questions])
            # Truncate the current distribution to avoid log(0)
            current_distribution = np.maximum(current_distribution, 1e-10)
            desired_distribution = np.array([positive_questions, negative_questions]) / total_questions
            intermediate_distribution = (current_distribution + desired_distribution) / 2
            jensen_shannon_divergence = 0.5 * np.sum(current_distribution * np.log(current_distribution / intermediate_distribution)) + 0.5 * np.sum(desired_distribution * np.log(desired_distribution / intermediate_distribution))
            # Print the jensen-shannon divergence
            print("Jensen-Shannon Divergence: ", jensen_shannon_divergence)

            print('--'*20)
            # Calculate the confusion matrix
            confusion_matrix = [[positive_accuracy, negative_questions - negative_accuracy], [positive_questions - positive_accuracy, negative_accuracy]]
            # Print the confusion matrix
            print("Confusion Matrix: TP ", confusion_matrix[0][0], " FP ", confusion_matrix[0][1], "\n                  FN ", confusion_matrix[1][0], " TN ", confusion_matrix[1][1])
            if positive_accuracy + negative_questions - negative_accuracy == 0:
                print("Precision: Unavailable")
            else:
                # print the precision
                precision = positive_accuracy / (positive_accuracy + negative_questions - negative_accuracy)
                print("Precision: ", precision)
            if positive_questions == 0:
                print("Recall: Unavailable")
            else:
                # print the recall
                recall = positive_accuracy / positive_questions
                print("Recall: ", recall)
            if precision + recall == 0:
                print("F1 Score: Unavailable")
            else:
                # Print the F1 score
                f1_score = 2 * precision * recall / (precision + recall)
                print("F1 Score: ", f1_score)
            print('--'*20)

    # save the wrong predictions in a json with proper formatting
    with open(path + 'wrong_predictions.json', 'w') as f:
        json.dump(wrong_predictions, f, indent=4)

