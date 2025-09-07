# import libaries to read json file
import json
from contextlib import redirect_stdout
import argparse
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
    Parse a free-form model output and extract a single MC choice A/B/C/D.
    Logic mirrors prior inline heuristics but centralized and clearer.
    Rules (any one positive pattern counts for a letter):
      - Leading token: "A" / "A." / "A:" (same for B/C/D)
      - Parenthetical form: "(A)"
      - Explicit: "Answer: A"
    Ambiguity handling:
      - We only look for leading forms inside the first 8 characters to avoid
        matching option lists reproduced later.
      - Parenthetical / 'Answer: X' can appear anywhere.
      - If more than one distinct letter is detected anywhere, mark illegal (return None).
    Returns: 'A' | 'B' | 'C' | 'D' | None
    """
    if not raw_pred:
        return None
    text = raw_pred.strip()
    upper = text.upper()
    prefix = upper[:8]

    letters = ['A', 'B', 'C', 'D']
    occurrences = {}

    def record(letter, idx):
        # Keep earliest index
        if letter not in occurrences or idx < occurrences[letter]:
            occurrences[letter] = idx

    for L in letters:
        # Leading forms
        if prefix.startswith(L) and (len(prefix) == 1 or not prefix[1].isalpha()):
            record(L, 0)
        if prefix.startswith(L + '.') or prefix.startswith(L + ':'):
            record(L, 0)
        # Parenthetical anywhere
        par_idx = upper.find(f'({L})')
        if par_idx != -1:
            record(L, par_idx)
        # "Answer: L"
        m = re.search(r'ANSWER:\s*' + L + r'\b', upper)
        if m:
            record(L, m.start())

    if not occurrences:
        return None

    # Disambiguate: accept only if exactly one distinct letter matched
    matched_letters = list(occurrences.keys())
    if len(matched_letters) == 1:
        return matched_letters[0]
    return None

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    file = path.split('/')[-1]
    path = '/'.join(path.split('/')[:-1]) + '/'
    with open(path + file) as f:
        data = json.load(f)

    total_accuracy = 0
    total_questions = 0
    A_count = B_count = C_count = D_count = 0
    correct_A_count = correct_B_count = correct_C_count = correct_D_count = 0
    gt_A_count = gt_B_count = gt_C_count = gt_D_count = 0
    wrong_predictions = []
    illegal_predictions = 0

    for i in range(0, len(data), 1):
        qid = data[i]['id']
        answer = data[i]['answer']
        raw_pred = data[i]['pred']
        predicted_answer = answer_parser(raw_pred)

        if predicted_answer is None:
            print("please check the prediction")
            illegal_predictions += 1
            print(raw_pred)
            continue

        if predicted_answer == 'A':
            A_count += 1
        elif predicted_answer == 'B':
            B_count += 1
        elif predicted_answer == 'C':
            C_count += 1
        elif predicted_answer == 'D':
            D_count += 1

        if answer == 'A':
            gt_A_count += 1
        elif answer == 'B':
            gt_B_count += 1
        elif answer == 'C':
            gt_C_count += 1
        elif answer == 'D':
            gt_D_count += 1

        if answer == predicted_answer:
            total_accuracy += 1
            total_questions += 1
            if predicted_answer == 'A':
                correct_A_count += 1
            elif predicted_answer == 'B':
                correct_B_count += 1
            elif predicted_answer == 'C':
                correct_C_count += 1
            elif predicted_answer == 'D':
                correct_D_count += 1
        else:
            total_questions += 1
            wrong_predictions.append({"answer": answer, "predicted_answer": predicted_answer, "qid": qid})

    with open(path + 'grade.txt', 'w') as f:
        with redirect_stdout(f):
            print('--'*20)
            print("Total Questions: ", total_questions)
            print("Total Accuracy: ", total_accuracy / total_questions if total_questions else 0)
            print("Illegal Predictions: ", illegal_predictions)
            print("P_A: ", A_count / total_questions if total_questions else 0, "    P_A^+: ", correct_A_count / total_accuracy if total_accuracy else 0)
            print("P_B: ", B_count / total_questions if total_questions else 0, "    P_B^+: ", correct_B_count / total_accuracy if total_accuracy else 0)
            print("P_C: ", C_count / total_questions if total_questions else 0, "    P_C^+: ", correct_C_count / total_accuracy if total_accuracy else 0)
            print("P_D: ", D_count / total_questions if total_questions else 0, "    P_D^+: ", correct_D_count / total_accuracy if total_accuracy else 0)
            if total_questions:
                print("Option Balance: ", (((A_count / total_questions - 0.25)**2 + (B_count / total_questions - 0.25)**2 + (C_count / total_questions - 0.25)**2 + (D_count / total_questions - 0.25)**2)/4)**0.5)
            if total_accuracy:
                print("Correct Option Balance: ", (((correct_A_count / total_accuracy - 0.25)**2 + (correct_B_count / total_accuracy - 0.25)**2 + (correct_C_count / total_accuracy - 0.25)**2 + (correct_D_count / total_accuracy - 0.25)**2)/4)**0.5)
            if total_questions:
                current_distribution = [A_count / total_questions, B_count / total_questions, C_count / total_questions, D_count / total_questions]
                current_distribution = np.maximum(current_distribution, 1e-10)
                desired_distribution = [gt_A_count / total_questions, gt_B_count / total_questions, gt_C_count / total_questions, gt_D_count / total_questions]
                intermediate_distribution = [(current_distribution[i] + desired_distribution[i]) / 2 for i in range(4)]
                jensen_shannon_divergence = 0.5 * sum([current_distribution[i] * np.log2(current_distribution[i] / intermediate_distribution[i]) for i in range(4)]) + 0.5 * sum([desired_distribution[i] * np.log2(desired_distribution[i] / intermediate_distribution[i]) for i in range(4)])
                print("Jensen-Shannon Divergence: ", jensen_shannon_divergence)

    with open(path + 'wrong_predictions.json', 'w') as f:
        json.dump(wrong_predictions, f, indent=4)

