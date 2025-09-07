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

def jensen_shannon_divergence(current_distribution, desired_distribution=[0.25, 0.25, 0.25, 0.25]):
    """
    Calculate the Jensen Shannon Divergence.
    """
    current_distribution = np.maximum(current_distribution, 1e-10)
    desired_distribution = np.maximum(desired_distribution, 1e-10)
    intermediate_distribution = 0.5 * current_distribution + 0.5 * desired_distribution
    jensen_shannon_divergence = 0.5 * np.sum(current_distribution * np.log(current_distribution / intermediate_distribution)) + 0.5 * np.sum(desired_distribution * np.log(desired_distribution / intermediate_distribution))
    return jensen_shannon_divergence

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

answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

if __name__ == '__main__':
    args = parse_args()
    path = args.path

    file = path.split('/')[-1]

    path = '/'.join(path.split('/')[:-1]) + '/'

    # Read the qa.json file
    try:
        with open(path + file) as f:
            data = json.load(f)
    except:
        with open(path + file,'r+') as f:
            f.seek(0, 2)
            f.seek(f.tell() - 5)
            f.write(']\n')
        with open(path + file) as f:
            data = json.load(f)
    print("Example of a question:")
    print(data[0])

    # Initialize the total accuracy, total number of questions for each type
    eov_accuracy = np.zeros(4, dtype=int)
    eov_predictions = np.zeros(4, dtype=int)
    eov_gts = np.zeros(4, dtype=int)

    pov_accuracy = np.zeros(4, dtype=int)
    pov_predictions = np.zeros(4, dtype=int)
    pov_gts = np.zeros(4, dtype=int)

    se_accuracy = np.zeros(4, dtype=int)
    se_predictions = np.zeros(4, dtype=int)
    se_gts = np.zeros(4, dtype=int)

    miv_accuracy = np.zeros(4, dtype=int)
    miv_predictions = np.zeros(4, dtype=int)
    miv_gts = np.zeros(4, dtype=int)

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
        answer = data[i]['A']
        # Get the predicted answer
        predicted_answer = answer_parser(data[i]['pred'])
        if predicted_answer is None:
            print("please check the prediction")
            illegal_predictions += 1
            print(data[i]['pred'])
            continue
        # If the answer is correct
        if answer == predicted_answer:
                
            # If the question type is EOV
            if question_type == 'wrong_out_video_action':
                eov_accuracy[answer_map[predicted_answer]] += 1
                eov_predictions[answer_map[predicted_answer]] += 1
                eov_gts[answer_map[answer]] += 1

            # If the question type is POV
            elif question_type == 'wrong_out_vid_person':
                pov_accuracy[answer_map[predicted_answer]] += 1
                pov_predictions[answer_map[predicted_answer]] += 1
                pov_gts[answer_map[answer]] += 1

            # If the question type is SE
            elif question_type == 'wrong_similar_action':
                se_accuracy[answer_map[predicted_answer]] += 1
                se_predictions[answer_map[predicted_answer]] += 1
                se_gts[answer_map[answer]] += 1

            # If the question type is MIV
            elif question_type == 'wrong_in_video_action' or question_type == 'wrong_in_vid_person':
                miv_accuracy[answer_map[predicted_answer]] += 1
                miv_predictions[answer_map[predicted_answer]] += 1
                miv_gts[answer_map[answer]] += 1
            
        # If the answer is incorrect
        else:
            # Collect the wrong predictions and its corresponding question
            wrong_predictions.append({"answer": answer, "predicted_answer": predicted_answer, "qid": qid})

            # If the question type is EOV
            if question_type == 'wrong_out_video_action':
                eov_predictions[answer_map[predicted_answer]] += 1
                eov_gts[answer_map[answer]] += 1

            # If the question type is POV
            elif question_type == 'wrong_out_vid_person':
                pov_predictions[answer_map[predicted_answer]] += 1
                pov_gts[answer_map[answer]] += 1

            # If the question type is SE
            elif question_type == 'wrong_similar_action':
                se_predictions[answer_map[predicted_answer]] += 1
                se_gts[answer_map[answer]] += 1

            # If the question type is miv
            elif question_type == 'wrong_in_video_action' or question_type == 'wrong_in_vid_person':
                miv_predictions[answer_map[predicted_answer]] += 1
                miv_gts[answer_map[answer]] += 1

    total_accuracy = eov_accuracy + pov_accuracy + se_accuracy + miv_accuracy
    total_predictions = eov_predictions + pov_predictions + se_predictions + miv_predictions
    total_gts = eov_gts + pov_gts + se_gts + miv_gts

    # save the metrics into a txt file 
    with open(path + '1.txt', 'w') as f:
        with redirect_stdout(f):
            print('--'*20)
            # Print the total number of questions
            print("Total Questions: ", sum(total_predictions))
            # Print the total accuracy
            print("Total Accuracy: ", sum(total_accuracy) / sum(total_predictions))
            # Calculate the weighted accuracy
            type_count = 0
            for i in [eov_predictions, pov_predictions, se_predictions, miv_predictions]:
                if sum(i) > 0:
                    type_count += 1
            weight = 1 / type_count
            print("Weighted Accuracy: ", sum([sum(i) / sum(j) * weight if sum(j) > 0 else 0 for i, j in zip([eov_accuracy, pov_accuracy, se_accuracy, miv_accuracy], [eov_predictions, pov_predictions, se_predictions, miv_predictions])]))
            print("Total Option Balance: ", (sum((total_predictions / sum(total_predictions) - 0.25) ** 2) / 4) ** 0.5)
            print("Total Correct Option Balance: ", (sum((total_accuracy / sum(total_accuracy) - 0.25) ** 2 ) / 4) ** 0.5)
            # Calculate the jensen shannon divergence
            print("Total Jensen-Shannon Divergence: ", jensen_shannon_divergence(total_predictions / sum(total_predictions), total_gts / sum(total_gts)))
            # Print the number of illegal predictions
            print("Illegal Predictions: ", illegal_predictions)
            #print(total_gts / sum(total_gts))
            if sum(eov_predictions) > 0:
                print("EOV Questions: ", sum(eov_predictions))
                print("EOV Accuracy: ", sum(eov_accuracy) / sum(eov_predictions))
                print("EOV Option Balance: ", (sum((eov_predictions / sum(eov_predictions) - 0.25) ** 2) / 4) ** 0.5)
                print("EOV Correct Option Balance: ", (sum((eov_accuracy / sum(eov_accuracy) - 0.25) ** 2) / 4) ** 0.5)
                print("EOV Jensen-Shannon Divergence: ", jensen_shannon_divergence(eov_predictions / sum(eov_predictions), eov_gts / sum(eov_gts)))
                #print(eov_gts / sum(eov_gts))

            if sum(pov_predictions) > 0:
                print("POV Questions: ", sum(pov_predictions))
                print("POV Accuracy: ", sum(pov_accuracy) / sum(pov_predictions))
                print("POV Option Balance: ", (sum((pov_predictions / sum(pov_predictions) - 0.25) ** 2) / 4) ** 0.5)
                print("POV Correct Option Balance: ", (sum((pov_accuracy / sum(pov_accuracy) - 0.25) ** 2) / 4) ** 0.5)
                print("POV Jensen-Shannon Divergence: ", jensen_shannon_divergence(pov_predictions / sum(pov_predictions), pov_gts / sum(pov_gts)))
                #print(pov_gts / sum(pov_gts))

            if sum(se_predictions) > 0:
                print("SE Questions: ", sum(se_predictions))
                print("SE Accuracy: ", sum(se_accuracy) / sum(se_predictions))
                print("SE Option Balance: ", (sum((se_predictions / sum(se_predictions) - 0.25) ** 2) / 4) ** 0.5)
                print("SE Correct Option Balance: ", (sum((se_accuracy / sum(se_accuracy) - 0.25) ** 2) / 4) ** 0.5)
                print("SE Jensen-Shannon Divergence: ", jensen_shannon_divergence(se_predictions / sum(se_predictions), se_gts / sum(se_gts)))
                #print(se_gts / sum(se_gts))

            if sum(miv_predictions) > 0:
                print("MIV Questions: ", sum(miv_predictions))
                print("MIV Accuracy: ", sum(miv_accuracy) / sum(miv_predictions))
                print("MIV Option Balance: ", (sum((miv_predictions / sum(miv_predictions) - 0.25) ** 2) / 4) ** 0.5)
                print("MIV Correct Option Balance: ", (sum((miv_accuracy / sum(miv_accuracy) - 0.25) ** 2) / 4) ** 0.5)
                print("MIV Jensen-Shannon Divergence: ", jensen_shannon_divergence(miv_predictions / sum(miv_predictions), miv_gts / sum(miv_gts)))
                #print(miv_gts / sum(miv_gts))
                

    # save the wrong predictions in a json with proper formatting
    with open(path + 'wrong_predictions.json', 'w') as f:
        json.dump(wrong_predictions, f, indent=4)

