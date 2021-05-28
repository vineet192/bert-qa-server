from flask import Flask
from flask import request
from flask import jsonify
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
import numpy as np
import nltk
import json
app = Flask(__name__)

tokenizer = BertWordPieceTokenizer(
    vocab='./mymodel/assets/vocab.txt', lowercase=True)
max_seq_length = 384
nltk.download('punkt')
model = tf.keras.models.load_model('./mymodel')


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/get-answer')
def get_answer():
    request_data = request.get_json()
    print('Recieved questions : ')
    for question in request_data['qas']:
        print(question['question'])
    data = {"data":
            [
                {"title": "Project Apollo",
                 "paragraphs": [
                     {
                         "context": request_data['paragraph'],
                         "qas": request_data['qas']
                     }]}]}


    processedData = preprocess(data)
    x, y = create_inputs_targets(processedData)
    pred_start, pred_end = model.predict(x)
    answers = extract_answers(pred_start, pred_end, processedData)
    return jsonify(answers)


def extract_answers(pred_start, pred_end, processedData):
    answers = []
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        test_sample = processedData[idx]
        offset = test_sample['context_token_to_char']
        start = np.argmax(start)
        end = np.argmax(end)
        pred_ans = None
        if start >= len(offset):
            continue
        pred_char_start = offset[start][0]

        if end < start:
            # Get the sentence
            lines = nltk.sent_tokenize(test_sample['raw_context'])
            line_length = 0
            print(test_sample['raw_context']
                  [offset[start][0]:offset[start][1]])
            for i, line in enumerate(lines):
                line_length += len(line)
                if offset[start][0] < line_length:
                    pred_ans = line
                    break
        elif end < len(offset):
            pred_ans = test_sample['raw_context'][pred_char_start:offset[end][1]]
        else:
            pred_ans = test_sample['raw_context'][pred_char_start:]

        answers.append({"qn": test_sample['raw_question'],
                        "ans": pred_ans})
    return answers


def preprocess(data):
    processed_data = []
    for item in data['data']:
        for para in item['paragraphs']:
            context = para['context']
            for qas in para['qas']:
                proc_data = {
                    'start_token_idx': -1,
                    'end_token_idx': -1
                }
                answer = None
                try:
                    answer = qas['answers'][0]['text']
                    answer_start = qas['answers'][0]['answer_start']
                except:
                    pass
                question = qas['question']
                context = " ".join(str(context).split())
                question = " ".join(str(question).split())
                proc_data['context'] = tokenizer.encode(context)
                proc_data['question'] = tokenizer.encode(question)
                proc_data['raw_context'] = context
                proc_data['raw_question'] = question

                if answer is not None:

                    answer_end = len(answer) + answer_start

                    # If the end of answer exceeds context.
                    if(answer_end >= len(context)):
                        continue

                    # Array of characters indicating where the answer is in the context.
                    answer_char_indices = [0] * len(context)
                    for idx in range(answer_start, answer_end):
                        answer_char_indices[idx] = 1

                    # Storing the encoded legal answer offsets (start and stop of encoded answer)
                    ans_token_idx = []
                    for idx, (start, end) in enumerate(proc_data['context'].offsets):
                        if sum(answer_char_indices[start:end]) > 0:
                            ans_token_idx.append(idx)

                    # skip if there are no legal answers.
                    if len(ans_token_idx) == 0:
                        continue

                    # Storing the start and end index of the tokenized(encoded) answer gotten from the context.
                    proc_data['start_token_idx'] = ans_token_idx[0]
                    proc_data['end_token_idx'] = ans_token_idx[-1]

                # Setting up input_word_ids, input_type_ids, input_mask.
                input_ids = proc_data['context'].ids + \
                    proc_data['question'].ids[1:]
                token_type_ids = [
                    0] * len(proc_data['context'].ids) + [1] * len(proc_data['question'].ids[1:])
                attention_mask = [1] * len(input_ids)
                padding_length = max_seq_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + ([0] * padding_length)
                    attention_mask = attention_mask + ([0] * padding_length)
                    token_type_ids = token_type_ids + ([0] * padding_length)

                # Skip if padding length is 0.
                # elif padding_length < 0:
                #     print('Sequence length is', len(input_ids))
                #     continue
                proc_data['input_word_ids'] = input_ids
                proc_data['input_type_ids'] = token_type_ids
                proc_data['input_mask'] = attention_mask
                proc_data['context_token_to_char'] = proc_data['context'].offsets

                
                print('Sequence length is', len(input_ids))
                processed_data.append(proc_data)
    return processed_data


def create_inputs_targets(processed_data):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in processed_data:
        for key in dataset_dict:
            dataset_dict[key].append(item[key])
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"],
         dataset_dict["input_mask"],
         dataset_dict["input_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y
