import json

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tokenizers import BertWordPieceTokenizer

app = Flask(__name__)

tokenizer = BertWordPieceTokenizer(
    vocab='./mymodel/assets/vocab.txt', lowercase=True)
max_seq_length = 512
model = tf.keras.models.load_model('./mymodel')


@app.route('/')
def hello_world():
    return 'Hello, World!'



def preprocess_context(raw_context, tokenized_question_length):
    #This function takes in the raw context (string), tokenizes and divides it into multiple chunks of max_seq_length and returns them as
    # 2 dimensional arrays. The same is done for the offsets as well.

    tokenized_context = tokenizer.encode(raw_context)
    chunk_length = max_seq_length - tokenized_question_length - 2
    tokens_length = len(tokenized_context)

    if len(tokenized_context.ids) <= chunk_length:
        return [tokenized_context]
    
    #Removing special [SEP] and [CLS] tokens at 0th and n-1th indices, as they need to be added seperaely for each chunk.
    stripped_context_tokens = tokenized_context.ids[1:tokens_length - 1]
    stripped_context_offsets = tokenized_context.offsets[1:tokens_length - 1]

    context_id_chunks = [stripped_context_tokens[i:i + chunk_length]
                         for i in range(0, len(stripped_context_tokens), chunk_length)]
    context_offset_chunks = [stripped_context_offsets[i:i + chunk_length]
                             for i in range(0, len(stripped_context_offsets), chunk_length)]

    for context_id_chunk, context_offset_chunk in zip(context_id_chunks, context_offset_chunks):
        
        #Add [SEP] and [CLS] tokens and their respective offsets which have a default value of (0,0)
        context_id_chunk = [101] + context_id_chunk + [102]
        context_offset_chunk = [(0, 0)] + context_offset_chunk + [(0,0)]


    return context_id_chunks, context_offset_chunks

def predict_answer(raw_question, raw_context):
    #Takes in a raw question and raw context and returns an answer in raw string format.

    tokenized_question = tokenizer.encode(raw_question)
    tokenized_context_chunks, context_offset_chunks = preprocess_context(raw_context, len(tokenized_question))

    x = create_input_targets(tokenized_question, tokenized_context_chunks)

    pred_start, pred_end = model.predict(x)

    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        offsets = context_offset_chunks[idx]
        start = np.argmax(start)
        end = np.argmax(end)
        pred_ans = None

        if start >= len(offsets):
            continue
        if start == 0 and end == 0:
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_ans = raw_context[pred_char_start:offsets[end][1]]
            return pred_ans
        else:
            pred_ans = raw_context[pred_char_start:]
    return None
        
        


def create_input_targets(tokenized_question, tokenized_context_chunks):
    #Takes in a raw question and tokenized context and returns the input target for the BERT model to predict upon.

    # Setting up input_word_ids, input_type_ids, input_mask.
    x = dict(
        input_word_ids = [],
        input_type_ids = [],
        input_mask = []
    )
    for tokenized_context_array in tokenized_context_chunks:
        input_word_ids = tokenized_context_array + tokenized_question.ids[1:]
        input_type_ids = [0] * len(tokenized_context_array) + [1] * len(tokenized_question.ids[1:])
        input_mask = [1] * len(input_word_ids)

        #Add the padding
        padding_length = max_seq_length - len(input_word_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

        x['input_word_ids'].append(input_word_ids)
        x['input_type_ids'].append(input_type_ids)
        x['input_mask'].append(input_mask)
        
    x = {k: np.array(v) for k, v in x.items()}

    return x

