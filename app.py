import json

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, Response
from tokenizers import BertWordPieceTokenizer
import nltk
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

nltk.download('punkt')
tokenizer = BertWordPieceTokenizer(
    vocab='./mymodel/assets/vocab.txt', lowercase=True)
max_seq_length = 512
model = tf.keras.models.load_model('./mymodel')


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.get('/get-answer')
def get_answer():
    req = request.get_json()
    question = req['question']
    context = req['context']

    answer = predict_answer(question, context)
    if answer == None:
        answer = 'Sorry the answer could not be found'
    return jsonify(answer=answer)




def preprocess_context(raw_context, tokenized_question_length):
    #This function takes in the raw context (string), tokenizes and divides it into multiple chunks of max_seq_length and returns them as
    # 2 dimensional arrays. The same is done for the offsets as well.

    tokenized_context = tokenizer.encode(raw_context)
    chunk_length = max_seq_length - tokenized_question_length - 2
    tokens_length = len(tokenized_context)
    if len(tokenized_context.ids) <= chunk_length:
        return [tokenized_context.ids], [tokenized_context.offsets]

    raw_context_sentences = sent_tokenize(raw_context)

    return sentences_to_chunks(raw_context_sentences, chunk_length)
    
    # #Removing special [SEP] and [CLS] tokens at 0th and n-1th indices, as they need to be added seperaely for each chunk.
    # stripped_context_tokens = tokenized_context.ids[1:tokens_length - 1]
    # stripped_context_offsets = tokenized_context.offsets[1:tokens_length - 1]

    # context_id_chunks = [stripped_context_tokens[i:i + chunk_length]
    #                      for i in range(0, len(stripped_context_tokens), chunk_length)]
    # context_offset_chunks = [stripped_context_offsets[i:i + chunk_length]
    #                          for i in range(0, len(stripped_context_offsets), chunk_length)]

    # for context_id_chunk, context_offset_chunk in zip(context_id_chunks, context_offset_chunks):
        
    #     #Add [SEP] and [CLS] tokens and their respective offsets which have a default value of (0,0)
    #     context_id_chunk = [101] + context_id_chunk + [102]
    #     context_offset_chunk = [(0, 0)] + context_offset_chunk + [(0,0)]


    #return context_id_chunks, context_offset_chunks

def predict_answer(raw_question, raw_context):
    #Takes in a raw question and raw context and returns an answer in raw string format.

    tokenized_question = tokenizer.encode(raw_question)
    tokenized_context_chunks, context_offset_chunks, raw_sentence_chunks = preprocess_context(raw_context, len(tokenized_question))
    print(type(tokenized_context_chunks))
    x = create_input_targets(tokenized_question, tokenized_context_chunks)

    pred_start, pred_end = model.predict(x).values()
    final_answer = None
    confidence = -1

    for chunk_idx, (context_offset_chunk, raw_sentence_chunk) in enumerate(zip(context_offset_chunks, raw_sentence_chunks)):
        print('Searching in chunk', chunk_idx+1, "...", len(context_offset_chunk))
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            offsets = context_offset_chunk
            chunk_confidence = max(start) + max(end)
            start = np.argmax(start)
            end = np.argmax(end)
            pred_ans = None

            if start >= len(offsets):
                continue
            if start == 0 and end == 0:
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_ans = raw_sentence_chunk[pred_char_start:offsets[end][1]]
            else:
                pred_ans = raw_sentence_chunk[pred_char_start:]
            print('answer : ', pred_ans, chunk_confidence)
            if chunk_confidence > confidence:
                    confidence = chunk_confidence
                    final_answer = pred_ans
                
    return final_answer
        
        
def sentences_to_chunks(sentences,chunk_length):
    tokenized_sentences = list(map(tokenizer.encode, sentences))
    size = 0
    raw_chunk = ''
    chunk_lookup = {
        'chunks' : [],
        'raw_sent' : [],
        'curr_chunk' : [],
        'curr_chunk_offsets' : [],
        'chunk_offsets' : []
    }
    for sent, raw_sent in zip(tokenized_sentences, sentences):
        sent_ids = sent.ids[1:len(sent.ids) - 1]
        sent_offsets = sent.offsets[1:len(sent.offsets) - 1]
        
        if size + len(sent_ids) > chunk_length:
            print('Chunk lengths : ', len(chunk_lookup['curr_chunk_offsets']), len(chunk_lookup['curr_chunk']))
            chunk_lookup['curr_chunk'] = tokenizer.encode(raw_chunk).ids
            chunk_lookup['curr_chunk_offsets'] = tokenizer.encode(raw_chunk).offsets
            padding_length = chunk_length - size
            if padding_length > 0:
                chunk_lookup['curr_chunk'] += [0] * padding_length

            #updating id chunk and offset chunk
            chunk_lookup['chunks'].append(chunk_lookup['curr_chunk'])
            chunk_lookup['chunk_offsets'].append(chunk_lookup['curr_chunk_offsets'])

            #reset the current chunks and offsets
            chunk_lookup['curr_chunk'] = []
            chunk_lookup['curr_chunk_offsets'] = []
            size = 0
            chunk_lookup['raw_sent'].append(raw_chunk)
            raw_chunk = ''
        
        size += len(sent_ids)
        raw_chunk += raw_sent 

    padding_length = chunk_length - size
    print('Chunk lengths outside loop: ', len(chunk_lookup['curr_chunk_offsets']), len(chunk_lookup['curr_chunk']))
    chunk_lookup['curr_chunk'] = tokenizer.encode(raw_chunk).ids
    chunk_lookup['curr_chunk_offsets'] = tokenizer.encode(raw_chunk).offsets
    if padding_length > 0:
        chunk_lookup['curr_chunk'] += [0] * padding_length
        
    #updating id chunk and offset chunk
    chunk_lookup['chunks'].append(chunk_lookup['curr_chunk'])
    chunk_lookup['chunk_offsets'].append(chunk_lookup['curr_chunk_offsets'])
    chunk_lookup['raw_sent'].append(raw_chunk)
        
    
    return chunk_lookup['chunks'], chunk_lookup['chunk_offsets'], chunk_lookup['raw_sent']

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
            input_word_ids = input_word_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            input_type_ids = input_type_ids + ([0] * padding_length)

        x['input_word_ids'].append(input_word_ids)
        x['input_type_ids'].append(input_type_ids)
        x['input_mask'].append(input_mask)

    x = {k: np.array(v) for k, v in x.items()}

    return x

