import sys
import os
import json
import time
from nlp_tools import tokenize_words, build_prob_language_model, generate_text_using_trigrams
from files_io import load_text, serialize, deserialize


if __name__ == '__main__':
    args = sys.argv[1:]
    N_WORDS = args[0]
    FILES_FOLDER = 'files'
    TMP_FOLDER = os.path.join(FILES_FOLDER, 'tmp')
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    #print("loading text..")
    T0 = time.time()
    source_text_filename = 'eng_wiki.txt'
    source_text_path = os.path.join(FILES_FOLDER, source_text_filename)
    en_text = load_text(source_text_path)
    T1 = time.time()
    #print("\tdone in {0:.2f} sec".format(T1-T0))

    tmp_folder_content = os.listdir(TMP_FOLDER)
    # If temp folder doesn't yet contain serialized token list, tokenize text and serialize tokens
    if len(tmp_folder_content) == 0:
        #print("tokenizing words..")
        tokens = tokenize_words(en_text)
        tokenized_words_filename = 'tokenized_' + source_text_filename
        tokenized_words_path = os.path.join(TMP_FOLDER, tokenized_words_filename)
        serialize(tokens, tokenized_words_path)

    # Otherwise, deserialize tokens list. Much faster, as no need to wait tokenization time
    else:
        #print("temporary file found. Deserializing words..")
        tokenized_words_filename = tmp_folder_content[0]
        tokenized_words_path = os.path.join(TMP_FOLDER, tokenized_words_filename)
        tokens = deserialize(tokenized_words_path)
    
    T2 = time.time()  
    #print("\tdone in {0:.2f} sec".format(T2-T1))

    #print("building probabilistic language model..")
    language_model = build_prob_language_model(tokens)
    T3 = time.time()
    #print("\tdone in {0:.2f} sec".format(T3-T2))

    #print("generating output text..")
    final_text = generate_text_using_trigrams(language_model, N_WORDS)
    T4 = time.time()
    #print("\tdone in {0:.2f} sec".format(T4-T3))

    #print("\ntotal execution time: {0:.2f} sec".format(T4-T0))
    response = {'error': 'false', 'data': final_text}
    print(json.dumps(response))
