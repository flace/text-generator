import nltk
import codecs
import sys
import json


def load_and_tokenize(path):
    with codecs.open(path, 'r', 'utf-8') as in_file:
        text = in_file.read()
    return text


def build_prob_language_model(input_words):
    """
    Build probabilistic language model from a given list of words.
    """
    # Bigrams: conditional frequency distribution
    cfreq_2grams = nltk.ConditionalFreqDist(nltk.bigrams(input_words))
    # Converting trigrams to tuple (word1, word2), word3,
    # and using (word1, word2) as a condition for conditional frequency distribution
    trigrams = list(nltk.trigrams(input_words))
    conditional_trigrams = [ ((tr[0],tr[1]), tr[2]) for tr in trigrams ]
    cfreq_3grams = nltk.ConditionalFreqDist(conditional_trigrams)
    # using Maximum LikelihoodEstimator to estimate conditional PROBABILITY distribution for bigrams and trigrams,
    # which generated the corresponding FREQUENCY distributions.
    cprob_2grams = nltk.ConditionalProbDist(cfreq_2grams, nltk.MLEProbDist)
    cprob_3grams = nltk.ConditionalProbDist(cfreq_3grams, nltk.MLEProbDist)
    language_model = (cprob_2grams, cprob_3grams)
    return language_model


def generate_text_using_trigrams(language_model, N_output_words):
    """
    Generate N words using probabilistic language model.
    Trigrams are used to generate text.
    """
    cprob_2grams, cprob_3grams = language_model
    # Starting from the end-of-sentence token, generating 2nd token using bigrams, and the rest - using trigrams
    word1 = '.'
    word2 = cprob_2grams[word1].generate()
    generated_words = [word1, word2]
    for idx in range(int(N_output_words)):
        word3 = cprob_3grams[(word1, word2)].generate()
        generated_words.append(word3)
        word1 = word2
        word2 = word3
    return ' '.join(generated_words[1:])


if __name__ == '__main__':
    args = sys.argv[1:]
    N_WORDS = args[0]
    path = 'files/eng_wiki.txt'
    en_text = load_and_tokenize(path)
    en_words = nltk.word_tokenize(en_text)
    en_language_model = build_prob_language_model(en_words)
    final_text = generate_text_using_trigrams(en_language_model, N_WORDS)
    response = {'error': 'false', 'data': final_text}
    print(json.dumps(response))
