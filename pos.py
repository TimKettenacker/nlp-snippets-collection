import pandas
from collections import Counter, defaultdict
tag_sequence = (('ADV', 'NOUN', '.', 'ADV', '.', 'VERB', 'ADP', 'ADJ', 'NOUN', 'CONJ', 'VERB', 'ADJ', 'NOUN', '.', 'DET', 'VERB', 'ADJ', 'PRT', 'ADP', 'NUM', 'NOUN', '.', 'PRON', 'VERB', 'PRT', 'VERB', 'NOUN', '.', 'VERB', 'NOUN', 'NUM', '.', 'NUM', '.', '.', '.'), ('ADP', 'ADV', 'NUM', 'NOUN', '.', 'DET', 'NOUN', 'CONJ', 'DET', 'NOUN', 'VERB', 'ADP', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN', 'ADP', 'NOUN', 'CONJ', 'NOUN', '.', 'NOUN', '.', 'NOUN', '.', 'NOUN', '.', 'VERB', '.', 'VERB', 'ADP', 'NOUN', '.', 'VERB', '.', 'VERB', '.', 'VERB', '.', 'VERB', '.'), ('ADJ', 'NOUN'), ('ADV', '.', 'NOUN', '.', 'PRON', 'VERB', 'VERB', 'ADP', 'DET', 'NOUN', '.', 'ADP', 'DET', 'NOUN', '.'), ('ADV', 'DET', 'ADJ', 'NOUN', 'DET', 'NOUN', 'CONJ', 'NOUN', 'ADP', 'NOUN', 'VERB', 'ADJ', '.', 'NOUN', 'VERB', 'DET', 'ADJ', 'VERB', 'NOUN', 'NOUN', 'ADP', 'NOUN', 'NUM', '.', 'NUM', '.'))
word_sequence = (('Whenever', 'artists', ',', 'indeed', ',', 'turned', 'to', 'actual', 'representations', 'or', 'molded', 'three-dimensional', 'figures', ',', 'which', 'were', 'rare', 'down', 'to', '800', 'B.C.', ',', 'they', 'tended', 'to', 'reflect', 'reality', '(', 'see', 'Plate', '6a', ',', '9b', ')', ';', ';'), ('For', 'almost', 'two', 'months', ',', 'the', 'defendant', 'and', 'the', 'world', 'heard', 'from', 'individuals', 'escaped', 'from', 'the', 'grave', 'about', 'fathers', 'and', 'mothers', ',', 'graybeards', ',', 'adolescents', ',', 'babies', ',', 'starved', ',', 'beaten', 'to', 'death', ',', 'strangled', ',', 'machine-gunned', ',', 'gassed', ',', 'burned', '.'), ('Clearer', 'meaning'), ('Yes', ',', 'gentlemen', ',', 'I', 'am', 'getting', 'to', 'the', 'point', ',', 'to', 'my', 'point', '.'), ('About', 'the', 'same', 'time', 'the', 'Alleghenies', 'and', 'Poconos', 'in', 'Pennsylvania', 'are', 'magnificent', '--', 'Renovo', 'holds', 'its', 'annual', 'Flaming', 'Foliage', 'Festival', 'on', 'Oct.', '14', ',', '15', '.'))

def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.

    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    assert len(sequences_A) == len(sequences_B), \
        "length of input parameters should be the same"
    d = defaultdict(Counter)
    for i in range(len(sequences_A)):
        for tag, word in zip(sequences_A[i], sequences_B[i]):
            d[tag][word] += 1
    return d
    raise NotImplementedError

emission_counts = pair_counts(tag_sequence, word_sequence)
emission_counts['NOUN']['point']

# now the preparation for the most common tag per word
word_counts = pair_counts(word_sequence, tag_sequence)
mfc_table = pandas.DataFrame(columns=word_counts.keys(), index=range(0,1))
for k in word_counts:
    mfc_table.iloc[0][k] = word_counts[k].most_common()[0][0]


def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.

    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    d = defaultdict(int)
    for s in range(len(sequences)):
        for obj in (sequences[s]):
            d[obj] += 1
    return d
    raise NotImplementedError


def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.

    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
    flat_sequence = [item for sublist in sequences for item in sublist]
    return Counter(zip(flat_sequence, flat_sequence[1:]))
    raise NotImplementedError


def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.

    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    d = defaultdict(int)
    for s in range(len(sequences)):
        d[(sequences[s][0])] += 1
    return d
    raise NotImplementedError


def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.

    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """
    d = defaultdict(int)
    for s in range(len(sequences)):
        d[sequences[s][len(sequences[s]) - 1]] += 1
    return d
    raise NotImplementedError


# transition probabilities: probability that a part of speech follows another part of speech -
# how often does it happen that a noun follows a modal, ...
# transition probability can be calculated as (noun, modal) sequence occurs 3 times and there are 9 nouns -
# the transition probability is 3/9
# emission probabilities: probability that a noun is the word "shower", a modal the word "can", ...
# let's say there is a sentence with "will" in it -
# emission probability can be calculated as "will"/all nouns in a given set, then something like 2/103 -
# then the emission probability that a noun is "will" is 2/103


basic_model = HiddenMarkovModel(name="base-hmm-tagger")

# TODO: create states with emission probability distributions P(word | tag) and add to the model
# (Hint: you may need to loop & create/add new states)
tag_states = []
ctw = pair_counts(tag_sequence, word_sequence)
for tag in data_tagset:
    tag_distribution = {}
    for word in ctw[tag]:
        tag_distribution[word] = ctw[tag][word] / unigram_counts(tag_sequence)[tag]
    tag_states.append(State(DiscreteDistribution(tag_distribution), name = tag))

basic_model.add_states(tag_states)

# TODO: add edges between states for the observed transition frequencies P(tag_i | tag_i-1)
# (Hint: you may need to loop & add transitions
basic_model.add_transition()


# NOTE: YOU SHOULD NOT NEED TO MODIFY ANYTHING BELOW THIS LINE
# finalize the model
basic_model.bake()

