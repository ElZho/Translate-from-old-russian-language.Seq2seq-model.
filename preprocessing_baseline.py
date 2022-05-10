import unicodedata
import string
import re
import random
import torch


SOS_token = 0
EOS_token = 1
# MAX_LENGTH = 340

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s).lower().strip()
    s = re.sub(r"([.!?;,])", r" \1", s)
    s = re.sub(r"([.!?;,])", r"", s)
    return s   

def create_langs(lang1, lang2, pairs, reverse=False):
  # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang

def readLangs(lang1, lang2, path, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    with open(path, encoding='utf-8') as file:
        lines = file.read()

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t') if len(s)>1] for l in lines.split('\n')]
    pairs=[pair for pair in pairs if len(pair)>1 ]
    random.shuffle(pairs)

    TRAIN_VAL_SPLIT = int(len(pairs) * 0.7)
    train_pairs = pairs[:TRAIN_VAL_SPLIT]
    test_pairs = pairs[TRAIN_VAL_SPLIT:]

    train_input_lang, train_output_lang= create_langs(lang1, lang2, train_pairs, reverse=False)
    test_input_lang, test_output_lang= create_langs(lang1, lang2, test_pairs, reverse=False)

    # Reverse pairs, make Lang instances
    # if reverse:
    #     pairs = [list(reversed(p)) for p in pairs]
    #     input_lang = Lang(lang2)
    #     output_lang = Lang(lang1)
    # else:
    #     input_lang = Lang(lang1)
    #     output_lang = Lang(lang2)

    return train_input_lang, train_output_lang, train_pairs, test_input_lang,\
           test_output_lang, test_pairs 


def filterPair(p, MAX_LENGTH=340):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH      


def filterPairs(pairs, MAX_LENGTH=340):
    return [pair for pair in pairs if filterPair(pair)]

def Count_words(pairs, input_lang, output_lang):
  for pair in pairs:
    if len(pair)>1:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1]) 
  return  input_lang, output_lang        

def prepareData(lang1, lang2, path, MAX_LENGTH=340, reverse=False):
    train_input_lang, train_output_lang, train_pairs, test_input_lang,\
           test_output_lang, test_pairs  = readLangs(lang1, lang2, path, reverse)
    print("Read %s sentence pairs train" % len(train_pairs))
    print("Read %s sentence pairs test" % len(test_pairs))
    train_pairs = filterPairs(train_pairs, MAX_LENGTH=340)
    test_pairs = filterPairs(test_pairs, MAX_LENGTH=340)
    print("Trimmed to %s sentence train pairs" % len(train_pairs))
    print("Trimmed to %s sentence test pairs" % len(test_pairs))
    print("Counting words...")
    # for pair in pairs:
    #     input_lang.addSentence(pair[0])
    #     output_lang.addSentence(pair[1])
    train_input_lang, train_output_lang=Count_words(train_pairs, train_input_lang,\
                                                               train_output_lang)
    test_input_lang, test_output_lang=Count_words(test_pairs, test_input_lang,\
                                                               test_output_lang)                                                           
    print("Counted words train:")
    print(train_input_lang.name, train_input_lang.n_words)
    print(train_output_lang.name, train_output_lang.n_words)
    print("Counted words test:")
    print(test_input_lang.name, test_input_lang.n_words)
    print(test_output_lang.name, test_output_lang.n_words)
    
    return train_input_lang, train_output_lang, train_pairs, test_input_lang,\
           test_output_lang, test_pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)    