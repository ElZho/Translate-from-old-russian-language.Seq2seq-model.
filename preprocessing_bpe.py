import unicodedata
import string
import re
import random
import torch
import itertools
import youtokentome as yttm
import tqdm
from tqdm import tqdm

SOS_token = 1
EOS_token = 0
#MAX_LENGTH = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s).lower().strip()
    s = re.sub(r"([.!?;,])", r"", s)
    #s = re.sub(r"([.!?;,])", r"", s)
    return s   

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

    return train_pairs, test_pairs 


def filterPair(p, MAX_LENGTH):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH      


def filterPairs(pairs, MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH)]

def transform_words(pairs):
  src=[pair[0] for pair in tqdm(pairs) if len(pair)>1]
  trg=[pair[1] for pair in tqdm(pairs) if len(pair)>1]
        
  return  src, trg

# def addWord(word):
#   vocab={}
#   n_words=0
#   if word not in vocab:
#     vocab.update({word: 1})
#     n_words += 1
#   else:
#     vocab[word] += 1 


# def Count_words(pairs):
#   vocab_src = [addWord(word) for word in pair[0].split(' ') for pair in pairs]
#   vocab_trg = [addWord(word) for word in pair[1].split(' ') for pair in pairs]
          
def save_texts_to_file(texts, out_file):
    with open(out_file, 'w') as outf:
        outf.write('\n'.join(texts))

def prepareData(lang1, lang2, path, MAX_LENGTH=40, reverse=False):
    
    train_pairs, test_pairs  = readLangs(lang1, lang2, path, reverse)
    print("Read %s sentence pairs train" % len(train_pairs))
    print("Read %s sentence pairs test" % len(test_pairs))
    train_pairs = filterPairs(train_pairs, MAX_LENGTH)
    test_pairs = filterPairs(test_pairs, MAX_LENGTH)
    print("Trimmed to %s sentence train pairs" % len(train_pairs))
    print("Trimmed to %s sentence test pairs" % len(test_pairs))
    print("Counting words...")
    
    train_src, train_trg =transform_words(train_pairs)
    test_src, test_trg=transform_words(test_pairs)

    train_src_path = '/content/train_src_bpe.txt'
    train_trg_path = '/content/train_trg_bpe.txt'
    train_model_src_path = "/content/train_bpe_src_model"
    train_model_trg_path = "/content/train_bpe_trg_model"

    test_src_path = '/content/test_src_bpe.txt'
    test_trg_path = '/content/test_trg_bpe.txt'
    test_model_src_path = "/content/test_bpe_src_model"
    test_model_trg_path = "/content/test_bpe_trg_model"   

    save_texts_to_file(train_src, train_src_path)
    save_texts_to_file(train_trg, train_trg_path)
    save_texts_to_file(test_src, test_src_path)
    save_texts_to_file(test_trg, test_trg_path)

    # Training model for train dataset
    yttm.BPE.train(data=train_src_path, vocab_size=200, model=train_model_src_path, pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    yttm.BPE.train(data=train_trg_path, vocab_size=200, model=train_model_trg_path, pad_id=0, unk_id=1, bos_id=2, eos_id=3)   

    # Training model for train dataset
    yttm.BPE.train(data=test_src_path, vocab_size=200, model=test_model_src_path, pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    yttm.BPE.train(data=test_trg_path, vocab_size=200, model=test_model_trg_path, pad_id=0, unk_id=1, bos_id=2, eos_id=3) 

    # Loading model
    tokenazer_train_src = yttm.BPE(model=train_model_src_path)
    tokenazer_train_trg = yttm.BPE(model=train_model_trg_path) 
    tokenazer_test_src = yttm.BPE(model=test_model_src_path)
    tokenazer_test_trg = yttm.BPE(model=test_model_trg_path)                                                     
    
    return train_pairs, tokenazer_train_src, tokenazer_train_trg, test_pairs, tokenazer_test_src, tokenazer_test_trg


def tensorFromSentence(tokenazer, sentence, MAX_LENGTH=40):
    indexes = tokenazer.encode(sentence)
    if len(indexes)>MAX_LENGTH-1:
      indexes=indexes[:MAX_LENGTH-1]
    indexes.append(0)        
    if len(indexes)<MAX_LENGTH:
      indexes.extend(itertools.repeat(0, (MAX_LENGTH-len(indexes))))
    return torch.tensor(indexes, dtype=torch.long, device=device)

def get_len(sentence, MAX_LENGTH=40):
    return min(len(sentence.split(' ')), MAX_LENGTH)

def tensorsFromPair(pair, tokenazer_src, tokenazer_trg, MAX_LENGTH=40):
    input_len=get_len(pair[0], MAX_LENGTH)
    input_tensor = tensorFromSentence(tokenazer_src, pair[0], MAX_LENGTH)
    target_tensor  = tensorFromSentence(tokenazer_trg, pair[1], MAX_LENGTH)
    return (input_tensor, input_len, target_tensor)    