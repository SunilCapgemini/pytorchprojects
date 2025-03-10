import re
import unicodedata
import torch
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler,DataLoader
import timecalculation
import torch.nn as nn
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self,name) -> None:
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'SOS',1:'EOS'}
        self.n_words = 2 # Count SOS and EOS

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
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r" \1",s)
    s = re.sub(r"[^a-zA-Z!?]+",r" ",s)
    return s.strip()

def readLangs(lang1,lang2, reverse=False):
    print('Reading lines...')

    lines = open('./data/%s-%s.txt' % (lang1,lang2),encoding='utf-8')\
        .read().strip().split('\n')
    
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if(reverse):
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang,output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ","i m","he is", "he s ","she is ","you are","you re","we are","we re","they are","they re"
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
    len(p[1].split(' ')) < MAX_LENGTH and \
    p[1].startswith(eng_prefixes)

    
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1,lang2,reverse)
    print('Actual length',len(pairs))
    pairs = filterPairs(pairs)
    print('After filtering pairs length',len(pairs))
    #Counting words
    print('counting words ...')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print('Counted words')
    print("No of french words",input_lang.name,input_lang.n_words)
    print("No of english words",output_lang.name,output_lang.n_words)
    return input_lang,output_lang, pairs
# input_lang,output_lang, pairs = prepareData('eng','fra',True)

def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def get_dataloader(batch_size):
    input_lang,output_lang,pairs = prepareData('eng','fra',True)
    n = len(pairs)
    input_ids = np.zeros((n,MAX_LENGTH),dtype=np.int32)
    target_ids = input_ids.copy()

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang,inp)
        tgt_ids = indexesFromSentence(output_lang,tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx,:len(inp_ids)] = inp_ids
        target_ids[idx,:len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids),
                               torch.LongTensor(target_ids))
    train_sampler =  RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)
    return input_lang, output_lang,train_dataloader

hidden_size = 128
batch_size = 32

# input_lang, output_lang, train_dataloader = get_dataloader(batch_size)




# for data in train_dataloader:
#     print(data[1])
#     print(data[1].shape)
#     print(len(data[1]))
#     for x11 in range(data[1].shape[0]):
#         print([output_lang.index2word[word.item()] for word in data[1][x11]])
#     break
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)