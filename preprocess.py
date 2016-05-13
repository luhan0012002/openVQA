import json
import argparse

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='lang', default='src', help='Preprocess src/tgt')
opts = parser.parse_args()

print 'preprocessing ' + opts.lang


train = '/media/iyu/caremedia4/poyaoh/HW4/data/train.' + opts.lang
#src_dev = '/media/iyu/caremedia4/poyaoh/HW4/data/dev.src'
#src_test = '/media/iyu/caremedia4/poyaoh/HW4/test.src'

#tgt_train = '/media/iyu/caremedia4/poyaoh/HW4/data/train.tgt'
#tgt_dev = '/media/iyu/caremedia4/poyaoh/HW4/data/dev.tgt'
#tgt_test = '/media/iyu/caremedia4/poyaoh/HW4/test.tgt'


f_itow = '/media/iyu/caremedia4/luhan/HW4/vocab/idx2' + opts.lang + '.json'
f_wtoi = '/media/iyu/caremedia4/luhan/HW4/vocab/' + opts.lang + '2idx.json'


#tgt_itow = '/media/iyu/caremedia4/poyaoh/HW4/idx2tgt.json'
#tgt_wtoi = '/media/iyu/caremedia4/poyaoh/HW4/tgt2idx.json'

#vocab = '/home/ubuntu/Data/luhan/data/vocab.json'
count_thr = 0

with open(train) as f:
    sents = f.readlines()
    word_counts = {}
    words = []
    for sent in sents:
        words = sent.split()
        for w in words:
            if w not in word_counts:
                word_counts[w] = 1
            else:
                word_counts[w] += 1

    total_words = sum(word_counts.itervalues())
    bad_words = [w for w,n in word_counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in word_counts.iteritems() if n > count_thr]

    #if len(bad_words) > 0:
    #    vocab.append('UNK')
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

print('# of vocab: ', len(vocab))
print('# of infrequent words: ', len(bad_words))


with open(f_itow, 'w') as f:
    json.dump(itow, f)
with open(f_wtoi, 'w') as f:
    json.dump(wtoi, f)
#with open(vocab, 'w') as f:
#    json.dump(vocab, f)

