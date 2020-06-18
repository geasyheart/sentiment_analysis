import gensim
from gensim.models import KeyedVectors

model = gensim.models.Word2Vec.load('data/simple.zh.text.model')

result = model.most_similar('男朋友', topn=10)

for each in result:
    print(each)

print('2'.center(50, '-'))
model2 = KeyedVectors.load_word2vec_format('data/simple.zh.text.vector')
result = model2.most_similar('男朋友', topn=10)

for each in result:
    print(each)
