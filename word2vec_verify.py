import gensim

model = gensim.models.Word2Vec.load('data/simple.zh.text.model')

result = model.most_similar('男朋友', topn=10)

for each in result:
    print(each)

