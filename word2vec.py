from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
model = Word2Vec(
    LineSentence('data/simple.reg.txt'),
    size=400,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count() - 2,
         )


outp1 = 'data/simple.zh.text.model'
outp2 = 'data/simple.zh.text.vector'
model.save(outp1)
model.wv.save_word2vec_format(outp2)
