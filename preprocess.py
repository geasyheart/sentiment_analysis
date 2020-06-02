import re
import jieba

pattern = re.compile(r'<content>(.*?)<\/content>')


def read_data():
    stop_words = []
    with open('data/baidu_stopwords.txt', 'r') as f:
        for line in f:
            stop_words.append(line.replace("\n", ''))
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    with open('data/news_tensite_xml.dat', 'rb') as f:
        for line in f:
            try:
                new_line = line.decode("GB18030").encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                new_line = line.decode("utf-8")

            content = re.findall(pattern, new_line)
            if not content:
                continue
            content = " ".join(content)
            # 标点符号去除和去除停用词
            content = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）《×]+", "", content)
            new_content = " ".join([i for i in jieba.cut(content) if i not in stop_words]) + "\n"
            yield new_content


if __name__ == '__main__':
    with open('data/simple.reg.txt', 'w') as f:
        for line in read_data():
            f.write(line)
