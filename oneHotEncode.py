import collections
import itertools
import array

def build_dict(text_list, min_freq):
	""""根据传入的文本列表，创建一个最小频次为min_freq的字典，并返回字典word -> wordid
	"""
	freq_dict = collections.Counter(itertools.chain(*text_list)) # 词频字典
	freq_dict = sorted(freq_dict.items(), key=lambda asd:asd[1], reverse=True) # 按词频从大到小排序
	words, _ =  zip(*filter(lambda wc: wc[1] >= min_freq, freq_dict)) # 保留词，抛弃频数

	return dict(zip(words, range(len(words))))
    
def text2vector(text_list, word2id):
	X = []
	for text in text_list:
		vect = array.array('l', [0] * len(word2id))
		for word in text:
			if word not in word2id:
				continue
			vect[word2id[word]] = 1

		X.append(vect)
	return X


def main(text_list, min_freq = 5):
	"""one hot 主程序：通过输入的文本向量list，得的样本one hot编码的特征向量
	参数
	===
		text_list: [[],[], ...]
			文本样本list
		min_freq: int
			词的最小词频
	返回
	---
		features: [[], [], ...]
			特征list
	"""
	word2id = build_dict(text_list, min_freq)

	return text2vector(text_list, word2id)
