class Calculate_auc(object):
	"""docstring for Calss"""
	def __init__(self, labels, values):
		"""构造方法
			传入参数
			---
				labels: list类型
					预测输出标签list
				values: list类型
					对应于标签的概率
		"""
		self.labels = labels
		self.values = values


	def optimizedMethod(self):
		"""优化后的AUC计算方法
			返回
			---
			AUC： float
		"""
		label_list = list(map(lambda s:s.label, self.__sorted_byUsingValue()))[::-1]

		n = len(label_list) # 样本总数
		M = label_list.count('1') # 正样本总数
		N = n - M # 负样本总数

		sums = 0
		for ind, lab in enumerate(label_list, start=1):
			if lab == '1':
				sums += ind

		return (sums - M*(M+1)/2) / (M*N)



	def __sorted_byUsingValue(self):
		"""通过输出概率值对标签数据排序
			
			返回一个 结构体数组
		"""
		structs = list(map(lambda l1, v1: Node(l1, v1), self.labels, self.values))
		random.shuffle(structs) # 随机打乱顺序

		res = []
		helparr = []
		for each in structs:
			if len(res) == 0:
				res.append(each)
				continue

			while len(res) != 0 and res[-1].value < each.value:
				helparr.append(res.pop(-1))

			res.append(each)

			while len(helparr) != 0:
				res.append(helparr.pop(-1))

		return res # 从栈顶到栈底 小到大



class Node:
	"""定义一个结构体
			
		用于存放 label(str) 和 value(float)
	"""
	def __init__(self, label, value):
		self.label = label
		self.value = value



import random

outputFileName = "T_AUC.data"
g = open(outputFileName, 'w')

for i in range(1, 1423):

	filename = r"C:\Users\liany\Desktop\calculate AUC\label_%d.txt"%(i)
	labels, values = [], []

	f = open(filename)
	for each in f:
		each = each.strip().split()
		labels.append(each[0])
		values.append(float(each[1]))
	f.close()

	# 计算AUC
	auc_caler = Calculate_auc(labels, values)
	auc = auc_caler.optimizedMethod()

	# 写入文件
	g.write("%d\t%.6f\n"%(i, auc))

g.close()
