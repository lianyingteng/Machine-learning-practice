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
		self.__labels = labels
		self.__values = values


	def optimizedMethod(self):
		"""优化后的AUC计算方法

			返回
			---
			AUC： float
		"""
		label_list = map(lambda s:s.label, self.__sorted_byUsingValue())

		n = len(self.__labels)
		M = self.__labels.count('1') # 正样本总数
		N = n - M

		sums = 0
		for i in range(len(label_list)):
			if label_list[i] == '1':
				sum += (n-i)

		return (sums - M*(M+1)/2) / M*N





	def __sorted_byUsingValue(self):
		"""通过输出概率值对标签数据排序
			
			返回一个 结构体数组
		"""
		structs = list(map(Node, self.__labels, self.__values))

		res = []
		helparr = []
		for each in structs:
			if len(res) == 0:
				res.append(each)
				continue

			while len(res) != 0 and res[-1].value < each.value:
				helparr.append(res.pop())

			res.append(each)

			while len(helparr) != 0:
				res.append(helparr.pop())

		return res



	class Node:
		"""定义一个结构体
			
			用于存放 label(str) 和 value(float)
		"""
		def __init__(self, label, value):
			self.label = label
			self.value = value



		