import sys
import array
import math
import operator


class SimpleLogisticRegression(object):
	"""docstring for SimpleLogisticRegression"""

	def __init__(self, alpha, feature_num):
		"""构造函数

			传入参数
			---
				alpha: double
					学习率
				feature_num: int
					样本特征数量
		"""
		self.__alpha = alpha
		self.__feature_num = feature_num
		self.__coef = [0.] * self.__feature_num  # n维参数向量
		self.__intercept = 0.  # bias


	def train(self, X, y, verbose = False):
		"""训练数据，生成模型 - 最终参数列表

			传入参数
			---
				tX: [[],[],[],……]
					训练样本特征向量 组
				tY: []
					训练样本类别list

			返回
			---
				模型最终似然函数的值
		"""
		last_target = sys.maxsize
		last_step = 0

		step = 0
		while True:
			step += 1
			# 计算梯度
			gradient = [0.] * (self.__feature_num + 1)
			for tx, ty in zip(X, y):
				# 单个样本的特征向量和标签
				delta = -ty + self.__sigmoid(tx)
				for i, xi in enumerate(tx):
					gradient[i] += delta * xi
				gradient[-1] += delta

			
			gradient = list(map(lambda g: g / len(X), gradient))		

			self.__coef = list(map(lambda c, g: c - self.__alpha * g, self.__coef, gradient[:-1]))
			self.__intercept  -= self.__alpha * gradient[-1]

			target = self.__target(X, y)

			if last_target - target  > 1e-8:
				last_target = target
				last_step = step
			elif step - last_target >= 10:
				break

			if verbose is True and step % 1000 == 0:
				sys.stderr.write("step %d: %.6f\n"%(step, target))
		
		if verbose is True:
			sys.stderr.write("Final value of Cost function is: %.6f\n"%(target))
		return target


	def predict(self, pX):
		"""根据LRModel，预测传入样本的类型 以及 给出其概率值

			传入参数
			---
				pX: [[],[],[],……]
					测试样本特征向量组（多个样本）

			返回
			---
				概率列表 y=1
		"""
		if not self.__check_validity_features(pX):
			sys.stderr.write("The dimension of the data can't match the training data")
			return None
		return list(map(self.__sigmoid, pX))


	def __check_validity_features(self, X):
		"""检查传入样本特征 的 合法性

			传入参数
			---
				X： [[],[], ……] 
					传入特征向量组

			返回值
			---
				True - 合法
				False - 不合法
		"""
		for x in X:
			if not isinstance(x, (list, tuple, array.array)):
				return False
			if len(x) != self.__feature_num:
				return False
		return True

	def __target(self, tX, tY):
		hF = list(map(self.__sigmoid, tX))
		return -sum(map(lambda ey, h: ey * math.log(h) + (1-ey) * math.log(1-h), tY, hF)) / len(tX)

	def __sigmoid(self, X):
		"""sigmoid函数

			传入参数
			---
				X： [[],[],……]
					一个样本的特征向量

			返回
			---
				sigmoid函数值
		"""
		return 1. / (1 + math.exp(-sum(map(operator.mul, self.__coef, X)) - self.__intercept))

if __name__ == '__main__':
	lr = SimpleLogisticRegression(0.1, 3)
	X = [[1, 3, 5], [2, 4, 6], [3, 5, 7], [4,  6, 8]]
	y = [0, 0, 1, 1]
	print(lr.predict(X))
	lr.train(X, y, verbose=True)
	print(lr.predict(X))

		




		