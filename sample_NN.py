import sklearn
import sklearn.datasets
import numpy as np

# 创建一个数据集
np.random.seed(0)
# X 特征向量； y label向量
X, y = sklearn.datasets.make_moons(200, noise=.2)


num_examples = len(X) # 训练集大小
nn_input_dim = 2 # 输入层维度
nn_output_dim = 2 # 输出层维度

# 梯度下降法参数
epsilon = 0.01 # 梯度下降的学习率
reg_lambda = 0.01 # 正则化参数

def calculate_loss(model):
	"""损失函数定义
	"""
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	
	# 前向传播计算预测输出
	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # softmax层输出

	# 计算 loss
	corect_logprobs = -np.log(probs[range(num_examples)])
	data_loss = np.sum(corect_logprobs) 

	# 将 正则化项 添加到 loss 中
	data_loss += reg_lambda/2 * (np.square(W1))
	return 1./num_examples * data_loss

def predict(model, x):
	"""模型预测

		参数：
			model 模型
			x 输入
	"""
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	# 前向传播
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)

def train(nn_hdim, num_passes=20000, print_loss=False):
	"""学习模型的参数并返回
		
		参数
			nn_hdim: 隐层节点数
			num_passes: 执行梯度下降的最大迭代次数
			print_loss: True 每 1000 次迭代打印一次损失
	"""

	# 随机初始化所有参数值
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
	b1 = np.zeros(1, nn_hdim)
	W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
	b2 = np.zeros(1, nn_output_dim)

	model = dict() # 模型（最后返回）

	# 梯度下降法
	for i in range(0, num_passes):

		# 前向传播
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		# 反向传播
		delta3 = probs
		delta3[range(num_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)

		# 加入正则化项 （偏置没有正则化项）
		dW2 += reg_lambda * W2
		dW1 += reg_lambda * W1

		# 梯度下降参数更新
		W1 += -epsilon * dW1
		b1 += -epsilon * db1
		W2 += -epsilon * dW2
		b2 += -epsilon * db2

		# 为模型设定新值
		model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

		# 打印loss
		if print_loss and i % 1000 == 0:
			print("Loss after iteration %i: %f"%(i, calculate_loss(model)))

	return model