from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1,normalize=False)
reg.coef_


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        '''
        线性回归
        '''
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
    '''
    前向传播
    '''
    return self.linear(x)

input_size = 1
output_size = 1
model = LinearRegression(input_size, output_size)

mse = nn.MSELoss() #loss

'''
优化器
'''
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
loss列表
训练轮数
'''
loss_list = []
iteration_number = 20000

'''
训练
'''
for iteration in range(iteration_number):
    
    '''
    optimizer.zero_grad()对应d_weights = [0] * n

	即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有	sampleloss关于weight的导数的累加和）
	'''
    optimizer.zero_grad()

	'''
	前向传播求出预测的值
	'''
    results = model(x)  #返回为tensor

	'''
	计算loss
	'''
    loss = mse(results, y)

	'''
	反向传播求梯度
	'''
    loss.backward()

	'''
	weights = [weights[k] + alpha * d_weights[k] for k in range(n)]
	更新所有参数
	'''
    optimizer.step()
	'''
	将loss加入loss列表
	'''
    loss_list.append(loss.data)
    
    if(iteration % 1000 == 0 ):
        print('epoch{}, loss {}'.format(iteration, loss.data))



plt.plot(range(iteration_number), loss_list)

'''
m1: [a x b], m2: [c x d]
m1 is [a x b] which is [batch size x in features]

m2 is [c x d] which is [in features x out features]

'''
