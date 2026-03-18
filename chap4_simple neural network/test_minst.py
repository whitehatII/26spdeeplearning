import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # normalize
    x = x / 255.0
    x_test = x_test / 255.0

    return (x, y), (x_test, y_test)


class Matmul:
    def __init__(self):
        self.mem = {}

    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem = {'x': x, 'W': W}
        return h

    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']

        ####################
        '''计算矩阵乘法的对应的梯度'''
        # dL/dx = dL/dy · W^T
        grad_x = np.matmul(grad_y, W.T)
        # dL/dW = x^T · dL/dy
        grad_W = np.matmul(x.T, grad_y)
        ####################
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        x = self.mem['x']
        # ReLU 的梯度：x>0 时为1，否则为0
        grad_x = grad_y * (x > 0)
        ####################
        return grad_x


class Softmax:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (partition + self.epsilon)

        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)  # (N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp + grad_y * s
        return tmp


class Log:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x + self.epsilon)

        self.mem['x'] = x
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']

        return 1. / (x + 1e-12) * grad_y

x = np.random.normal(size=[5, 6])
label = np.zeros_like(x)
label[0, 1]=1.
label[1, 0]=1
label[2, 3]=1
label[3, 5]=1
label[4, 0]=1

W1 = np.random.normal(size=[6, 5])
W2 = np.random.normal(size=[5, 6])

mul_h1 = Matmul()
mul_h2 = Matmul()
relu = Relu()
softmax = Softmax()
log = Log()

h1 = mul_h1.forward(x, W1) # shape(5, 4)
h1_relu = relu.forward(h1)
h2 = mul_h2.forward(h1_relu, W2)
h2_soft = softmax.forward(h2)
h2_log = log.forward(h2_soft)


h2_log_grad = log.backward(label)
h2_soft_grad = softmax.backward(h2_log_grad)
h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)
h1_relu_grad = relu.backward(h2_grad)
h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)

print(h2_log_grad)
print('--'*20)
print(W2_grad)

with tf.GradientTape() as tape:
    x, W1, W2, label = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
    tape.watch(W1)
    tape.watch(W2)
    h1 = tf.matmul(x, W1)
    h1_relu = tf.nn.relu(h1)
    h2 = tf.matmul(h1_relu, W2)
    prob = tf.nn.softmax(h2)
    log_prob = tf.math.log(prob)
    loss = tf.reduce_sum(label * log_prob)
    grads = tape.gradient(loss, [prob])
    print (grads[0].numpy())


class myModel:
    def __init__(self):
        ####################
        '''声明模型对应的参数'''
        self.W1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.W2 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([10]))
        ####################

    def __call__(self, x):
        ####################
        '''实现模型函数体，返回未归一化的logits'''
        x = tf.reshape(x, [-1, 784])

        h1 = tf.matmul(x, self.W1) + self.b1
        h1 = tf.nn.relu(h1)

        logits = tf.matmul(h1, self.W2) + self.b2
        ####################
        return logits


model = myModel()

optimizer = optimizers.Adam()

@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # compute gradient
    trainable_vars = [model.W1, model.W2, model.b1, model.b2]
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy

@tf.function
def test(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


train_data, test_data = mnist_dataset()
batch_size = 128
train_x, train_y = train_data

for epoch in range(50):
    idx = np.random.permutation(len(train_x))
    train_x = train_x[idx]
    train_y = train_y[idx]
    for i in range(0, len(train_x), batch_size):
        x_batch = train_x[i:i+batch_size]
        y_batch = train_y[i:i+batch_size]
        loss, accuracy = train_one_step(
            model, optimizer,
            tf.constant(x_batch, dtype=tf.float32),
            tf.constant(y_batch, dtype=tf.int64)
        )
    print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
loss, accuracy = test(model,
                      tf.constant(test_data[0], dtype=tf.float32),
                      tf.constant(test_data[1], dtype=tf.int64))

print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())