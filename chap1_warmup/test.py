import numpy as np

a = np.array([4,5,6])

print(type(a))
print(a.shape)
print(a[0])

print("=================")

b = np.array([[4,5,6],[1,2,3]])

print(b.shape)
print(b[0,0])
print(b[0,1])
print(b[1,1])

print("=================")

a = np.zeros((3,3), dtype=int)

b = np.ones((4,5))

c = np.eye(4)

d = np.random.random((3,2))

print(a)
print(b)
print(c)
print(d)

print("=================")

a = np.array([
[1,2,3,4],
[5,6,7,8],
[9,10,11,12]
])

print(a)

print(a[2,3])
print(a[0,0])

print("=================")

b = a[0:2,2:4]

print(b)

print(b[0,0])

print("=================")

c = a[1:3,:]

print(c)

print(c[0,-1])

print("=================")

a = np.array([
[1,2],
[3,4],
[5,6]
])

print(a[[0,1,2],[0,1,0]])

print("=================")

a = np.array([
[1,2,3],
[4,5,6],
[7,8,9],
[10,11,12]
])

b = np.array([0,2,0,1])

print(a[np.arange(4), b])

print("=================")

a[np.arange(4), b] += 10

print(a)

print("=================")

x = np.array([1,2])

print(x.dtype)

print("=================")

x = np.array([1.0,2.0])

print(x.dtype)

print("=================")

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x + y)

print("=================")

print(x - y)
print(np.subtract(x,y))

print("=================")

print(x * y)

print(np.multiply(x,y))

print(np.dot(x,y))

print("=================")

print(x / y)

print(np.divide(x,y))

print("=================")

print(np.sqrt(x))

print("=================")

print(x.dot(y))

print(np.dot(x,y))

print("=================")

print(np.sum(x))

print(np.sum(x,axis=0))

print(np.sum(x,axis=1))

print("=================")

print(np.mean(x))

print(np.mean(x,axis=0))

print(np.mean(x,axis=1))

print("=================")

print(x.T)

print("=================")

print(np.exp(x))

print("=================")

print(np.argmax(x))

print(np.argmax(x,axis=0))

print(np.argmax(x,axis=1))

print("=================")

import matplotlib.pyplot as plt

x = np.arange(0,100,0.1)

y = x * x

plt.plot(x,y)

plt.show()

print("=================")

import matplotlib.pyplot as plt

x = np.arange(0,3*np.pi,0.1)

y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1)
plt.plot(x,y2)

plt.show()
