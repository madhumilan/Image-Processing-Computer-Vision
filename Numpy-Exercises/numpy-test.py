import numpy as np
import time

arr = np.array([1,2,3])
print(arr)
print(arr.shape)

arr = np.array([[1,2,3],[4,5,6]])
# print(arr)
# print("Type",type(arr))
# print("Size",arr.size)
# print("Shape",arr.shape)

# arr = ["Hello", 1, True]
# print(arr)
# arr = np.array(["Hello", 1, True])
# print(arr)
# print("address of arr =", arr.data)

# arr_copy = arr									# Not a copy just a reference is created
# print("Address of copy =", arr_copy.data)
# arr_copy[1] = 2
# print(arr)

arr = np.append(arr, 99)
print(arr)
print(arr.shape)

arr = np.delete(arr, 6)
print(arr)

zeros = np.zeros((2,3))
print(zeros)
ones = np.ones((3,4))
print(ones, ones.dtype)

x = np.arange(10)
print(x)
x = np.arange(21,30,2)
print(x)
x = np.linspace(0,7,3)
print(x)

x = np.full((2,2), 7)
print(x)

x = np.eye(5)
print(x)

# Math Operations
x = np.array([[1,2],
			[3,4]])
y = np.array([[5,6],[7,8]])
t1 = time.time()
print(x+y)
t2 = time.time()
print(t2-t1)

t1 = time.time()
print(np.add(x,y))
t2 = time.time()
print(t2 - t1)

# Dot Product
v = np.array([7,8])
w = np.array([9,10])
print(v.dot(w))
print(np.dot(v,w))

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))


