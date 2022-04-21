import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# initialization of tensors
x = tf.constant(4.0)
y = tf.constant(4, shape=(1,1), dtype= tf.float32)
z = tf.constant([[1,2,3],[4,5,6]])
x = tf.ones((2,3))
y = tf.zeros((3,4))
z = tf.eye(3)  #Identity matrix
x = tf.random.normal((3,3), mean=0, stddev=1)
x = tf.random.uniform((1,3), minval=0, maxval= 4)
x = tf.range(9)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float16)  #changing dtype
#tf.float (16,32,64), tf.int(8,16,32,64), tf.bool
#print(x)

# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = tf.add(x,y)
z = x + y

z = x - y           # z = tf.subtract(x,y)

z = x / y           # z = tf.divide(x,y)

z = x * y           # z = tf.multiply(x,y)

z = tf.tensordot(x, y, axes=1)          #dot product
z = tf.reduce_sum(x * y, axis=0)

z = x ** 5                  #elementwise exponent by 5
#print(z)

#matrix multiplication
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = x @ y                         # z = tf.matmul(x, y)
#print(z)

# Indexing
#same as python

x = tf.constant([1, 2, 4, 8, 0, 5, 6])
#print(x[:])
#print(x[2:5])

indices = tf.constant([0,3])
x_ind = tf.gather(x, indices)
#print(x_ind)

x = tf.constant([[1, 2],
                [3, 4],
                [5, 6]])
#print(x[0,:])
#print(x[0:2, :])                      #x[row, column]

# Reshaping
x = tf.range(9)
x = tf.reshape(x, (3,3))
print(x)

x = tf.transpose(x, perm=[1,0])
print(x)
