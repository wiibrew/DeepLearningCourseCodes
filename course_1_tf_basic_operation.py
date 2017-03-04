'''
basic tf operation examples, 
1. write a tf function use tf.xxxx
2. feed data to tf.placeholder and set data to tf.Variable
3.run...
'''

#
import tensorflow as tf

# direct sum with constand value
a = tf.constant(2)
b = tf.constant(3)
c=a+b
d=a*b

sess=tf.Session()
print sess.run(c)
print sess.run(d)

# 
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# 
add = tf.add(a, b)
mul = tf.multiply(a, b)
print sess.run(add, feed_dict={a: 2, b: 3})
print sess.run(mul, feed_dict={a: 2, b: 3})



#
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix2, matrix1)
print sess.run(product)

#here you should also be able to use tf.placeholder
mat1=tf.Variable(tf.random_normal([3,2]))
mat2=tf.Variable(tf.random_normal([2,3]))
product=tf.matmul(mat1,mat2)

m1=[[1,3],[2,1],[0,5]]
m2=[[3,2,1],[1,2,3]]

print sess.run(product,feed_dict={mat1:m1,mat2:m2})