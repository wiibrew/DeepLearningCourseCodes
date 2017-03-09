#get the mnist data 
# wget http://deeplearning.net/data/mnist/mnist.pkl.gz




from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#pre-define the  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Create model
def multilayer_perceptron(x, weights, biases):
    #now, we want to change this to a CNN network

    #first reshape the data to 4-D

    x_image = tf.reshape(x, [-1,28,28,1])

    #then apply cnn layers

    h_conv1 = tf.nn.relu(conv2d(x_image, weights['conv1']) + biases['conv_b1'])
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['conv2']) + biases['conv_b2'])
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['fc1']) + biases['fc1_b'])


    # Output layer with linear activation
    out_layer = tf.matmul(h_fc1, weights['out']) + biases['out_b']
    return out_layer

# Store layers weight & biases
weights = {
    'conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'fc1' : tf.Variable(tf.random_normal([7*7*64,256])),
    'out': tf.Variable(tf.random_normal([256,n_classes]))
}
biases = {
    'conv_b1': tf.Variable(tf.random_normal([32])),
    'conv_b2': tf.Variable(tf.random_normal([64])),
    'fc1_b': tf.Variable(tf.random_normal([256])),
    'out_b': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))