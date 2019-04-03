import tensorflow as tf
from data_generator import SequenceData

# param config
learning_rate = 0.01
batch_size = 128
steps = 10000
min_step = 10

seq_max_len = 20
n_hidden = 64
n_classes = 2

train_set = SequenceData(n_samples=1000, max_seq_len=seq_max_len)
test_set = SequenceData(n_samples=100, max_seq_len=seq_max_len)

x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
seq_len = tf.placeholder(tf.int32, [None])

weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}


def dynmicRNN(x, seq_len, weights, biases):
    # x shape:(batch_size, max_seq_len, n_input)
    # seq_len shape: (batch_size,)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
                                        sequence_length=seq_len)
    # outputs shape:(batch_size, seq_max_len, n_hidden)

    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']


pred = dynmicRNN(x, seq_len, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                              labels=y))
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(1, steps + 1):
        batch_x, batch_y, batch_seq_len = train_set.next(batch_size)
        sess.run(opt, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
        if i % min_step == 0:
            acc, loss = sess.run([accuracy, cost],
                                 feed_dict={x: batch_x, y: batch_y,
                                            seq_len: batch_seq_len})
            print("step " + str(i) + " , min_batch loss= " +
                  "{:.6f}".format(loss) + ", training acc= " +
                  "{:.4f}".format(acc))
    print("optimization finished!")

    test_acc = sess.run(accuracy, feed_dict={x: test_set.data,
                                             y: test_set.labels,
                                             seq_len: test_set.seq_len})
    print("test acc: " + "{:.4f}".format(test_acc))
