# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise

# csv_file = open('time_series.csv', 'a', newline='')
# csv_write = csv.writer(csv_file, dialect='excel')
# for x_, y_ in zip(x, y):
#     csv_write.writerow((x_, y_))

# plt.plot(x, y)
# plt.show()
# plt.savefig('time_series.jpg')

data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
        }
reader = NumpyReader(data)
# csv_file_name = 'time_series.csv'
# reader_csv = tf.contrib.timeseries.CSVReader(csv_file_name)

train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader,
                                                           batch_size=16,
                                                           window_size=40)
# with tf.Session() as sess:
#     # full_data = reader.read_full()
#     batch_data = train_input_fn.create_batch()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     # print(sess.run(full_data))
#     one_batch = sess.run(batch_data[0])
#     coord.request_stop()
#     print("one batch: ", one_batch)

ar = tf.contrib.timeseries.ARRegressor(periodicities=200,
                                       input_window_size=30,
                                       output_window_size=10,
                                       num_features=1,
                                       loss=tf.contrib.timeseries.ARModel.
                                       NORMAL_LIKELIHOOD_LOSS)
ar.train(input_fn=train_input_fn, steps=6000)
evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
# keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple',
# 'times', 'global_step']
evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
(prediction,) = tuple(ar.predict(
    input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
        evaluation, steps=250)))

plt.figure(figsize=(15, 5))
plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1),
         label='origin')
plt.plot(prediction['times'].reshape(-1), prediction['mean'].reshape(-1),
         label='prediction')
plt.xlabel('time_step')
plt.ylabel('values')
plt.legend(loc=4)
plt.savefig('predict_result.jpg')
plt.show()
