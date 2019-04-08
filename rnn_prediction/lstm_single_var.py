# coding:utf-8
import functools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader
from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model


class _LSTMModel(ts_model.SequentialTimeSeriesModel):
  """A time series model-building example using an RNNCell."""

  def __init__(self, num_units, num_features, exogenous_feature_columns=None,
               dtype=tf.float32):
    """Initialize/configure the model object.

    Note that we do not start graph building here. Rather, this object is a
    configurable factory for TensorFlow graphs which are run by an Estimator.

    Args:
      num_units: The number of units in the model's LSTMCell.
      num_features: The dimensionality of the time series (features per
        timestep).
      exogenous_feature_columns: A list of `tf.feature_column`s representing
          features which are inputs to the model but are not predicted by
          it. These must then be present for training, evaluation, and
          prediction.
      dtype: The floating point data type to use.
    """
    super(_LSTMModel, self).__init__(
        # Pre-register the metrics we'll be outputting (just a mean here).
        train_output_names=["mean"],
        predict_output_names=["mean"],
        num_features=num_features,
        exogenous_feature_columns=exogenous_feature_columns,
        dtype=dtype)
    self._num_units = num_units
    # Filled in by initialize_graph()
    self._lstm_cell = None
    self._lstm_cell_run = None
    self._predict_from_lstm_output = None

  def initialize_graph(self, input_statistics=None):
    """Save templates for components, which can then be used repeatedly.

    This method is called every time a new graph is created. It's safe to start
    adding ops to the current default graph here, but the graph should be
    constructed from scratch.

    Args:
      input_statistics: A math_utils.InputStatistics object.
    """
    super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
    with tf.variable_scope("", use_resource=True):
      # Use ResourceVariables to avoid race conditions.
      self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
      # Create templates so we don't have to worry about variable reuse.
      self._lstm_cell_run = tf.make_template(
          name_="lstm_cell",
          func_=self._lstm_cell,
          create_scope_now_=True)
      # Transforms LSTM output into mean predictions.
      self._predict_from_lstm_output = tf.make_template(
          name_="predict_from_lstm_output",
          func_=functools.partial(tf.layers.dense, units=self.num_features),
          create_scope_now_=True)

  def get_start_state(self):
    """Return initial state for the time series model."""
    return (
        # Keeps track of the time associated with this state for error checking.
        tf.zeros([], dtype=tf.int64),
        # The previous observation or prediction.
        tf.zeros([self.num_features], dtype=self.dtype),
        # The most recently seen exogenous features.
        tf.zeros(self._get_exogenous_embedding_shape(), dtype=self.dtype),
        # The state of the RNNCell (batch dimension removed since this parent
        # class will broadcast).
        [tf.squeeze(state_element, axis=0)
         for state_element
         in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

  def _filtering_step(self, current_times, current_values, state, predictions):
    """Update model state based on observations.

    Note that we don't do much here aside from computing a loss. In this case
    it's easier to update the RNN state in _prediction_step, since that covers
    running the RNN both on observations (from this method) and our own
    predictions. This distinction can be important for probabilistic models,
    where repeatedly predicting without filtering should lead to low-confidence
    predictions.

    Args:
      current_times: A [batch size] integer Tensor.
      current_values: A [batch size, self.num_features] floating point Tensor
        with new observations.
      state: The model's state tuple.
      predictions: The output of the previous `_prediction_step`.
    Returns:
      A tuple of new state and a predictions dictionary updated to include a
      loss (note that we could also return other measures of goodness of fit,
      although only "loss" will be optimized).
    """
    state_from_time, prediction, exogenous, lstm_state = state
    with tf.control_dependencies(
        [tf.assert_equal(current_times, state_from_time)]):
      # Subtract the mean and divide by the variance of the series.  Slightly
      # more efficient if done for a whole window (using the normalize_features
      # argument to SequentialTimeSeriesModel).
      transformed_values = self._scale_data(current_values)
      # Use mean squared error across features for the loss.
      predictions["loss"] = tf.reduce_mean(
          (prediction - transformed_values) ** 2, axis=-1)
      # Keep track of the new observation in model state. It won't be run
      # through the LSTM until the next _imputation_step.
      new_state_tuple = (current_times, transformed_values,
                         exogenous, lstm_state)
    return (new_state_tuple, predictions)

  def _prediction_step(self, current_times, state):
    """Advance the RNN state using a previous observation or prediction."""
    _, previous_observation_or_prediction, exogenous, lstm_state = state
    # Update LSTM state based on the most recent exogenous and endogenous
    # features.
    inputs = tf.concat([previous_observation_or_prediction, exogenous],
                       axis=-1)
    lstm_output, new_lstm_state = self._lstm_cell_run(
        inputs=inputs, state=lstm_state)
    next_prediction = self._predict_from_lstm_output(lstm_output)
    new_state_tuple = (current_times, next_prediction,
                       exogenous, new_lstm_state)
    return new_state_tuple, {"mean": self._scale_back_data(next_prediction)}

  def _imputation_step(self, current_times, state):
    """Advance model state across a gap."""
    # Does not do anything special if we're jumping across a gap. More advanced
    # models, especially probabilistic ones, would want a special case that
    # depends on the gap size.
    return state

  def _exogenous_input_step(
      self, current_times, current_exogenous_regressors, state):
    """Save exogenous regressors in model state for use in _prediction_step."""
    state_from_time, prediction, _, lstm_state = state
    return (state_from_time, prediction,
            current_exogenous_regressors, lstm_state)


def main():
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 50) + np.cos(np.pi * x / 50) +\
        np.sin(np.pi * x / 25) + np.exp(0.001 * x) + noise
    plt.plot(x, y)
    plt.show()

    data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
            }
    reader = NumpyReader(data)

    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader,
                                                               batch_size=5,
                                                               window_size=100)

    estimator = ts_estimators.TimeSeriesRegressor(
        model=_LSTMModel(num_features=1, num_units=128),
        optimizer=tf.train.AdamOptimizer(0.01))
    estimator.train(input_fn=train_input_fn, steps=2000)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions, ) = tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(evaluation,
                                                                     steps=200)))

    observed_times = evaluation['times'][0]
    observed = evaluation['observed'][0, :, :]
    evaluated_times = evaluation['times'][0]
    evaluated = evaluation['mean'][0]
    predicted_times = predictions['times']
    predicted = predictions['mean']

    plt.figure(figsize=(15, 5))
    plt.axvline(999, linestyle="dotted", linewidth=4, color='r')
    observed_line = plt.plot(observed_times, observed,
                             label='observation', color='k')
    evaluated_line = plt.plot(evaluated_times, evaluated,
                              label='evaluation', color='g')
    predicted_line = plt.plot(predicted_times, predicted,
                              label='prediction', color='r')
    plt.legend(handles=[observed_line[0],
                        evaluated_line[0],
                        predicted_line[0]],
               loc='upper left')

    plt.savefig('lstm_single_var.jpg')
    plt.show()


if __name__ == "__main__":
    main()
