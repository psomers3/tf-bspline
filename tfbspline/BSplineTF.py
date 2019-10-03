import tensorflow as tf
import numpy as np
from .util import interpolate

# TODO: Make function to directly return BSpline object from given points


class BSpline(object):
    def __init__(self, start_position, end_position, num_internal_knots, degree=3):
        """
        Creates a Tensorflow-based b-spline object using x, y coordinate data.
        :param start_position: start x-value of the bspline.
        :param end_position: end x-value of the bspline.
        :param num_internal_knots: minimum number of knots that will be placed in b-spline range (more will actually be
                                   used in total, including those outside of specified range). Actual knots used may be
                                   obtained from get_knots().
        :param degree: degree of the b-spline
        """

        self.start_position = start_position
        self.end_position = end_position + 0.000001  # this is done because effective range is not inclusive of
                                                     # the end position
        self.num_internal_knots = num_internal_knots
        self.degree = degree
        self.knot_positions = self._get_knot_positions()

        # initialize knot values randomly
        self.knot_values = tf.Variable(tf.random.uniform([1, self.knot_positions.shape[-1]]),
                                       name="knot_y",
                                       trainable=True,
                                       dtype=tf.float32)

        # optimizer specs to use when fitting data points
        self.lr = 0.5  # learning rate
        self.beta_1 = 0.9  # gradient momentum
        self.beta_2 = 0.99
        self.epsilon = None

        self.convergence_criteria = tf.constant(0.00001, dtype=tf.float32)
        self.gradient_momentum = tf.Variable(0, dtype=tf.float32)
        self.current_loss = tf.Variable(0, dtype=tf.float32)
        self.alpha = 1 - self.beta_1

    def _get_knot_positions(self):
        """
        Function to help get knot positions to use with the get_spline function.
        :return: tensorflow tensor of shape=(1, num_knots) with the x_values for b-spline knots
        """
        interval = (self.end_position - self.start_position) / (self.num_internal_knots + 1)
        added_knots = self.degree + 1
        return tf.constant(np.asarray([np.linspace(self.start_position - (interval * 0.5 * added_knots),
                                                   self.end_position + (interval * 0.5 * added_knots),
                                                   self.num_internal_knots + 2 + added_knots)]),
                           dtype=tf.float32)

    @tf.function
    def set_knot_values(self, knot_y_values):
        """
        Manually set the knot y-values.
        :param knot_y_values: new y-values. Must include exactly the same number of values as number of knots
                              given by get_knots().
        :return: None
        """

    def get_knots(self):
        """
        :return: current knot positions and values.
        """
        return tf.concat([self.knot_positions, self.knot_values], axis=0)

    @tf.function
    def __call__(self, positions=None, raster=None):
        """
        Returns the calculated b-spline points. MUST SPECIFY ONLY POSITIONS OR RASTER, NOT BOTH
        :param positions: specific x-values to return values for.
        :param raster: division length to divide full b-spline range by.
        :return: b-spline points with shape=(2, num_points) (i.e. ([x], [y]))
        """

    @tf.function
    def fit_points(self, x, y):
        """
        Fits b-spline to given x, y data by adjusting internal knot y-values
        :param x: x values
        :param y: y values
        :return: None
        """
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)  # numpy values are often float64, but need float32 for fitting
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)
        self.current_loss.assign(0)
        self.gradient_momentum.assign(0)

        # define train function that will be called over and over until convergence
        def train():
            # start tape to keep track of calculations for gradient calculation
            with tf.GradientTape() as t:
                # append knots into one tensor (because that's what get_spline takes)
                knots = tf.concat([self.knot_positions, self.knot_values], axis=0)
                # get values of spline at the x coordinates of the data points
                values = get_spline(knots, x, degree=self.degree)
                # calculate the MSE loss
                current_loss = tf.keras.losses.mse(y, values[:, 1])
                loss_rate = tf.abs(tf.subtract(current_loss, self.current_loss))
                self.current_loss.assign(current_loss)
                self.gradient_momentum.assign(self.gradient_momentum*self.beta_1 + self.alpha*loss_rate)

            # get gradient of loss with respect to knot y-values
            grads = t.gradient(current_loss, [self.knot_values])
            # apply gradient descent step
            self.optimizer.apply_gradients(zip(grads, [self.knot_values]))

        train()
        # train until convergence
        while tf.greater(self.gradient_momentum, self.convergence_criteria):
            train()


@tf.function
def get_spline(knots, positions=None, raster=None, degree=3):
    """
    Returns the b-spline points for the requested positions if raster is None. Else, returns evenly spaced points
    describing the b-spline defined by the interval size raster
    :param knots: Knot values given as shape (2, num_knots) (i.e. ([x_values], [y_values]) ). Knots MUST BE EVENLY
                  SPACED. This function is based on a Cardinal B-Spline function from TensorFlow-graphics.
    :param positions: x-values to get b-spline values for. Valid x-values are in the range:
                      index_offset = (degree-1)/2
                      [knots[0+index_offset], knots[-index_offset]) <-- Note the open right side bracket
    :param raster: length to divide x axis interval into.
    :param degree: degree of b-spline
    :return: points on the b-spline
    """

    # needed additional knots that exist on the ends for calculating spline
    added_knots = degree - 1

    # zero-based knot length
    knot_range = tf.subtract(knots[0][-1], knots[0][0])

    # length of knot range at each end not included in b-spline
    end_percentage = (added_knots/2)/(knots.shape[-1]-1)

    # How much the actual range is offset from the end of the knot range
    offset = tf.math.multiply(knot_range, end_percentage)
    start = tf.math.add(knots[0][0], offset)
    end = tf.math.subtract(knots[0][-1], offset)
    original_range = tf.subtract(end, start)

    # get maximum value for b-spline index value
    max_pos = knots.shape[-1] - degree

    # if raster, divide knot range evenly by raster value. Otherwise, convert positions to knot range
    if raster is not None:
        delta = tf.math.divide(max_pos, tf.divide(original_range, raster))
        positions = tf.range(start=0, limit=max_pos, delta=delta, dtype=tf.float32)
    elif positions is not None:
        positions = tf.cast(tf.convert_to_tensor(positions), dtype=tf.float32)
        # shift to start from 0
        positions = tf.subtract(positions, start)
        # scale to range of applicable knot indices
        positions = tf.math.multiply(tf.math.divide(positions, original_range), max_pos)
    else:
        raise Exception('Need to provide either positions OR raster value.')
    return interpolate(knots, positions, degree, cyclical=False)
