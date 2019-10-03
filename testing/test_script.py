from tfbspline import BSpline
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Create junk data calculated with y = x^2 + noise
    start = -5
    end = 2
    data_x = np.linspace(start, end, 40)
    noise = np.random.normal(0, 0.05, 40)
    data_y = np.square(np.linspace(-1, 1, 40)) + noise

    # create b-spline
    spline = BSpline(start, end, num_internal_knots=3, degree=3)
    # store original knots to plot later
    original_knots = spline.get_knots()

    # fit spline to data
    spline.fit_points(data_x, data_y)

    # get fitted knots
    fitted_knots = spline.get_knots()

    # plot fitted spline
    spline_points = spline(raster=0.001)
    plt.plot(spline_points[:, 0], spline_points[:, 1], 'g')

    # plot knots
    plt.plot(fitted_knots[0], fitted_knots[1], 'r.')

    # plot original spline
    spline.set_knot_values(original_knots[1])
    spline_points = spline(raster=0.001)
    plt.plot(spline_points[:, 0], spline_points[:, 1], 'y')

    # plot data
    plt.plot(data_x, data_y, 'b.')

    # add legend
    plt.legend(['fitted b-spline', 'knots', 'unfitted b-spline', 'data'], loc='best')
    plt.show()
