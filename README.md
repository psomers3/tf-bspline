# tf-bspline
A TensorFlow implementation of a b-spline interpolator. This is done so that TensorFlow can
follow the operations (i.e. for use in calucating derivatives). This utilizes the 
cardinal b-spline interpolation from tensorflow-graphics, however, only select files
are taken from that package because it takes a long time to import. Be aware that updates to 
tensorflow-graphics must be manually incorporated in tf-bspline.

This package will not exceed the speed of scipy's interpolation operations and is in fact
slower.
## BSpline()
```python
BSpline(self, start_position, end_position, num_internal_knots, degree)
```

Creates a Tensorflow-based b-spline object using x, y coordinate data.

```start_position```-> start x-value of the bspline.

```end_position```-> end x-value of the bspline.

```num_internal_knots```-> minimum number of knots that will be placed in b-spline range (more will actually be
                           used in total, including those outside of specified range). Actual knots used may be
                           obtained from get_knots().
                           
```degree```-> degree of the b-spline

### set_knot_values
```python
BSpline.set_knot_values(self, knot_y_values)
```

Manually set the knot y-values.

``` knot_y_values```-> new y-values. Must include exactly the same number of values as number of knots
                       given by get_knots().

### get_knots
```python
BSpline.get_knots(self)
```

```Returns```-> current knot positions and values.

### fit_points
```python
BSpline.fit_points(self, x, y)
```

Fits b-spline to given x, y data by adjusting internal knot y-values

```x```-> x values

```y```-> y values

## get_spline
```python
get_spline(knots, positions=None, raster=None, degree=3)
```

Returns the b-spline points for the requested positions if raster is None. Else, returns evenly spaced points
describing the b-spline defined by the interval size raster

```knots```-> Knot values given as shape (2, num_knots) (i.e. ([x_values], [y_values]) ). Knots MUST BE EVENLY
              SPACED. This function is based on a Cardinal B-Spline function from TensorFlow-graphics.

```positions```-> x-values to get b-spline values for. Valid x-values are in the range:
                  index_offset = (degree-1)/2
                  [knots[0+index_offset], knots[-index_offset]) <-- Note the open right side bracket

```raster```-> length to divide x axis interval into.

```degree```-> degree of b-spline

```Returns```-> points on the b-spline

# Example
Here is an example test script with output.
```python
from tfbspline import BSpline
import numpy as np
import matplotlib.pyplot as plt

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
```
<p align="center">
  <img src="misc/example.png">
</p>