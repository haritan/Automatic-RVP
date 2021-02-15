import numpy as np
from scipy.interpolate._cubic import CubicHermiteSpline

__all__ = ["Makima1DInterpolator"]


def prepare_input(x, y, axis, dydx=None):
	"""Prepare input for cubic spline interpolators.

	All data are converted to numpy arrays and checked for correctness.
	Axes equal to `axis` of arrays `y` and `dydx` are rolled to be the 0th
	axis. The value of `axis` is converted to lie in
	[0, number of dimensions of `y`).
	"""

	x, y = map(np.asarray, (x, y))
	if np.issubdtype(x.dtype, np.complexfloating):
		raise ValueError("`x` must contain real values.")
	x = x.astype(float)

	if np.issubdtype(y.dtype, np.complexfloating):
		dtype = complex
	else:
		dtype = float

	if dydx is not None:
		dydx = np.asarray(dydx)
		if y.shape != dydx.shape:
			raise ValueError("The shapes of `y` and `dydx` must be identical.")
		if np.issubdtype(dydx.dtype, np.complexfloating):
			dtype = complex
		dydx = dydx.astype(dtype, copy=False)

	y = y.astype(dtype, copy=False)
	axis = axis % y.ndim
	if x.ndim != 1:
		raise ValueError("`x` must be 1-dimensional.")
	if x.shape[0] < 2:
		raise ValueError("`x` must contain at least 2 elements.")
	if x.shape[0] != y.shape[axis]:
		raise ValueError("The length of `y` along `axis`={0} doesn't match the length of `x`".format(axis))

	if not np.all(np.isfinite(x)):
		raise ValueError("`x` must contain only finite values.")
	if not np.all(np.isfinite(y)):
		raise ValueError("`y` must contain only finite values.")

	if dydx is not None and not np.all(np.isfinite(dydx)):
		raise ValueError("`dydx` must contain only finite values.")

	dx = np.diff(x)
	if np.any(dx <= 0):
		raise ValueError("`x` must be strictly increasing sequence.")

	y = np.rollaxis(y, axis)
	if dydx is not None:
		dydx = np.rollaxis(dydx, axis)

	return x, dx, y, axis, dydx


class Makima1DInterpolator(CubicHermiteSpline):
	def __init__(self, x, y, axis=0):
		# Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
		# https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation
		x, dx, y, axis, _ = prepare_input(x, y, axis)
		# determine slopes between breakpoints
		m = np.empty((x.size + 3,) + y.shape[1:])
		dx = dx[(slice(None),) + (None,) * (y.ndim - 1)]
		m[2:-2] = np.diff(y, axis=0) / dx

		# add two additional points on the left ...
		m[1] = 2. * m[2] - m[3]
		m[0] = 2. * m[1] - m[2]
		# ... and on the right
		m[-2] = 2. * m[-3] - m[-4]
		m[-1] = 2. * m[-2] - m[-3]

		# if m1 == m2 != m3 == m4, the slope at the breakpoint is not defined.
		# This is the fill value:

		# old:
		# t = .5 * (m[3:] + m[:-3])

		t = (m[3:] + m[:-3]) * 0  # this was changed
		# get the denominator of the slope t

		# old:
		# dm = np.abs(np.diff(m, axis=0))

		dm = np.abs(np.diff(m, axis=0)) + np.abs((m[:-1] + m[1:]) / 2)  # this was changed

		f1 = dm[2:]
		f2 = dm[:-2]
		f12 = f1 + f2
		# These are the mask of where the the slope at breakpoint is defined:
		ind = np.nonzero(f12 > 1e-9 * np.max(f12))
		x_ind, y_ind = ind[0], ind[1:]
		# Set the slope at breakpoint
		t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] + f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]

		super(Makima1DInterpolator, self).__init__(x, y, t, axis=0,	extrapolate=False)
		self.axis = axis

	def extend(self, c, x, right=True):
		raise NotImplementedError("Extending a 1-D Makima interpolator is not yet implemented")

	# These are inherited from PPoly, but they do not produce an Makima
	# interpolator. Hence stub them out.
	@classmethod
	def from_spline(cls, tck, extrapolate=None):
		raise NotImplementedError("This method does not make sense for an Makima interpolator.")

	@classmethod
	def from_bernstein_basis(cls, bp, extrapolate=None):
		raise NotImplementedError("This method does not make sense for an Makima interpolator.")
