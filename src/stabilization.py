import os
import sys
from typing import Union
import numpy as np
from makima import Makima1DInterpolator
import matplotlib.pyplot as plt


class StabilizationResult:
	pass

	def get_results(self) -> np.ndarray:
		pass


class StabilizationFail(StabilizationResult):
	def __str__(self):
		return 'Could not find a stable region with the current criteria. Please try another way'


class StabilizationSmallStableRegion(StabilizationFail):
	def __init__(self, min_alpha_region, max_alpha_region, region_size, requested_output_size):
		self.min_alpha_region = min_alpha_region
		self.max_alpha_region = max_alpha_region
		self.region_size = region_size
		self.requested_output_size = requested_output_size

	def __str__(self):
		return f'The stable region found between alpha {self.min_alpha_region} and alpha {self.max_alpha_region}' \
				f' contains only {self.region_size} points and not {self.requested_output_size} as defined'


class StabilizationSuccess(StabilizationResult):
	def __init__(self, results: np.ndarray, original_data: np.ndarray):
		self.results = results
		self.original_data = original_data

	def get_results(self) -> np.ndarray:
		return self.results

	def plot_results(self, output_dir=''):
		plt.plot(self.original_data[:, 0], self.original_data[:, 1], 'ro', markersize=3)
		plt.plot(self.results[:, 0], self.results[:, 1], marker='o', markerfacecolor='none',
					linestyle='none', markeredgecolor='blue', markersize=5)
		# plt.plot(self.results[:, 0], self.results[:, 1], 'b.', markersize=3)
		plt.xlabel('alpha')
		plt.ylabel('energy')
		plt.legend(('original data', 'results'))
		plt.title('Stabilization Stable Zone')
		plt.savefig(os.path.join(output_dir, 'stabilization_plot_results'))
		plt.show()


def expand_by_curve(data, i, der, threshold):
	right = i + 1
	left = i
	if abs(der) < 3 * 1e-3:
		max_curve = 3 * 1e-3
		min_curve = -3 * 1e-3
	elif der < 0:
		max_curve = der * (2 - threshold)
		min_curve = der * threshold
	else:
		min_curve = der * (2 - threshold)
		max_curve = der * threshold

	diff_right = data - data[i - 1]

	energy_diff_right = diff_right[i + 1:, 1]
	alpha_diff_right = diff_right[i + 1:, 0]
	right_curves = energy_diff_right / alpha_diff_right
	first_strike = False
	for cur_curve, j in zip(right_curves, range(i + 2, len(data) + 1)):
		if min_curve <= cur_curve <= max_curve:
			right = j
			first_strike = False
		elif not first_strike:
			first_strike = True
		else:
			break

	diff_left = data - data[i]
	energy_diff_left = diff_left[:i - 1, 1]
	alpha_diff_left = diff_left[:i - 1, 0]
	left_curves = energy_diff_left / alpha_diff_left
	first_strike = False
	for cur_curve, j in zip(reversed(left_curves), reversed(range(i - 1))):
		if min_curve <= cur_curve <= max_curve:
			left = j
			first_strike = False
		elif not first_strike:
			first_strike = True
		else:
			break

	return left, right


def calculate_optimal_range(data, threshold, max_derivative, min_results):
	energy_diff = np.diff(data[:, 1])
	alpha_diff = np.diff(data[:, 0])

	np.seterr(divide='ignore')  # suppress division by 0 warning
	derivative = energy_diff / alpha_diff
	# Add infinity at the beginning and at the end as the derivative can't be calculated there.
	derivative = np.concatenate([[np.inf], derivative, [np.inf]])

	sorted_derivative = sorted(derivative, key=abs)
	sorted_indexes = np.argsort(abs(derivative))

	for i, der in zip(sorted_indexes, sorted_derivative):
		if der >= max_derivative:
			return None
		left, right = expand_by_curve(data, i, der, threshold)
		if (right - left) >= min_results:
			return data[left:right]

	return None


def stabilization(
		data,
		threshold,
		maximum_derivative,
		output_size,
		interpolation_percentage,
		minimum_stable_zone_points,
		smooth_only,
) -> Union[StabilizationFail, StabilizationSuccess]:
	sorted_data = data[data[:, 0].argsort()]

	def interpolate_data(dat, size) -> np.ndarray:
		alpha = np.linspace(dat[0, 0], dat[-1, 0], size)
		# energy = interp1d(sorted_data[:, 0], sorted_data[:, 1], kind='cubic', assume_sorted=True)(alpha)
		energy = Makima1DInterpolator(sorted_data[:, 0], sorted_data[:, 1])(alpha)
		return np.column_stack((alpha, energy))

	if smooth_only:
		return StabilizationSuccess(interpolate_data(sorted_data, output_size), data)

	interpolated_data = interpolate_data(sorted_data, round(len(sorted_data) * interpolation_percentage))
	optimal_range = calculate_optimal_range(interpolated_data, threshold, maximum_derivative,
											minimum_stable_zone_points)
	if optimal_range is None:
		return StabilizationFail()

	stable_region_size = len(
		sorted_data[np.logical_and(optimal_range[0, 0] <= sorted_data[:, 0],
									sorted_data[:, 0] <= optimal_range[-1, 0]), :])
	if stable_region_size < output_size:
		# verifies that the original data in the optimal range contains more points then the minimum results.

		def find_nearest(array, value):
			""" Find nearest value is an array """
			idx = (np.abs(array - value)).argmin()
			return array[idx]

		return StabilizationSmallStableRegion(find_nearest(sorted_data[:, 0], optimal_range[0, 0]),
												find_nearest(sorted_data[:, 0], optimal_range[-1, 0]), stable_region_size,
												output_size)

	return StabilizationSuccess(interpolate_data(optimal_range, output_size), data)


def verify_data(
		data,
		threshold,
		output_size,
		minimum_stable_zone_points,
		maximum_derivative,
		interpolation_percentage,
):
	if threshold < 1.0:
		return 'Illegal threshold. Value must be higher than 1.0'
	if output_size < 15:
		return 'Illegal number for acceptable results. Must be 15 or higher'
	if minimum_stable_zone_points <= 0:
		return 'Illegal size for stable zone. Must be a positive integer'
	if maximum_derivative <= 0.0:
		return 'Illegal maximum derivative. Must be a positive number'
	if interpolation_percentage <= 0.0:
		return 'Illegal interpolation percentage. Must be a positive number'
	if np.shape(data)[1] != 2:
		return 'Input contains wrong number of columns. Must be 2'
	if len(data) < output_size:
		return f'Input contains less then {output_size} points'
	u, c = np.unique(data[:, 0], return_counts=True)
	dup = u[c > 1]
	if len(dup) > 0:
		return 'Input contains multiple energies for the same alpha'

	return None


def run_stabilization(
		input_file: str,
		output_file='stabilization_output.dat',
		threshold=1.3,
		maximum_derivative=1.0,
		output_size=25,
		interpolation_percentage=0.4,
		minimum_stable_zone_points=10,
		smooth_only=False,
		plot=True,
):
	"""

	:param input_file: Path to input file.
	:param output_file: The name of the output file. Default is stabilization_output.dat.
	:param threshold: The threshold to determine if a point will be in the stable zone or not. Default is 1.3.
	:param maximum_derivative: The absolute value of the maximum derivative allowed when calculating the stable zone.
		If this derivative is reached, then a stable zone can not be found. Default is 1.0.
	:param output_size: The number of points in the output. Default is 25.
	:param interpolation_percentage: The percentage of data used when calculating the stable zone. Default is 0.4.
	:param minimum_stable_zone_points: The minimum number of points in the stable zone. Default is 10.
	:param smooth_only: If this option is set to True, then the data will be reduced to the given output size by means
		of interpolation without trying to find a stable zone. Default is False.
	:param plot: Indicator if to plot the stabilization results. Default is True.
	"""
	if not os.path.exists(input_file):
		print('Input file not found', file=sys.stderr)
		exit(0)
	data = np.genfromtxt(input_file)
	error = verify_data(data, threshold, output_size, minimum_stable_zone_points,
						maximum_derivative, interpolation_percentage)

	if error is not None:
		print(error, file=sys.stderr)
		sys.exit()

	result = stabilization(data, threshold, maximum_derivative, output_size, interpolation_percentage,
							minimum_stable_zone_points, smooth_only)

	if isinstance(result, StabilizationFail):
		print(result, file=sys.stderr)
	else:
		np.savetxt(output_file, result.get_results(), fmt='%.15f')
		if plot:
			result.plot_results()
