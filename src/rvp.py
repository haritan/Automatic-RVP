import sys
import concurrent.futures
import os
import numpy as np
import pandas as pd
from itertools import repeat
from stabilization import stabilization, verify_data, StabilizationFail, StabilizationSuccess
from pade import pade, save_results
from clustering import clustering


def run_pade(data_in, start, finish, dir_name):
	curr_result = pade(data_in)
	if curr_result is None:
		return None

	save_results(data_in, curr_result, os.path.join(dir_name, f'pade_output_from_{start}_to_{finish}.dat'))

	return curr_result.loc[curr_result['imag'] < 0, ['real', 'imag', 'alpha', 'theta', 'imag_err']]


def auto_rvp(
		input_file,
		threshold=1.3,
		minimum_stable_zone_points=10,
		interpolation_percentage=0.4,
		stabilization_output_size=25,
		maximum_derivative=1.0,
		skip_stabilization=False,
		stabilization_smooth_only=False,
		min_pade_input_size=8,
		max_pade_input_size=35,
		plot=True,
	):
	"""

	:param input_file: Path to input file.
	:param threshold: The threshold to determine if a point will be in the stable zone or not. Default is 1.3.
	:param minimum_stable_zone_points: The minimum number of points in the stable zone. Default is 10.
	:param interpolation_percentage: The percentage of data used when calculating the stable zone. Default is 0.4.
	:param stabilization_output_size: The number of points in the output. Default is 25.
	:param maximum_derivative: The absolute value of the maximum derivative allowed when calculating the stable zone.
		If this derivative is reached, then a stable zone can not be found. Default is 1.0.
	:param skip_stabilization: If this option is set to True, then the stabilization phase will be skipped. This option is mutually exclusive
		with the stabilization_smooth_only option, and an error will occur if both are True. Default is False.
	:param stabilization_smooth_only: If this option is set to True, then the data will be reduced to the given output size by means
		of interpolation without trying to find a stable zone. Default is False.
	:param min_pade_input_size: The minimum number of points passed in one iteration to pade. Default is 8.
	:param max_pade_input_size: The maximum number of points passed in one iteration to pade. Default is 35.
	:param plot: Indicator if to plot the stabilization results. Default is True.
	"""

	if not os.path.exists(input_file):
		print('Input file not found', file=sys.stderr)
		sys.exit()
	if min_pade_input_size < 8:
		print('Illegal number for pade minimum input size. Must be 8 or higher', file=sys.stderr)
		sys.exit()
	if stabilization_output_size < min_pade_input_size:
		print('Stable part output size can not be lower then minimum input size for pade', file=sys.stderr)
		sys.exit()
	if max_pade_input_size < min_pade_input_size:
		print('Maximum input size for pade can not be lower then minimum input size for pade', file=sys.stderr)
		sys.exit()
	if skip_stabilization and stabilization_smooth_only:
		print('The arguments skip_stabilization and stabilization_smooth_only can not be both True', file=sys.stderr)
		sys.exit()

	data = np.genfromtxt(input_file)
	error = verify_data(data, threshold, stabilization_output_size, minimum_stable_zone_points,
						maximum_derivative, interpolation_percentage)

	if error is not None:
		print(error, file=sys.stderr)
		sys.exit()

	stabilization_result = stabilization(data, threshold, stabilization_output_size, stabilization_output_size, interpolation_percentage,
									minimum_stable_zone_points, stabilization_smooth_only) if not skip_stabilization \
									else StabilizationSuccess(data, data)

	if isinstance(stabilization_result, StabilizationFail):
		print(stabilization_result, file=sys.stderr)
		sys.exit()

	stable_zone = stabilization_result.get_results()

	dir_name = 'results' + '_' + input_file.split('.')[0]
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)

	# save results to output dir
	np.savetxt(os.path.join(dir_name, 'stabilization_output.dat'), stable_zone, fmt='%.15f')

	# copy original data to output dir
	np.savetxt(os.path.join(dir_name, input_file), data, fmt='%.15f')

	sliced_data = []
	left = []
	right = []
	for i in range(len(stable_zone - min_pade_input_size) + 1):
		for j in range(min_pade_input_size, max_pade_input_size + 1):
			if i + j > len(stable_zone):
				continue
			left.append(i + 1)
			right.append(i + j)
			sliced_data.append(stable_zone[i:i + j])

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results = executor.map(run_pade, sliced_data, left, right, repeat(dir_name))

		result_df = pd.DataFrame()
		for r in results:
			if r is not None:
				result_df = pd.concat([result_df, r])

	result_df.to_csv(os.path.join(dir_name, 'clustering_input.csv'), index=False)

	clustering_results = clustering(result_df)
	if clustering_results is None:
		print('Failed to find a cluster')
	else:
		clustering_results.save_results(dir_name)
		if not skip_stabilization and plot:
			stabilization_result.plot_results(dir_name)
