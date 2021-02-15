import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class Results:
	def __init__(self, cleaned_data_length, result: pd.DataFrame, original_data_length):
		self.cleaned_data_length = cleaned_data_length
		self.result = result
		self.original_data_length = original_data_length

	def save_results(self, dir_name=''):
		with open(os.path.join(dir_name, 'output.dat'), 'w') as out_file:
			out_file.write(f'Number of points in input: {self.original_data_length}\n')
			out_file.write(f'Number of points after filtering: {self.cleaned_data_length}\n')
			out_file.write('\n')
			out_file.write(self.result.to_string(justify='center', index=False, formatters={
				'real_mean': '{:,.6e}'.format, 'real_std': '{:,.2e}'.format, 'imag_mean': '{:,.6e}'.format,
				'imag_std': '{:,.2e}'.format, 'imag_coeff_of_var': '{:,.2f}'.format, 'alpha_mean': '{:,.2e}'.format,
				'alpha_std': '{:,.2e}'.format, 'theta_mean': '{:,.2e}'.format, 'theta_std': '{:,.2e}'.format,
				'epsilon': '{:,.3f}'.format, 'cluster_size_percentage': '{:,.2f}'.format}))


def clustering(data: pd.DataFrame):
	cleaned_data = data[abs((data['imag_err'] / data['imag'])) < 0.25].drop('imag_err', axis='columns').reset_index(drop=True)
	scaler = StandardScaler()
	scaled_data = pd.DataFrame(scaler.fit_transform(cleaned_data), index=cleaned_data.index, columns=cleaned_data.columns)
	# This is a different way to calculate std.
	# In the scaling above the std is normalized by dividing by n, where n is the number of observations.
	# The scaling below the std is normalized by dividing by n - 1, where n is the number of observations. Matlab Normalize is using this version
	# scaled_data = pd.DataFrame((cleaned_data - cleaned_data.mean()) / cleaned_data.std(ddof=1), index=cleaned_data.index, columns=cleaned_data.columns)

	size = len(cleaned_data)
	min_samples = min(round(size * 0.08), 100)
	result1 = result2 = result3 = pd.DataFrame()
	g1 = g2 = g3 = np.empty(0)

	def calculate_cluster_values(clusters_condition):
		good_clusters = cleaned_data[cleaned_data['label'].isin(labels[clusters_condition])].groupby(
																							'label', as_index=False)

		result_mean = good_clusters.mean().rename(columns={'real': 'real_mean', 'imag': 'imag_mean',
														'alpha': 'alpha_mean', 'theta': 'theta_mean'}, inplace=False)
		result_std = good_clusters.std().rename(columns={'real': 'real_std', 'imag': 'imag_std',
														'alpha': 'alpha_std', 'theta': 'theta_std'}, inplace=False)

		current_result = result_mean.merge(result_std, on='label')

		current_result['epsilon'] = eps
		current_result['size'] = counts[clusters_condition]
		current_result['cluster_size_percentage'] = (current_result['size'] / size) * 100

		current_result['imag_coeff_of_var'] = coeff_of_var[clusters_condition]

		return current_result

	def update_results(clusters_condition):
		nonlocal g1, g2, g3, result1, result2, result3
		first_condition = np.logical_and(clusters_condition, first_clusters_condition)
		second_condition = np.logical_and(clusters_condition, second_clusters_condition)
		third_condition = np.logical_and(clusters_condition, third_clusters_condition)
		if np.any(first_condition):
			result1 = pd.concat([result1, calculate_cluster_values(first_condition)])
			g1 = np.concatenate((g1, indx[first_condition]))
		if np.any(second_condition):
			result2 = pd.concat([result2, calculate_cluster_values(second_condition)])
			g2 = np.concatenate((g2, indx[second_condition]))
		if np.any(third_condition):
			result3 = pd.concat([result3, calculate_cluster_values(third_condition)])
			g3 = np.concatenate((g3, indx[third_condition]))

	for eps in np.arange(0.001, 5, 0.001):
		clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_data[['real', 'imag']])
		cleaned_data['label'] = clustering.labels_
		labels, counts = np.unique(cleaned_data['label'][cleaned_data['label'] >= 0], return_counts=True)

		if len(labels) == 0:
			continue

		data_grouped_by_label = cleaned_data[cleaned_data['label'] >= 0].groupby('label')
		mean = data_grouped_by_label['imag'].mean()
		std = data_grouped_by_label['imag'].std()
		coeff_of_var = np.array(abs(std / mean) * 100)

		first_clusters_condition = np.logical_and(coeff_of_var <= 6.5, counts < size * 0.1)
		second_clusters_condition = np.logical_and(coeff_of_var < 3, counts >= size * 0.1)
		third_clusters_condition = np.logical_and(coeff_of_var >= 3, np.logical_and(coeff_of_var <= 6.5, counts >= size * 0.1))

		if np.any(coeff_of_var <= 6.5):
			bad_old = [ind in list(cleaned_data.loc[np.concatenate((g1, g2)), 'label']) for ind in labels]
			new = [ind not in list(cleaned_data.loc[np.concatenate((g1, g2, g3)), 'label']) for ind in labels]

			core_points = cleaned_data.iloc[clustering.core_sample_indices_]['label']
			indx = np.array([clustering.core_sample_indices_[list(core_points).index(fd)] for fd in labels])

			if np.any(np.logical_and(bad_old, coeff_of_var <= 6.5)):
				g1_delete = [k not in labels[coeff_of_var <= 6.5] for k in cleaned_data.loc[g1, 'label']]
				g2_delete = [k not in labels[coeff_of_var <= 6.5] for k in cleaned_data.loc[g2, 'label']]

				result1 = result1.loc[g1_delete]
				g1 = g1[g1_delete]

				result2 = result2.loc[g2_delete]
				g2 = g2[g2_delete]

				update_results(bad_old)

			if np.any(np.logical_and(new, coeff_of_var <= 6.5)):
				update_results(new)

		# if len(labels) == 1 and (cleaned_data['label'] == 0).all():
		if len(labels) == 1 and len(cleaned_data[cleaned_data['label'] == 0]) > size * 0.95:
			# print('stopped at epsilon:', eps)
			break

	if not result1.empty:
		result1['grade'] = 3

	if not result2.empty:
		result2['grade'] = 2

	if not result3.empty:
		result3['grade'] = 1

	result = pd.concat([result2, result3])

	if result.empty:
		if result1.empty:
			return None
		else:
			result = result1

	# rearranging the results order
	result = result.rename(columns={'label': 'cluster'})
	result['cluster'] = np.arange(1, len(result) + 1)
	result = result[['cluster', 'grade', 'real_mean', 'real_std', 'imag_mean', 'imag_std', 'imag_coeff_of_var',
					'alpha_mean', 'alpha_std', 'theta_mean', 'theta_std','epsilon', 'size', 'cluster_size_percentage']]

	return Results(len(cleaned_data), result, len(data))


def run_clustering(input_file='clustering_input.csv'):
	"""
	A program to automatically calculate complex resonance points based on Pade approximation.

	This program receives data of alpha vs. energy as produced by commercial chemistry programs
	and returns the optimal data needed for calculating complex resonance points using the Pade approximation.
	:param input_file: Input file to be used.
	"""
	if not os.path.exists(input_file):
		print('Input file not found', file=sys.stderr)
		sys.exit()
	data = pd.read_csv(input_file)
	results = clustering(data)
	if results is None:
		print('Failed to find cluster')
	else:
		results.save_results()
