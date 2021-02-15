import os
import sys
import numpy as np
import pandas as pd
from sympy import Rational, fraction, nroots
from sympy.abc import t


def pade(A):
	X = np.subtract.outer(A[1:, 0], A[:, 0]).T
	column_size = np.size(A, 0)
	M = (A[0, 1] / (A[1:, 1])) - 1

	Z = np.zeros((column_size, column_size - 1))
	Z[0, 0] = M[0] / X[0, 0]
	Z[0, 1:] = (Z[0, 0] * X[0, 1:] / M[1:]) - 1

	for i in range(1, column_size - 1):
		Z[i, i] = Z[i - 1, i] / X[i, i]
		Z[i, i + 1:] = (Z[i, i] * X[i, i + 1:] / Z[i - 1, i + 1:]) - 1

	diag_z = Z.diagonal()

	def build_frac(a, z, size):
		plus_one = Rational(str(z[size - 2])) * (t - Rational(str(a[size - 2, 0]))) + Rational(1)
		frac = None
		for i in reversed(range(size - 1)):
			if i == 0:
				frac = Rational(str(a[0, 1])) / plus_one
			else:
				frac = (Rational(str(z[i - 1])) * (t - Rational(str(a[i - 1, 0])))) / plus_one
			frac = frac.ratsimp()
			plus_one = Rational(1) + frac
		return frac

	eqa = build_frac(A, diag_z, column_size)

	diag_z = list(diag_z)
	diag_z[-1] = 0
	eqa_minus1 = build_frac(A, diag_z, column_size)

	eqa_der = eqa.diff(t)
	eqa_der = eqa_der.ratsimp()

	numerator, denominator = fraction(eqa_der)
	try:
		sol = nroots(numerator, n=15, maxsteps=100)
	except:
		try:
			sol = nroots(numerator, n=15, maxsteps=150)
		except:
			return None

	sol = np.array(sol, dtype=complex)
	num = len(sol)

	eqa_solve = np.empty(num, dtype=complex)
	eqa_der_solve = np.empty(num, dtype=complex)
	eqa_minus1_solve = np.empty(num, dtype=complex)

	for i in range(num):
		eqa_solve[i] = eqa.evalf(subs={t: sol[i]})
		eqa_der_solve[i] = eqa_der.evalf(subs={t: sol[i]})
		eqa_minus1_solve[i] = eqa_minus1.evalf(subs={t: sol[i]})

	df = pd.DataFrame()
	df['alpha'] = np.abs(sol)
	df['theta'] = np.angle(sol)

	df['real'] = eqa_solve.real
	df['imag'] = eqa_solve.imag

	df['real_der'] = eqa_der_solve.real
	df['imag_der'] = eqa_der_solve.imag
	df['abs_der'] = np.abs(eqa_der_solve)

	pade_error = eqa_solve - eqa_minus1_solve
	df['real_err'] = pade_error.real
	df['imag_err'] = pade_error.imag

	df = df.loc[df['theta'] > 0].reset_index()

	return df


def save_results(data, results_df, file_name):
	"""
	saves the results into file output.dat
	"""
	with open(file_name, 'w') as file:
		file.write(f'#Number of input points = {np.size(data, 0)}\n')
		file.write(f'#{"alpha":>10s} {"Energy":>20s}\n')
		for row in data:
			file.write(f'#{row[0]:.15f} {row[1]:.15f}\n')
		file.write('#\n#\n#\n#*******************#\n#\n#\n')
		file.write(
			f'#{"Real":>12s} {"Imag":>20s} {"Alpha":>20s} {"Theta":>20s} {"Real(der)":>21s} {"Imag(der)":>14s} {"Abs(der)":>14s} {"Real(err)":>14s} {"Imag(err)":>14s} {"Number":>8s}\n')
		for j in range(len(results_df.index)):
			file.write(
				f'{results_df.at[j, "real"]:20.15f} {results_df.at[j, "imag"]:20.15f} {results_df.at[j, "alpha"]:20.15f} {results_df.at[j, "theta"]:20.15f} {results_df.at[j, "real_der"]:14.3e} {results_df.at[j, "imag_der"]:14.3e} {results_df.at[j, "abs_der"]:14.3e} {results_df.at[j, "real_err"]:14.3e} {results_df.at[j, "imag_err"]:14.3e} {j + 1:5.0f}\n')


def run_pade(input_file='input.dat', output_file='output.dat'):
	if not os.path.exists(output_file):
		print('Input file not found', file=sys.stderr)
		sys.exit()
	if not os.path.exists(input_file):
		print('Output file not found', file=sys.stderr)
		sys.exit()

	data = np.genfromtxt(input_file)
	result_df = pade(data)
	if result_df is not None:
		save_results(data, result_df, output_file)
	else:
		print('Failed to solve equation')
