from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
	long_description = fh.read()

setup(
	name='automatic-rvp',
	version='1.0.0',
	description='RVP Program',
	long_description=long_description,
	long_description_content_type='text/markdown',
	packages=find_packages(),
	url='https://github.com/haritan/RVP',
	author='Yochai Safrai',
	author_email='yochai.safrai@gmail.com',
	py_modules=['rvp', 'makima', 'pade', 'clustering', 'stabilization'],
	maintainer='Idan Haritan',
	maintainer_email='idan.haritan@gmail.com',
	include_package_data=True,
	package_dir={'': 'src'},
	install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
	classifiers=[
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.7',
		],
	python_requires='>=3.7',
	zip_safe=False,
)
