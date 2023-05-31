import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='hyponic',
    version='0.1.1',
    author='Vladislav Kulikov, Daniel Satarov, Ivan Chernakov',
    author_email='v.kulikov@innopolis.university, d.satarov@innopolis.university, i.chernakov@innopolis.university',
    description='Hyperparameter Optimization with Nature-Inspired Computing',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/slewie/HypONIC',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'numexpr>=2.8.4',
        'numba>=0.57.0',
        'matplotlib>=3.6.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
