import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='hyponic',
    version='0.0.1-alpha-0',
    author='Vladislav Kulikov, Daniel Satarov, Ivan Chernakov',
    author_email='v.kulikov@innopolis.university, d.satarov@innopolis.university, i.chernakov@innopolis.university',
    description='Hyperparameter Optimization with Nature-Inspired Computing',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/slewie/HypONIC',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.19.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
