from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='wpencoder',
    version='0.1',
    description='Internal package for encoding occurrences',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    url='https://github.com/wphospital/wpencoder',
    author='WPH DNA',
    author_email='WPHDataAnalytics@wphospital.org',
    license='MIT',
    packages=['wpencoder'],
    install_requires=[
        'pandas',
        'numpy',
        'sklearn'
    ],
    # package_data={'wpconnect': ['oracle_dlls/*.dll', 'queries/*.sql']},
    # include_package_data=True,
    zip_safe=False
)
