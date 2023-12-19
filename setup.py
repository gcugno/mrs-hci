from setuptools import setup

setup(
    name='mrshci',
    version='0.0.1',
    packages=['mrshci'],
    url='https://github.com/gcugno/mrs-hci',
    license='MIT',
    author='Gabriele Cugno',
    author_email='gcugno@umich.edu',
    description='Toolkit for analysis of jwst exoplanet data',
    install_requires=["jwst >= 1.12.5",
                      "astropy",
                      "spectres",
                      "matplotlib"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
