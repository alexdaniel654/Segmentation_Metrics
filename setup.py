from setuptools import setup, find_packages

# Get requirements from text file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Use README.md as the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="segmentationmetrics",
    version="1.0.0",
    description="Binary segmentation accuracy metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexdaniel654/Segmentation_Metrics",
    license="Apache-2.0",

    python_requires='>=3.8, <4',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: Apache Software License',
    ],
)
