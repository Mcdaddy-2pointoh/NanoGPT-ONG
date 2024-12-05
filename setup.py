from setuptools import setup, find_packages

setup(
    name='nano_gpt',
    version='0.2',
    packages=find_packages(),
    include_package_data=True,
    description="Building a Language model at low cost with coherent speech and rational output",
    author='Sharvil Dandekar',
    author_email='sharvil.dandekar@gmail.com',
    url='https://sharvil-dandekar.vercel.app',
    install_requires=[
        "filelock==3.16.1",
        "fsspec==2024.9.0",
        "Jinja2==3.1.4",
        "MarkupSafe==3.0.1",
        "mpmath==1.3.0",
        "networkx==3.4.1",
        "numpy==2.1.2",
        "pandas==2.2.3",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.2",
        "setuptools==75.1.0",
        "six==1.16.0",
        "sympy==1.13.3",
        "typing_extensions==4.12.2",
        "tzdata==2024.2"
    ],
)
