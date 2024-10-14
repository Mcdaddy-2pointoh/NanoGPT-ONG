from setuptools import setup, find_packages

setup(
    name='nano_gpt',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description="An attempt to follow Andrej Karapathy's [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY). In order to better understand 'Attention is all you need' and the transformers architecture that is the corner stone of GenAI.",
    author='Sharvil Dandekar',
    author_email='sharvil.dandekar@gmail.com',
    url='https://your-url.com',
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
        "torch==2.4.1",
        "typing_extensions==4.12.2",
        "tzdata==2024.2"
    ],
)