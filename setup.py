from setuptools import setup, find_packages

setup(
    name="pop-trading",
    version="0.1.0",
    description="Trading strategy evolution system using grammatical evolution",
    author="",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "pytest>=6.2.5",
        "ta>=0.10.0",
    ],
    python_requires=">=3.8",
)
