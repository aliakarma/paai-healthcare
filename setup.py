"""
setup.py — installs the paai_healthcare package so all sub-modules
can be imported without path manipulation.
"""

from setuptools import setup, find_packages

setup(
    name="paai_healthcare",
    version="1.0.0",
    description="Privacy-Aware Agentic AI for IoT Healthcare",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Toqeer Ali Syed, Ali Akarma et al.",
    author_email="toqeer@iu.edu.sa",
    url="https://github.com/toqeersyed/paai-healthcare",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scipy>=1.13",
        "scikit-learn>=1.4",
        "stable-baselines3>=2.3",
        "sb3-contrib>=2.3",
        "gymnasium>=0.29",
        "torch>=2.2",
        "rdflib>=7.0",
        "networkx>=3.3",
        "pyyaml>=6.0",
        "cryptography>=42.0",
        "tqdm>=4.66",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "paai-generate=data.synthetic.generate_patients:main",
            "paai-train=rl.train:main",
            "paai-evaluate=evaluation.run_evaluation:main",
        ]
    },
)
