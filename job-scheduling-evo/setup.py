from setuptools import setup, find_packages

setup(
    name="job-scheduling-evo",
    version="1.0.0",
    description="Job Shop Scheduling với GA và PSO - So sánh với thuật toán truyền thống",
    author="Nhóm Thiết kế và Đánh giá Thuật toán",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src*"]),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "scipy>=1.13.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "jupyter>=1.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
