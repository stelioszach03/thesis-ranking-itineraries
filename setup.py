"""
Setup script for NYC Itinerary Ranking project
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nyc-itinerary-ranking",
    version="1.0.0",
    author="Stelios Zacharioudakis",
    author_email="your.email@example.com",
    description="Quality-based itinerary planning with dynamic algorithms for NYC tourism",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/thesis-ranking-itineraries",
    project_urls={
        "Documentation": "https://github.com/yourusername/thesis-ranking-itineraries/docs",
        "Source": "https://github.com/yourusername/thesis-ranking-itineraries",
        "Thesis": "https://github.com/yourusername/thesis-ranking-itineraries/thesis/thesis_final.pdf",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "demo": [
            "flask>=2.1.0",
            "streamlit>=1.10.0",
            "folium>=0.12.0",
            "plotly>=5.8.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "jupyterlab>=3.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "nyc-planner=src.hybrid_planner:main",
            "nyc-demo=demo.demo_nyc:main",
            "prepare-nyc-data=src.prepare_nyc_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.csv", "*.yaml", "*.yml"],
    },
    keywords=[
        "itinerary planning",
        "tourism",
        "dynamic algorithms",
        "user preferences",
        "multi-criteria optimization",
        "NYC",
        "LPA*",
        "A* search",
        "greedy algorithms",
    ],
    project_name="NYC Itinerary Ranking",
    license="MIT",
)