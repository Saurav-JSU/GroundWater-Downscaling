from setuptools import setup, find_packages

setup(
    name="grace_downscaling",
    version="0.1",
    packages=find_packages(where="src"),  # Tell it to look inside the "src" folder
    package_dir={"": "src"},              # Root is "src/"
    install_requires=[
        "geopandas",
        "pandas",
        "numpy",
        "xarray",
        "rioxarray",
        "rasterio",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "earthengine-api",
        "geemap",
        "requests",
        "tqdm"
    ],
    author="Saurav Bhattarai",
    description="GRACE downscaling project using Random Forest",
)
