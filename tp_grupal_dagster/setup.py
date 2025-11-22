from setuptools import find_packages, setup

setup(
    name="tp_grupal_dagster",
    packages=find_packages(exclude=["tp_grupal_dagster_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
