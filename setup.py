import os

from setuptools import find_packages, setup
from glob import glob

package_name = "home_nlp"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="efliao@cs.nctu.edu.tw",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mic_node = home_nlp.mic_node:main",
            "asr_node = home_nlp.asr_node:main",
            "ws_asr_node = home_nlp.ws_asr_node:main",
            "llm_node = home_nlp.llm_node:main",
        ],
    },
)
