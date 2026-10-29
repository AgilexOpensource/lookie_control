import os
from glob import glob

from setuptools import setup

package_name = "kstone_mock"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Agilex Robotics",
    maintainer_email="contact@agilex.ai",
    description="Temporary kstone_ros stand-in: MuJoCo whole-body IK and viewer",
    license="Apache-2.0",
    entry_points={"console_scripts": ["kstone_mock_node = kstone_mock.mock_node:main"]},
)
