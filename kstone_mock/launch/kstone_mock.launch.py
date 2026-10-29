"""Launch the kstone_ros stand-in (MuJoCo whole-body IK + viewer)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(
        get_package_share_directory("kstone_mock"), "config",
        "lookie__pf-lookie_trio_r01__ee-revo2std_r01.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("forward_to_driver", default_value="false"),
        Node(
            package="kstone_mock",
            executable="kstone_mock_node",
            name="kstone_mock",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {
                    "viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
                    "forward_to_driver": ParameterValue(
                        LaunchConfiguration("forward_to_driver"), value_type=bool),
                },
            ],
        ),
    ])
