"""Launch only the Lookie teleop node.

xr_media_ros and kstone_ros (or kstone_mock) must be started separately.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(
        get_package_share_directory("lookie_teleop"), "config",
        "lookie__pf-lookie_trio_r01__ee-revo2std_r01.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("start_enabled", default_value="false"),
        DeclareLaunchArgument("view_mode", default_value="third_person",
                              choices=["third_person", "first_person"]),
        Node(
            package="lookie_teleop",
            executable="lookie_teleop_node",
            name="lookie_teleop",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {"start_enabled": ParameterValue(LaunchConfiguration("start_enabled"), value_type=bool),
                 "view_mode": LaunchConfiguration("view_mode")},
            ],
        ),
    ])
