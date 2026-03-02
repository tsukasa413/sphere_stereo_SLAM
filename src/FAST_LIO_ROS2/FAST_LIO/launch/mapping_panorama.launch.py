#!/usr/bin/env python3
"""
FAST-LIO Launch File for Omnidirectional Panorama SLAM

This launch file integrates FAST-LIO with the 4-eye fisheye panorama SLAM system.
It starts:
  1. FAST-LIO mapping node with panorama-optimized config
  2. RViz for visualization (optional)

Usage:
    ros2 launch fast_lio mapping_panorama.launch.py
    ros2 launch fast_lio mapping_panorama.launch.py rviz:=false
    ros2 launch fast_lio mapping_panorama.launch.py config_path:=/custom/path/config.yaml
"""

import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

from launch_ros.actions import Node


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')

    # === Parameters with default values (override config file) ===
    feature_extract_enable_param = LaunchConfiguration('feature_extract_enable', default='false')
    point_filter_num_param = LaunchConfiguration('point_filter_num', default='1')  # No downsampling for 5Hz
    max_iteration_param = LaunchConfiguration('max_iteration', default='4')  # Reduced for real-time
    filter_size_surf_param = LaunchConfiguration('filter_size_surf', default='0.1')  # Small for density
    filter_size_map_param = LaunchConfiguration('filter_size_map', default='0.3')
    cube_side_length_param = LaunchConfiguration('cube_side_length', default='1000.0')  # Large for drift prevention
    runtime_pos_log_enable_param = LaunchConfiguration('runtime_pos_log_enable', default='true')

    # === Default paths ===
    default_config_path = os.path.join(package_path, 'config', 'panorama_config.yaml')
    default_rviz_config_path = os.path.join(package_path, 'rviz_cfg', 'fastlio.rviz')

    # === Launch arguments ===
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (ROSBag playback) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path (panorama_config.yaml)'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz for visualization'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )

    # === FAST-LIO Node ===
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fastlio_mapping',
        parameters=[
            config_path,
            {
                'use_sim_time': use_sim_time,
                'feature_extract_enable': feature_extract_enable_param,
                'point_filter_num': point_filter_num_param,
                'max_iteration': max_iteration_param,
                'filter_size_surf': filter_size_surf_param,
                'filter_size_map': filter_size_map_param,
                'cube_side_length': cube_side_length_param,
                'runtime_pos_log_enable': runtime_pos_log_enable_param
            }
        ],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )

    # === RViz Node (Optional) ===
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use),
        output='screen'
    )

    # === Info Message ===
    info_msg = LogInfo(
        msg=[
            '\n',
            '════════════════════════════════════════════════════════════════════════════════\n',
            '  FAST-LIO Omnidirectional Panorama SLAM                                       \n',
            '════════════════════════════════════════════════════════════════════════════════\n',
            '  Config: ', config_path, '\n',
            '  Topics:                                                                       \n',
            '    - Input Point Cloud: /omnidirectional/point_cloud                          \n',
            '    - Input IMU:         /camera/imu                                           \n',
            '    - Output Odometry:   /Odometry                                             \n',
            '    - Output Map:        /cloud_registered                                     \n',
            '  Frame Rate: 5Hz point cloud + 200Hz IMU                                      \n',
            '  Optimization: IEKF with online extrinsic calibration                         \n',
            '════════════════════════════════════════════════════════════════════════════════\n',
            '\n'
        ]
    )

    # === Build Launch Description ===
    ld = LaunchDescription()
    
    # Add arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    
    # Add info message
    ld.add_action(info_msg)
    
    # Add nodes
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)

    return ld
