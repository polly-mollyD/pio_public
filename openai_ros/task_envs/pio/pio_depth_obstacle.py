import rospy
import numpy
import os
import math
import random
from gym import spaces
from openai_ros.robot_envs import pio_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState



MARBLE_TABLE_RADIUS = 1.70

ROOM_OBSTACLES = {
    'room_train_1': [(1.3, 0.6, 0.35), (-1, -0.8, 0.25), (0.90, 1.35, 0.1), (1.35, -1.40, 0.1)],
    'room_train_2': [(1.3, 0.6, 0.35), (-1, -0.8, 0.25), (-1.08, 1, MARBLE_TABLE_RADIUS), (-1.5, -1.2, 0.1), (1.35, -1.30, 0.25), (1.70, 0.83, 0.1)],
    'room_train_3': [(-2.200642, 0.539028, 0.35), (1.651777, -1.367956, 0.35), (2.161544, -0.403086, 0.25), (0.049550, 1.507285, 0.25), (-1.500703, 0.688165, MARBLE_TABLE_RADIUS), (1.259907, 0.702212, 0.1), (-1.55, -1.30, 0.1), (0.60, -0.65, 0.1)],
    'room_train_4': [(0.579085, -1.698920, 0.35), (1.35, -1.30, 0.25), (1.350701, 0.345253, MARBLE_TABLE_RADIUS), (1.550032, 2.407817, 0.1), (1.405598, 0.132820, 0.1)],
    'room_train_5': [(-1.009772, -1.267472, 0.35), (-2.093810, -0.585917, 0.35), (-1.541170, -0.895915, 0.25), (-1.350701, 0.345253, MARBLE_TABLE_RADIUS), (-1.550032, 2.407817, 0.1), (-1.405598, 0.132820, 0.1), (-0.8, 2.0, 0.1)],
    'room_trial_01': [(-1.108, -2.189, 0.35), (1.35, -1.3, 0.25), (-0.19, 0.726, MARBLE_TABLE_RADIUS), (1.7, 0.83, 0.1), (1.55, -0.45, 0.1), (-1.55, -0.65, 0.1), (1.2, -1.35, 0.1)],
    'room_trial_02': [(1.3, 0.6, 0.35), (2.101910, 1.254600, 0.25), (0.0, -0.85, MARBLE_TABLE_RADIUS), (-0.739029, 2.288444, 0.10), (-0.65, 1.45, 0.1), (1.35, -1.4, 0.1)],
    'room_trial_03': [(1.373, 0.084, 0.35), (2.19, -0.934, 0.25), (-1.317049, 0.658561, MARBLE_TABLE_RADIUS), (-1.4, 0.65, 0.1), (1.55, -0.35, 0.1), (0.75, 0.75, 0.1)],
    'room_trial_04': [(0.341, -1.251, 0.35), (1.35, -1.3, 0.25), (1.387, 1.007, MARBLE_TABLE_RADIUS), (1.7, 0.83, 0.1), (-0.7, 1.45, 0.1), (1.55, 1.0, 0.1), (-1.15, -1.2, 0.1)],
    'room_trial_05': [(1.397, 1.492, 0.35), (1.35, -1.3, 0.25), (-0.791072, -0.468912, MARBLE_TABLE_RADIUS), (0.5, 1.5, 0.1), (0.930630, -2.262466, 0.1), (2.254, -0.65, 0.1)],
    'room_trial_06': [(1.3, 0.6, 0.35), (1.35, -1.3, 0.25), (0.65, 1.0, MARBLE_TABLE_RADIUS), (-1.35, -0.75, 0.1), (-1.55, 1.2, 0.1), (1.1, -1.5, 0.1)],
    'room_trial_07': [(0.746, 1.049, 0.35), (-2.165, 2.093, 0.25), (-0.321, -1.598, MARBLE_TABLE_RADIUS), (-1.4, -1.35, 0.1), (1.45, 1.25, 0.1), (2.259, 0.544, 0.1)],
    'room_trial_08': [(2.135, 1.173, 0.35), (-2.165, 2.093, 0.25), (-0.321, -1.598, MARBLE_TABLE_RADIUS), (0.306, 2.311, 0.1), (-2.265, -0.239, 0.1), (2.259, 0.544, 0.1)],
    'room_trial_09': [(0.466, 2.097, 0.35), (-0.487156, -1.704914, 0.25), (-0.766156, 0.908940, MARBLE_TABLE_RADIUS), (-1.5, -0.75, 0.1), (-0.7, 1.5, 0.1), (0.85, 0.35, 0.1)],
    'room_trial_10': [(-2.201, 0.539, 0.35), (1.652, -1.368, 0.35), (2.162, -0.403, 0.25), (0.05, 1.507, 0.25), (-0.5, 0.0, 1.35), (1.26, 0.702, 0.1), (-1.55, -1.3, 0.1), (0.6, -0.65, 0.1)],
}

ROOM_HALF_EXTENT = 1.5
MIN_CLEARANCE = 0.5


def _random_free_pose(obstacles):

    for _ in range(500):
        x = random.uniform(-ROOM_HALF_EXTENT, ROOM_HALF_EXTENT)
        y = random.uniform(-ROOM_HALF_EXTENT, ROOM_HALF_EXTENT)

        if all(
            math.hypot(x - ox, y - oy) >= MIN_CLEARANCE + radius
            for ox, oy, radius in obstacles
        ):
            return x, y, random.uniform(-math.pi, math.pi)

    return None


class PioDepthObstacleEnv(pio_env.PioEnv):
    def __init__(self):

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/pio/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"



        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/pio/config",
                               yaml_file_name="pio_depth_obstacle.yaml")


        super(PioDepthObstacleEnv, self).__init__(ros_ws_abspath)

        number_actions = rospy.get_param('/pio/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        self.reward_range = (-numpy.inf, numpy.inf)


        # Motion parameters convert discrete RL actions into /cmd_vel commands.
        self.linear_forward_speed = rospy.get_param('/pio/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/pio/linear_turn_speed')
        self.angular_speed = rospy.get_param('/pio/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/pio/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/pio/init_linear_turn_speed')


        self.depth_sectors = rospy.get_param('/pio/depth_sectors')
        self.min_range = rospy.get_param('/pio/min_range')
        self.medium_range = rospy.get_param('/pio/medium_range')
        self.max_depth_range = rospy.get_param('/pio/max_depth_range')


        high = numpy.full((self.depth_sectors), 2)
        low = numpy.full((self.depth_sectors), 0)
        self.observation_space = spaces.Box(low, high)

        # Rewards
        self.forwards_reward = rospy.get_param("/pio/forwards_reward")
        self.turn_reward = rospy.get_param("/pio/turn_reward")
        self.end_episode_points = rospy.get_param("/pio/end_episode_points")

        # Keep one CvBridge instance and the latest closest obstacle distance for done checks.
        self.bridge = CvBridge()
        self.closest_depth = float("inf")

        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


    def _reset_sim(self):
        # Reset Gazebo before choosing the robot's starting pose.
        # Otherwise, the reset would overwrite the randomized pose.
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()
        self._set_init_pose()
        self.gazebo.pauseSim()
        return True


    def _set_init_pose(self):
        obstacles = ROOM_OBSTACLES.get(os.environ.get('PIO_WORLD', ''))
        pose = _random_free_pose(obstacles) if obstacles else None
        if pose:
            x, y, yaw = pose
            state = ModelState()
            state.model_name = 'pio'
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = 0.05
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)
            state.reference_frame = 'world'
            self._set_model_state(state)

        self.move_base(self.init_linear_forward_speed,
                        self.init_linear_turn_speed)
        return True


    def _init_env_variables(self):
        self._episode_done = False


    def _set_action(self, action):

        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        self.move_base(linear_speed, angular_speed)

    def _get_obs(self):
        depth_image = self.get_camera_depth_image_raw()
        discretized_observations = self.discretize_depth_observation(depth_image)
        return discretized_observations


    def _is_done(self, observations):
        self._episode_done = (self.closest_depth < self.min_range)

        if self._episode_done:
            rospy.logwarn("PIO entered minimum range: %.3f m", self.closest_depth)

        return self._episode_done
    

    def _compute_reward(self, observations, done):
        if done:
            return -self.end_episode_points

        if self.last_action == "FORWARDS":
            return self.forwards_reward

        return self.turn_reward


    def discretize_depth_observation(self, depth_image_msg):

        try:
            depth_array = self.bridge.imgmsg_to_cv2(
                depth_image_msg,
                desired_encoding="passthrough"
            )
        except Exception as e:
            rospy.logerr("Could not convert depth image: " + str(e))
            self.closest_depth = 0.0
            return [0] * self.depth_sectors

        depth_array = numpy.array(depth_array, dtype=numpy.float32)

        # Some Gazebo/OpenNI depth topics may arrive with an extra channel dimension.
        # The sector logic expects a single 2D depth image: height x width.
        if len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0]

        # Some depth images are in millimeters, others in meters.
        # If values look too large, assume millimeters and convert to meters.
        finite_values = depth_array[numpy.isfinite(depth_array)]
        if finite_values.size > 0 and numpy.nanmedian(finite_values) > 20.0:
            depth_array = depth_array / 1000.0

        # Remove invalid values.
        depth_array[numpy.isnan(depth_array)] = self.max_depth_range
        depth_array[numpy.isinf(depth_array)] = self.max_depth_range
        depth_array = numpy.clip(depth_array, 0.0, self.max_depth_range)

        height, width = depth_array.shape

        # Use the middle/front part of the camera image.
        row_start = int(height * 0.10)
        row_end = int(height * 0.85)
        cropped = depth_array[row_start:row_end, :]

        # Split the camera horizontally; each sector becomes one element of the RL state.
        sector_width = int(width / self.depth_sectors)
        observations = []
        sector_depths = []

        for sector in range(self.depth_sectors):
            col_start = sector * sector_width

            if sector == self.depth_sectors - 1:
                col_end = width
            else:
                col_end = (sector + 1) * sector_width

            sector_data = cropped[:, col_start:col_end]

            valid_sector = sector_data[sector_data > 0.0]

            if valid_sector.size == 0:
                sector_distance = self.max_depth_range
            else:
                column_depths = []

                for column in numpy.array_split(sector_data, 8, axis=1):
                    valid_column = column[
                        column > 0.0
                    ]

                    if valid_column.size > 0:
                        column_depths.append(
                            numpy.percentile(
                                valid_column,
                                2
                            )
                        )

                if column_depths:
                    sector_distance = min(column_depths)
                else:
                    sector_distance = self.max_depth_range

            sector_depths.append(sector_distance)

            # Map continuous meters into the small symbolic state used by qlearn.py.
            if sector_distance < self.min_range:
                observations.append(0)
            elif sector_distance < self.medium_range:
                observations.append(1)
            else:
                observations.append(2)


        self.sector_depths = sector_depths
        self.closest_depth = min(self.sector_depths)

        return observations
