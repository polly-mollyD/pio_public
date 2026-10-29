import rospy
import time
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros.openai_ros_common import ROSLauncher


class PioEnv(robot_gazebo_env.RobotGazeboEnv):
    """ROS and Gazebo interface for the Pio robot."""
    

    def __init__(self, ros_ws_abspath):
        """Launch the Pio simulation and initialize its ROS interface.

        Subscribe to odometry and depth images, and create the velocity publisher.
        Controller resets are disabled for this robot.
        """

        ROSLauncher(rospackage_name="pio_description",
                    launch_file_name="gazebo.launch",
                    ros_ws_abspath=ros_ws_abspath)

        self.controllers_list = []

        self.robot_name_space = ""

        super(PioEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False)




        self.gazebo.unpauseSim()
        self._check_all_sensors_ready()

        self._odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_callback)
        self._depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_depth_image_raw_callback)

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()


    def cleanup_subscribers(self):
        self._odom_sub.unregister()
        self._depth_sub.unregister()


    def _check_all_systems_ready(self):
        self._check_all_sensors_ready()
        return True


    def _check_all_sensors_ready(self):
        self._check_odom_ready()
        self._check_camera_depth_image_raw_ready()


    def _check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")

        return self.odom


    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image_raw = None
        while self.camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_depth_image_raw = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=1.0)
            except:
                rospy.logerr("Current /camera/depth/image_raw not ready yet, retrying for getting camera_depth_image_raw")
        return self.camera_depth_image_raw


    def _odom_callback(self, data):
        self.odom = data


    def _camera_depth_image_raw_callback(self, data):
        self.camera_depth_image_raw = data


    def _check_publishers_connection(self):
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass


    def move_base(self, linear_speed, angular_speed):
        """Publish a velocity command and wait 0.2 seconds."""
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)

        time.sleep(0.2)

    def get_odom(self):
        return self.odom

    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw
