import os, math, xacro, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from ament_index_python.packages import get_package_share_path
from tf_transformations import euler_from_quaternion, quaternion_from_euler

class ThymioController(Node):
    def __init__(self):
        super().__init__('thymio_controller')

        # Publisher & Timer
        self.pub_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.loop)

        # Sensor subscriber
        self.create_subscription(LaserScan, '/thymio_front_center_proximity_sensor', self.scan_cb, 10)
        self.dist_ = float('inf')

        # Service clients
        self.del_client_   = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_client_ = self.create_client(SpawnEntity,  '/spawn_entity')
        urdf_path = os.path.join(get_package_share_path('thymio_mini_project'), 'urdf', 'thymio.urdf.xacro')
        self.robot_description_ = xacro.process_file(urdf_path).toxml()

        # Odometry
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.yaw_        = 0.0
        self.target_yaw_ = 0.0
        self.pos_        = {'x': 0.0, 'y': 0.0}
        self.start_pos_  = {'x': 0.0, 'y': 0.0}
        self.odom_ready_ = False
        self.spawn_yaw_  = math.pi / 2  # yaw absolu connu au spawn

        self.turns_    = 0
        self.MAX_TURNS = 4
        self.tick_     = 0

        # State machine: phase1 → deleting → spawning → phase2 → turning → stopped
        self.state_ = 'phase1'

    def scan_cb(self, msg):
        self.dist_ = float(msg.ranges[0])

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, self.yaw_ = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pos_ = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y
        }
        # Premier odom après spawn : fixer start_pos_ et démarrer phase2
        if self.state_ == 'spawning' and not self.odom_ready_:
            self.odom_ready_ = True
            self.start_pos_  = dict(self.pos_)
            self.state_      = 'phase2'

    def wrap_angle_(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def dist_traveled_(self):
        return math.sqrt(
            (self.pos_['x'] - self.start_pos_['x'])**2 +
            (self.pos_['y'] - self.start_pos_['y'])**2
        )

    def move_until_wall(self, cmd, stop_dist):
        if   self.dist_ <= stop_dist:          cmd.linear.x = 0.0
        elif self.dist_ < stop_dist + 0.05:    cmd.linear.x = 0.03
        elif self.dist_ < 0.25:                cmd.linear.x = 0.05
        else:                                  cmd.linear.x = 0.1
        return cmd

    def turn_angle(self, cmd):
        error = (self.target_yaw_ - self.yaw_ + math.pi) % (2 * math.pi) - math.pi
        if abs(error) > 0.01:                              # tolérance 0.01 rad = 0.57°
            speed = max(0.03, min(0.5, 1.5 * abs(error))) # vitesse min 0.03 pour éviter overshoot
            cmd.angular.z = math.copysign(speed, error)
            return cmd, False
        cmd.angular.z = 0.0
        return cmd, True

    def delete_robot(self):
        self.state_ = 'deleting'
        self.del_client_.wait_for_service()
        req      = DeleteEntity.Request()
        req.name = 'thymio'
        self.del_client_.call_async(req).add_done_callback(self.on_deleted)

    def on_deleted(self, future):
        self.spawn_client_.wait_for_service()
        req      = SpawnEntity.Request()
        req.name  = 'thymio'
        req.xml   = self.robot_description_
        pose = Pose()
        pose.position.x, pose.position.y = 0.5, 0.3
        q = quaternion_from_euler(0, 0, math.pi / 2)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        req.initial_pose = pose
        self.state_ = 'spawning'
        self.spawn_client_.call_async(req).add_done_callback(self.on_spawned)

    def on_spawned(self, future):
        self.turns_      = 0
        self.dist_       = float('inf')
        self.odom_ready_ = False
        # state_ reste 'spawning' jusqu'au premier odom

    def loop(self):
        cmd = Twist()
        self.tick_ += 1

        if self.state_ == 'phase1':
            cmd = self.move_until_wall(cmd, 0.05)
            if cmd.linear.x == 0.0:
                self.delete_robot()

        elif self.state_ == 'phase2':
            stop = 0.05 if self.turns_ >= self.MAX_TURNS else 0.015
            cmd  = self.move_until_wall(cmd, stop)
            if cmd.linear.x == 0.0 and self.dist_traveled_() > 0.20:
                if self.turns_ < self.MAX_TURNS:
                    # angle absolu basé sur le yaw de spawn → pas d'accumulation d'erreur
                    self.target_yaw_ = self.wrap_angle_(self.spawn_yaw_ + (self.turns_ + 1) * math.pi / 2)
                    self.state_      = 'turning'
                else:
                    self.state_ = 'stopped'

        elif self.state_ == 'turning':
            cmd, done = self.turn_angle(cmd)
            if done:
                self.turns_     += 1
                self.start_pos_  = dict(self.pos_)
                self.state_      = 'phase2'

        self.pub_.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ThymioController())
    rclpy.shutdown()
