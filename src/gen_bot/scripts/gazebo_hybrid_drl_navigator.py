#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import tensorflow as tf
from hybrid_astar import HybridAStarPlanner

class GazeboHybridDRLNavigator(Node):
    def __init__(self, model_path="drl_actor_tf210.h5"):
        super().__init__('gazebo_hybrid_drl_navigator')
        
        self.create_subscription(Odometry, '/odom', self.bot_odom_cb, 10)
        self.create_subscription(Odometry, '/obstacle/odom', self.obs_odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Load trained DRL Policy
        self.actor = tf.keras.models.load_model(model_path)
        self.planner = HybridAStarPlanner()
        
        self.max_v, self.max_w = 0.35, 1.8
        self.bot_pos = np.array([0.25, 0.25], dtype=np.float32)
        self.bot_theta = 0.0
        self.obs_pos = np.array([1.20, 0.20], dtype=np.float32)
        self.v_curr, self.w_curr = 0.0, 0.0
        
        self.waypoints = []
        self.current_wp_idx = 0
        
        # 10 Hz Control Loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def bot_odom_cb(self, msg):
        self.bot_pos[0] = msg.pose.pose.position.x
        self.bot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.bot_theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        self.v_curr = msg.twist.twist.linear.x
        self.w_curr = msg.twist.twist.angular.z

    def obs_odom_cb(self, msg):
        self.obs_pos[0] = msg.pose.pose.position.x
        self.obs_pos[1] = msg.pose.pose.position.y

    def navigate_to_target(self, target_pos):
        """ Call Hybrid A* Planner for path to h1, h2, or h3 """
        self.waypoints = self.planner.plan(self.bot_pos, target_pos)
        self.current_wp_idx = 0
        self.get_logger().info(f"New Target Set: {target_pos}. Waypoints: {len(self.waypoints)}")

    def control_loop(self):
        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        active_wp = self.waypoints[self.current_wp_idx]
        dist_to_wp = np.linalg.norm(active_wp - self.bot_pos)
        
        if dist_to_wp < 0.15:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.get_logger().info("Target Reached!")
                self.cmd_pub.publish(Twist())
                return
            active_wp = self.waypoints[self.current_wp_idx]

        # Construct DRL State Vector
        g_vec = active_wp - self.bot_pos
        dist_goal = float(np.linalg.norm(g_vec))
        angle_goal = float((math.atan2(g_vec[1], g_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)

        o_vec = self.obs_pos - self.bot_pos
        dist_obs = float(np.linalg.norm(o_vec))
        angle_obs = float((math.atan2(o_vec[1], o_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)

        state = np.array([dist_goal, angle_goal, dist_obs, angle_obs, self.v_curr, self.w_curr], dtype=np.float32)

        # Infer velocity commands from DRL Actor
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.squeeze(self.actor(tf_state)).numpy()

        v_cmd = float(((action[0] + 1.0) / 2.0) * self.max_v)
        w_cmd = float(action[1] * self.max_w)

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = GazeboHybridDRLNavigator()
    
    # Set dynamic target for h1, h2, or h3 (e.g., target at x=2.1, y=0.9)
    navigator.navigate_to_target(target_pos=[2.1, 0.9])
    
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()