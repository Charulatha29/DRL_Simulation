#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
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
        self.marker_pub = self.create_publisher(Marker, '/visual_goal_marker', 10)
        
        # Load trained DRL Policy
        self.actor = tf.keras.models.load_model(model_path)
        self.planner = HybridAStarPlanner()
        
        self.max_v, self.max_w = 0.35, 1.8
        self.bot_pos = np.array([0.25, 0.25], dtype=np.float32)
        self.bot_theta = 0.0
        self.obs_pos = np.array([1.20, 0.20], dtype=np.float32)
        self.target_pos = np.array([2.15, 0.95], dtype=np.float32)
        self.v_curr, self.w_curr = 0.0, 0.0
        
        self.prev_d_obs = None
        self.prev_theta_obs = None
        self.dt = 0.1
        
        self.waypoints = []
        self.current_wp_idx = 0
        self.stuck_counter = 0
        
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

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "active_goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.2
        marker.color.a = 0.9
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def navigate_to_target(self, target_pos):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.waypoints = self.planner.plan(self.bot_pos, self.target_pos)
        self.current_wp_idx = 0
        self.stuck_counter = 0
        self.get_logger().info(f"New Target Set: {self.target_pos}. Waypoints: {len(self.waypoints)}")

    def control_loop(self):
        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        active_wp = self.waypoints[self.current_wp_idx]
        self.publish_goal_marker(active_wp[0], active_wp[1])
        
        dist_to_wp = np.linalg.norm(active_wp - self.bot_pos)
        
        if dist_to_wp < 0.15:
            self.current_wp_idx += 1
            self.stuck_counter = 0
            if self.current_wp_idx >= len(self.waypoints):
                self.get_logger().info("Target Reached!")
                self.cmd_pub.publish(Twist())
                return
            active_wp = self.waypoints[self.current_wp_idx]
        else:
            self.stuck_counter += 1

        # Trigger Dynamic Replanning if path is permanently blocked (>5 seconds)
        if self.stuck_counter > 50:
            self.get_logger().warn("Path blocked by obstacle! Triggering Hybrid A* Dynamic Replan...")
            self.navigate_to_target(self.target_pos)
            return

        # Construct 8D DRL State Vector
        g_vec = active_wp - self.bot_pos
        dist_goal = float(np.linalg.norm(g_vec))
        angle_goal = float((math.atan2(g_vec[1], g_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)

        o_vec = self.obs_pos - self.bot_pos
        dist_obs = float(np.linalg.norm(o_vec))
        angle_obs = float((math.atan2(o_vec[1], o_vec[0]) - self.bot_theta + np.pi) % (2 * np.pi) - np.pi)

        if self.prev_d_obs is not None:
            v_rel_x = float((dist_obs * math.cos(angle_obs) - self.prev_d_obs * math.cos(self.prev_theta_obs)) / self.dt)
            v_rel_y = float((dist_obs * math.sin(angle_obs) - self.prev_d_obs * math.sin(self.prev_theta_obs)) / self.dt)
        else:
            v_rel_x, v_rel_y = 0.0, 0.0

        self.prev_d_obs = dist_obs
        self.prev_theta_obs = angle_obs

        state = np.array([dist_goal, angle_goal, dist_obs, angle_obs, v_rel_x, v_rel_y, self.v_curr, self.w_curr], dtype=np.float32)

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
    navigator.navigate_to_target(target_pos=[2.1, 0.9])
    
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
