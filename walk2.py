#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class Walker(Node):
    def __init__(self):
        super().__init__('walker_hw3')

        # --- Topics (adjusted for enforce_prefixes:=false)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/base_scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # --- Behavior params
        self.safe_front = 0.8
        self.safe_side  = 0.6
        self.v_fwd      = 0.4
        self.w_turn     = 0.9

        # --- Scan state ---
        self.min_front = self.min_left = self.min_right = None

        # --- Odom tracking ---
        self.start_pos = self.last_pos = None
        self.total_dist = 0.0

        # --- Timers ---
        self.start_time = self.get_clock().now()
        self.max_seconds = 300.0
        self.last_move_time = self.get_clock().now()
        self.escape_until = None

        self.timer = self.create_timer(0.1, self.step)
        self.get_logger().info('Walker HW3 started.')

    # ---------- Callbacks ----------

    def scan_cb(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return
        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        ranges = np.array(msg.ranges)
        ranges[~np.isfinite(ranges)] = np.inf

        def sect(min_a, max_a):
            mask = (angles >= min_a) & (angles <= max_a)
            vals = ranges[mask]
            return np.min(vals) if vals.size else np.inf

        self.min_front = sect(-0.35, 0.35)
        self.min_left  = sect(0.35, 1.2)
        self.min_right = sect(-1.2, -0.35)

    def odom_cb(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self.start_pos is None:
            self.start_pos = self.last_pos = (x, y)
            return
        step = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
        self.total_dist += step
        self.last_pos = (x, y)
        if step > 0.003:
            self.last_move_time = self.get_clock().now()

    # ---------- Main Control ----------

    def step(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9
        if elapsed > self.max_seconds:
            self._stop_and_exit(elapsed)
            return

        # Escape logic
        if self.escape_until and now < self.escape_until:
            self._publish(0.0, self.w_turn)
            return
        else:
            self.escape_until = None

        f = self.min_front or np.inf
        l = self.min_left or np.inf
        r = self.min_right or np.inf

        if f < self.safe_front:
            self._publish(0.0, self.w_turn if l > r else -self.w_turn)
        elif l < self.safe_side:
            self._publish(0.1, -self.w_turn)
        elif r < self.safe_side:
            self._publish(0.1, self.w_turn)
        else:
            self._publish(self.v_fwd, 0.0)

        idle = (now - self.last_move_time).nanoseconds * 1e-9
        if idle > 2.5:
            self.escape_until = now + rclpy.time.Duration(seconds=1.2)
            self._publish(0.0, self.w_turn)

    # ---------- Helpers ----------

    def _publish(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _stop_and_exit(self, elapsed):
        self._publish(0.0, 0.0)
        self.get_logger().info(f'Simulation done: {elapsed:.1f}s | Distance: {self.total_dist:.2f} m')
        rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(Walker())

if __name__ == '__main__':
    main()
