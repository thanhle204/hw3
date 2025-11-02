
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

        # --- Topics (Stage with enforce_prefixes:=false) ---
        self.cmd_pub = self.create_publisher(Twist, '/robot_0/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/robot_0/base_scan', self.scan_cb, 10)

        
        self.odom_sub = self.create_subscription(Odometry, '/robot_0/odom', self.odom_cb, 10)
        # self.odom_sub = self.create_subscription(Odometry, '/robot_0/base_pose_ground_truth', self.odom_cb, 10)

        # --- Behavior params
        self.safe_front = 0.8     # m: an toàn phía trước
        self.safe_side  = 0.6     # m: an toàn bên hông
        self.v_fwd      = 0.4     # m/s
        self.w_turn     = 0.9     # rad/s

        # --- Scan state ---
        self.min_front = None
        self.min_left  = None
        self.min_right = None

        # --- Odom / distance tracking ---
        self.start_pos = None
        self.last_pos  = None
        self.total_dist = 0.0

        # --- Run time ---
        self.start_time = self.get_clock().now()
        self.max_seconds = 300.0  # 5 minutes

        # --- Stuck detection ---
        self.last_move_time = self.get_clock().now()
        self.escape_until = None  # rclpy Time until which we keep escaping

        # Control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.step)

        self.get_logger().info('Walker HW3 started.')

    # ---------- Callbacks ----------

    def scan_cb(self, msg: LaserScan):
        """Compute min distances in front / left / right sectors."""
        n = len(msg.ranges)
        if n == 0:
            self.min_front = self.min_left = self.min_right = None
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, n)
        ranges = np.array(msg.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = np.inf  # replace NaN/inf

        # sectors (radians)
        front_mask = (angles > -0.35) & (angles < 0.35)         # ~±20°
        left_mask  = (angles >= 0.35) & (angles < 1.2)          # 20°..~70°
        right_mask = (angles > -1.2) & (angles <= -0.35)        # -70°..-20°

        def mins(mask):
            if not mask.any():
                return None
            m = np.min(ranges[mask])
            return float(m) if np.isfinite(m) else None

        self.min_front = mins(front_mask)
        self.min_left  = mins(left_mask)
        self.min_right = mins(right_mask)

    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.start_pos is None:
            self.start_pos = (x, y)
            self.last_pos  = (x, y)
            self.last_move_time = self.get_clock().now()
            return

        step = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
        self.total_dist += step
        self.last_pos = (x, y)

        # if moved enough, refresh last_move_time
        if step > 0.003:
            self.last_move_time = self.get_clock().now()

    # ---------- Control step ----------

    def step(self):
        # Stop after 300s
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9
        if elapsed >= self.max_seconds:
            self._stop_and_exit(elapsed)
            return

        # If escaping from stuck, keep turning until escape_until
        if self.escape_until is not None and now < self.escape_until:
            self._publish(0.0, self.w_turn)  # keep turning left
            return
        else:
            self.escape_until = None

        # Very simple obstacle avoidance
        f = self.min_front if self.min_front is not None else np.inf
        l = self.min_left  if self.min_left  is not None else np.inf
        r = self.min_right if self.min_right is not None else np.inf

        if f < self.safe_front:
            # Turn toward the side with more clearance
            if l > r:
                self._publish(0.0, +self.w_turn)
            else:
                self._publish(0.0, -self.w_turn)
        elif l < self.safe_side:
            self._publish(0.1, -self.w_turn)  # steer right a bit
        elif r < self.safe_side:
            self._publish(0.1, +self.w_turn)  # steer left a bit
        else:
            self._publish(self.v_fwd, 0.0)

        # Stuck detection: if no movement for ~2.5s, rotate to escape for 1.2s
        idle_sec = (now - self.last_move_time).nanoseconds * 1e-9
        if idle_sec > 2.5:
            self.escape_until = now + rclpy.time.Duration(seconds=1.2)
            self._publish(0.0, self.w_turn)

    # ---------- Helpers ----------

    def _publish(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.cmd_pub.publish(t)

    def _stop_and_exit(self, elapsed):
        self._publish(0.0, 0.0)
        self.get_logger().info(
            f'Run finished: {elapsed:.1f}s | distance ≈ {self.total_dist:.2f} m'
        )
        rclpy.shutdown()


def main():
    rclpy.init()
    node = Walker()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
