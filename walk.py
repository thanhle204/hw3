#!/usr/bin/env python3
import math, random, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

def euclid(ax, ay, bx, by): return math.hypot(ax - bx, ay - by)

class Walker(Node):
    def __init__(self):
        super().__init__('walker_hw3')

        # --- IO
        self.pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subL = self.create_subscription(LaserScan, '/base_scan', self.scan_cb, 10)
        self.subO = self.create_subscription(Odometry,  '/odom',      self.odom_cb, 10)

        # --- Hospital-friendly thresholds
        self.front_block = 0.42      # enter SPIN if front < this
        self.front_clear = 0.50      # exit SPIN if front >= this (hysteresis)
        self.side_hug    = 0.33      # if side < this and front clear -> creep & steer away
        self.emerg       = 0.16      # true emergency stop

        # --- Random-wander gating (when space is open)
        self.open_front = 0.80       # consider “open” if front >= this
        self.open_side  = 0.55       # and both sides >= this

        # --- Speeds
        self.v_fwd   = 0.20
        self.v_creep = 0.08
        self.v_back  = -0.09
        self.w_spin  = 0.75
        self.w_steer = 0.55          # gentle steer when hugging
        self.w_wander = 0.35         # mild wander turn when space is open

        # --- Short backup caps (only a little!)
        self.back_dist_max = 0.14
        self.rear_emerg = 0.18
        self.rear_safe  = 0.32

        # --- Stuck logic
        self.idle_secs_to_spin = 2.0
        self.spin_no_improve_to_backup = 2.0

        # --- 300s cap
        self.start_time = self.get_clock().now()
        self.max_seconds = 300.0

        # --- State (scan/odom)
        self.f = self.l = self.r = float('inf')
        self.rear = float('inf'); self.has_rear = False
        self.x = self.y = 0.0
        self.start_xy = None
        self.max_disp = 0.0
        self.last_move_time = self.get_clock().now()

        # --- Modes: NORMAL | SPIN | BACKUP
        self.mode = 'NORMAL'
        self.spin_dir = +1.0
        self.best_front_seen = 0.0
        self.last_improve_time = None
        self.back_start_xy = None

        # --- Wander state (when open, keep a random slight heading for a short time)
        self.wander_until = None
        self.wander_wz = 0.0

        # --- Timers
        self.timer = self.create_timer(0.1, self.step)
        self.print_timer = self.create_timer(1.0, self.print_stats)  # print displacement once per sec

        self.get_logger().info('HW3 walker running (random-wander + displacement reporting).')

    # ---------------- Callbacks ----------------
    def scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0: return
        ang = np.linspace(msg.angle_min, msg.angle_max, n)
        rng = np.array(msg.ranges, dtype=float)
        rng[(~np.isfinite(rng)) | (rng <= 0.0)] = np.inf

        def pct(a0,a1):
            m = (ang >= a0) & (ang <= a1)
            v = rng[m]
            return float(np.percentile(v, 20)) if v.size else float('inf')

        self.f = pct(-0.30, 0.30)      # front ~±17°
        self.l = pct( 0.45, 1.20)      # left
        self.r = pct(-1.20,-0.45)      # right

        # Rear rays if lidar covers behind
        rear_mask = (ang <= -2.5) | (ang >= 2.5)
        rear_vals = rng[rear_mask]
        self.has_rear = rear_vals.size > 0
        self.rear = float(np.percentile(rear_vals, 20)) if self.has_rear else float('inf')

    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x; y = msg.pose.pose.position.y
        if self.start_xy is None:
            self.start_xy = (x, y)
        # progress detection
        if euclid(x, y, self.x, self.y) > 0.003:
            self.last_move_time = self.get_clock().now()
        self.x, self.y = x, y
        # displacement from start (furthest point)
        d = euclid(self.x, self.y, *self.start_xy)
        if d > self.max_disp:
            self.max_disp = d

    # ---------------- Control ----------------
    def step(self):
        # End after 300 sec
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9
        if elapsed > self.max_seconds:
            self._stop_and_exit(elapsed)
            return

        # Emergency stop
        if min(self.f, self.l, self.r) < self.emerg:
            return self.cmd(0.0, 0.0)

        # BACKUP: short, capped, rear-aware
        if self.mode == 'BACKUP':
            if self.has_rear and self.rear < self.rear_emerg:
                self.mode = 'SPIN'
                self._spin_reset(now)
                return self.cmd(0.0, self.spin_dir * self.w_spin)

            if self.back_start_xy is None:
                self.back_start_xy = (self.x, self.y)
            moved = euclid(self.x, self.y, *self.back_start_xy)

            target = self.back_dist_max
            if self.has_rear and self.rear < self.rear_safe:
                target = max(0.08, 0.5 * self.back_dist_max)

            if moved < target:
                return self.cmd(self.v_back, 0.0)

            self.mode = 'SPIN'
            self._spin_reset(now)
            return self.cmd(0.0, self.spin_dir * self.w_spin)

        # SPIN: rotate in place until front clears; if not improving, short backup then continue
        if self.mode == 'SPIN':
            if self.f >= self.front_clear:
                self.mode = 'NORMAL'
                self._maybe_start_wander(now)  # pick a wander heading immediately after finding an opening
                return self.cmd(self.v_fwd, self.wander_wz)

            if (self.last_improve_time is None) or (self.f > self.best_front_seen + 0.02):
                self.best_front_seen = self.f
                self.last_improve_time = now

            if self.last_improve_time and (now - self.last_improve_time).nanoseconds * 1e-9 > self.spin_no_improve_to_backup:
                self.mode = 'BACKUP'
                self.back_start_xy = (self.x, self.y)
                return self.cmd(self.v_back, 0.0)

            return self.cmd(0.0, self.spin_dir * self.w_spin)

        # NORMAL:
        # 1) Front blocked → enter SPIN (choose initial direction toward more open side)
        if self.f < self.front_block:
            self.mode = 'SPIN'
            self._spin_reset(now)
            self.spin_dir = +1.0 if self.l > self.r else -1.0
            return self.cmd(0.0, self.spin_dir * self.w_spin)

        # 2) Side-hug: front clear but one side too close → creep and steer away (keep moving!)
        if self.l < self.side_hug and self.r >= self.l + 0.02:
            self._cancel_wander()
            return self.cmd(self.v_creep, -self.w_steer)
        if self.r < self.side_hug and self.l >= self.r + 0.02:
            self._cancel_wander()
            return self.cmd(self.v_creep,  self.w_steer)

        # 3) Open-space wander: when front and both sides are open, keep a random mild heading for a short time
        if self._space_open():
            if self.wander_until is None or now >= self.wander_until:
                self._pick_wander(now)
            return self.cmd(self.v_fwd, self.wander_wz)

        # 4) Otherwise: go straight
        self._cancel_wander()
        return self.cmd(self.v_fwd, 0.0)

    # ------------- Wander helpers -------------
    def _space_open(self):
        return (self.f >= self.open_front) and (self.l >= self.open_side) and (self.r >= self.open_side)

    def _pick_wander(self, now):
        # choices: straight (50%), slight left (25%), slight right (25%)
        choice = random.random()
        if choice < 0.5:
            self.wander_wz = 0.0
        elif choice < 0.75:
            self.wander_wz = +self.w_wander
        else:
            self.wander_wz = -self.w_wander
        dur = random.uniform(1.2, 2.5)  # seconds to keep this heading
        self.wander_until = now + Duration(seconds=dur)

    def _maybe_start_wander(self, now):
        if self._space_open():
            self._pick_wander(now)
        else:
            self._cancel_wander()

    def _cancel_wander(self):
        self.wander_until = None
        self.wander_wz = 0.0

    # ------------- Spin helpers -------------
    def _spin_reset(self, now):
        self.best_front_seen = self.f
        self.last_improve_time = now

    # ------------- Output & shutdown -------------
    def cmd(self, vx, wz):
        # odom-based idle watchdog: if not moving for a while AND front is clear, nudge forward
        now = self.get_clock().now()
        idle = (now - self.last_move_time).nanoseconds * 1e-9
        if self.mode == 'NORMAL' and idle > self.idle_secs_to_spin and self.f >= self.front_clear:
            vx = self.v_creep
            # small steer away from the closer side when nudging
            wz = -self.w_steer if self.l < self.r else self.w_steer

        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def _stop_and_exit(self, elapsed):
        self.cmd(0.0, 0.0)
        self.get_logger().info(f'[HW3] Time: {elapsed:.1f}s | Max displacement: {self.max_disp:.2f} m (goal ≥ 10.0 m)')
        rclpy.shutdown()

    def print_stats(self):
        # Prints once per second so you can watch progress in terminal
        self.get_logger().info(f'Displacement: {self.max_disp:.2f} m | Front/L/R: {self.f:.2f}/{self.l:.2f}/{self.r:.2f}')

def main():
    rclpy.init()
    rclpy.spin(Walker())

if __name__ == '__main__':
    main()
