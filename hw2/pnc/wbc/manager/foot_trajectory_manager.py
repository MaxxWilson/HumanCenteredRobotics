import copy

import numpy as np
from scipy.spatial.transform import Rotation as R

from util import util


class FootTrajectoryManager(object):
    """
    Foot SE(3) Trajectory Manager
    -----------------------------
    Usage:
        use_current or
        initialize_swing_foot_trajectory --> update_swing_foot_desired
    """
    def __init__(self, pos_task, ori_task, robot):
        self._pos_task = pos_task
        self._ori_task = ori_task
        self._robot = robot

        self._swing_start_time = 0.
        self._swing_duration = 0.

        self._pos_traj_init_to_mid = None
        self._pos_traj_mid_to_end = None
        self._quat_hermite_curve = None

        assert self._pos_task.target_id == self._ori_task.target_id
        self._target_id = self._pos_task.target_id

        # Attribute
        self._swing_height = 0.05

    def use_current(self):
        foot_iso = self._robot.get_link_iso(self._target_id)
        foot_vel = self._robot.get_link_vel(self._target_id)

        foot_pos_des = foot_iso[0:3, 3]
        foot_lin_vel_des = foot_vel[3:6]
        self._pos_task.update_desired(foot_pos_des, foot_lin_vel_des,
                                      np.zeros(3))

        foot_rot_des = util.rot_to_quat(foot_iso[0:3, 0:3])
        foot_ang_vel_des = foot_vel[0:3]
        self._ori_task.update_desired(foot_rot_des, foot_ang_vel_des,
                                      np.zeros(3))

    def initialize_swing_foot_trajectory(self, start_time, swing_duration,
                                         landing_foot):
        pass

    def update_swing_foot_desired(self, curr_time):
        pass

    @property
    def swing_height(self):
        return self._swing_height

    @swing_height.setter
    def swing_height(self, val):
        self._swing_height = val
