import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from util import liegroup

import numpy as np

from pnc.interface import Interface
from config.manipulator_config import ManipulatorConfig


class ManipulatorInterface(Interface):
    def __init__(self):
        super(ManipulatorInterface, self).__init__()

        if ManipulatorConfig.DYN_LIB == "dart":
            from pnc.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                cwd + "/robot_model/manipulator/three_link_manipulator.urdf",
                True, ManipulatorConfig.PRINT_ROBOT_INFO)
        elif ManipulatorConfig.DYN_LIB == "pinocchio":
            from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
            self._robot = PinocchioRobotSystem(
                cwd + "/robot_model/manipulator/three_link_manipulator.urdf",
                cwd + "/robot_model/manipulator", True,
                ManipulatorConfig.PRINT_ROBOT_INFO)
        else:
            raise ValueError("wrong dynamics library")

    def get_command(self, sensor_data):
        # Update Robot
        self._robot.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

        # TODO: Question 2
        jtrq_cmd = self._compute_jpos_command()
        # TODO: Question 3
        # jtrq_cmd = self._compute_osc_command()
        # TODO: Question 4
        # jtrq_cmd = self._compute_wbc_command()

        jpos_cmd = np.zeros_like(jtrq_cmd)
        jvel_cmd = np.zeros_like(jtrq_cmd)

        # Compute Cmd
        command = self._robot.create_cmd_ordered_dict(jpos_cmd, jvel_cmd,
                                                      jtrq_cmd)

        # Increase time variables
        self._count += 1
        self._running_time += ManipulatorConfig.DT

        return command

    def _compute_wbc_command(self):

        # initialize
        jtrq = np.zeros(self._robot.n_a)
        kp1 = 100
        kd1 = 20
        kp2 = 50
        kd2 = 14

        return jtrq

    def _compute_jpos_command(self):

        # initialize
        jtrq = np.zeros(self._robot.n_a)

        # Set PD Gains for each joint
        kp = np.array([6.0, 5.0, 3.0])
        kd = np.array([10.0, 5.0, 3.0])
        q_des = np.array([0.35, 1.57, 0.35])
        q_dot_des = np.array([0, 0, 0])

        jtrq = kp * (q_des - self._robot.get_q()) + kd * (q_dot_des - self._robot.get_q_dot())

        return jtrq

    def _compute_osc_command(self):
        # initialize
        jtrq = np.zeros(self._robot.n_a)
        kp = 100
        kd = 20
        xosc_des = np.array([1, 1, 1.57])

        return jtrq
