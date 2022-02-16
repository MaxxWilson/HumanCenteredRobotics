import os
import sys
from tkinter.tix import X_REGION

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from util import liegroup

import numpy as np
import math

from pnc.interface import Interface
from config.manipulator_config import ManipulatorConfig

from typing import Tuple
import matplotlib.pyplot as plt

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

        self.planned_traject = dict.fromkeys({"j1", "j2", "j3", "x", "y", "z"})
        for key in self.planned_traject:
            self.planned_traject[key] = list()

        self.robot_traject = dict.fromkeys({"time", "q1", "q2", "q3", "q1_vel", "q2_vel", "q3_vel", "tau1", "tau2", "tau3", "x", "y", "theta", "x_vel", "y_vel", "theta_vel", "a_ref"})
        for key in self.robot_traject:
            self.robot_traject[key] = list()

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
        # jtrq_cmd = self._compute_jpos_command()
        # TODO: Question 3
        # jtrq_cmd = self._compute_osc_command()
        # TODO: Question 4
        jtrq_cmd = self._compute_wbc_command()

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

        q_des = np.array([0.35, 1.57, 0.35])
        q_vel_des = np.array([0, 0, 0])

        # On first run, compute desired trajectory given start and end conditions for each joint
        if(self._count == 0):
            q = self._robot.get_q()
            q_dot = self._robot.get_q_dot()
            t_range = (0, 2)
            self.planned_traject["j1"] = self._compute_cubic_trajectory((q[0], q_des[0]), (q_dot[0], q_vel_des[0]), t_range)
            self.planned_traject["j2"] = self._compute_cubic_trajectory((q[1], q_des[1]), (q_dot[1], q_vel_des[1]), t_range)
            self.planned_traject["j3"] = self._compute_cubic_trajectory((q[2], q_des[2]), (q_dot[2], q_vel_des[2]), t_range)

        # initialize
        jtrq = np.zeros(self._robot.n_a)

        # Set PD Gains for each joint
        kp = 120.0
        kd = 60.0

        if(self._count < np.size(self.planned_traject["j1"]["time"])):
            # Set desired joint angle, velocity, and acceleration vectors
            qi_des = np.array([
                self.planned_traject["j1"]["pos"][self._count],
                self.planned_traject["j2"]["pos"][self._count],
                self.planned_traject["j3"]["pos"][self._count]])
            qi_vel_des = np.array([
                self.planned_traject["j1"]["vel"][self._count],
                self.planned_traject["j2"]["vel"][self._count],
                self.planned_traject["j3"]["vel"][self._count]])
            qi_accel_des = np.array([
                self.planned_traject["j1"]["accel"][self._count],
                self.planned_traject["j2"]["accel"][self._count],
                self.planned_traject["j3"]["accel"][self._count]])

            # Inverse Dynamics control
            q_ddot = (qi_accel_des + kp * (qi_des - self._robot.get_q()) + kd * (qi_vel_des - self._robot.get_q_dot()))
            jtrq = self._robot.get_mass_matrix() @ q_ddot + self._robot.get_coriolis() + self._robot.get_gravity()

            # Record joint torque commands
            self.robot_traject["tau1"].append(jtrq[0])
            self.robot_traject["tau2"].append(jtrq[1])
            self.robot_traject["tau3"].append(jtrq[2])

            # Record robots true state
            self.robot_traject["time"].append(self._running_time)
            self.robot_traject["q1"].append(self._robot.get_q()[0])
            self.robot_traject["q2"].append(self._robot.get_q()[1])
            self.robot_traject["q3"].append(self._robot.get_q()[2])
            self.robot_traject["q1_vel"].append(self._robot.get_q_dot()[0])
            self.robot_traject["q2_vel"].append(self._robot.get_q_dot()[1])
            self.robot_traject["q3_vel"].append(self._robot.get_q_dot()[2])

        else:
            # Plot planned vs true trajectories after motion is done
            plt.figure()
            plt.title("Joint 1")
            plt.subplot(3, 1, 1)
            plt.plot(self.planned_traject["j1"]["time"], self.planned_traject["j1"]["pos"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["q1"])
            plt.subplot(3, 1, 2)
            plt.plot(self.planned_traject["j1"]["time"], self.planned_traject["j1"]["vel"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["q1_vel"])
            plt.subplot(3, 1, 3)
            plt.plot(self.planned_traject["j1"]["time"], self.planned_traject["j1"]["accel"], linestyle='-')

            plt.figure()
            plt.title("Joint 2")
            plt.subplot(3, 1, 1)
            plt.plot(self.planned_traject["j2"]["time"], self.planned_traject["j2"]["pos"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["q2"])
            plt.subplot(3, 1, 2)
            plt.plot(self.planned_traject["j2"]["time"], self.planned_traject["j2"]["vel"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["q2_vel"])
            plt.subplot(3, 1, 3)
            plt.plot(self.planned_traject["j2"]["time"], self.planned_traject["j2"]["accel"], linestyle='-')

            plt.figure()
            plt.title("Joint 3")
            plt.subplot(3, 1, 1)
            plt.plot(self.planned_traject["j3"]["time"], self.planned_traject["j3"]["pos"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["q3"])
            plt.subplot(3, 1, 2)
            plt.plot(self.planned_traject["j3"]["time"], self.planned_traject["j3"]["vel"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["q3_vel"])
            plt.subplot(3, 1, 3)
            plt.plot(self.planned_traject["j3"]["time"], self.planned_traject["j3"]["accel"], linestyle='-')

            plt.figure()
            plt.title("Joint Torques")
            plt.subplot(3, 1, 1)
            plt.plot(self.robot_traject["time"], self.robot_traject["tau1"])
            plt.subplot(3, 1, 2)
            plt.plot(self.robot_traject["time"], self.robot_traject["tau2"])
            plt.subplot(3, 1, 3)
            plt.plot(self.robot_traject["time"], self.robot_traject["tau3"])

            plt.show(block=True)
            sys.exit(0)

        return jtrq

    def _compute_osc_command(self):
        # initialize
        jtrq = np.zeros(self._robot.n_a)

        xosc_des = np.array([1.57, 1, 1])
        x_vel_osc_des = np.array([0, 0, 0])

        [theta_i, x_i, y_i] = self.get_end_effector_position_2D("ee")
        [theta_vel_i, x_vel_i, y_vel_i] = self.get_end_effector_velocity_2D("ee")

        # Set PD Gains for each joint
        kp = 100.0
        kd = 60.0

        # On first run, compute desired trajectory given start and end conditions for each joint
        if(self._count == 0):
            t_range = (0, 2)
            self.planned_traject["theta"] = self._compute_cubic_trajectory((theta_i, xosc_des[0]), (theta_vel_i, x_vel_osc_des[0]), t_range)
            self.planned_traject["x"] = self._compute_cubic_trajectory((x_i, xosc_des[1]), (x_vel_i, x_vel_osc_des[1]), t_range)
            self.planned_traject["y"] = self._compute_cubic_trajectory((y_i, xosc_des[2]), (y_vel_i, x_vel_osc_des[2]), t_range)

        if(self._count < np.size(self.planned_traject["x"]["time"])):
            # Set desired joint angle, velocity, and acceleration vectors
            xi_osc_des = np.array([
                self.planned_traject["theta"]["pos"][self._count],
                self.planned_traject["x"]["pos"][self._count],
                self.planned_traject["y"]["pos"][self._count]])
            xi_vel_osc_des = np.array([
                self.planned_traject["theta"]["vel"][self._count],
                self.planned_traject["x"]["vel"][self._count],
                self.planned_traject["y"]["vel"][self._count]])
            xi_accel_osc_des = np.array([
                self.planned_traject["theta"]["accel"][self._count],
                self.planned_traject["x"]["accel"][self._count],
                self.planned_traject["y"]["accel"][self._count]])

            # Operational Space Control
            ee_jac = self._robot.get_link_jacobian("ee") # theta, x, y
            
            # Calculate acceleration reference in Task Space
            a_ref = (xi_accel_osc_des + kp * (xi_osc_des - np.array([theta_i, x_i, y_i])) + kd * (xi_vel_osc_des - np.array([theta_vel_i, x_vel_i, y_vel_i])))
            #M_eff = np.linalg.pinv(ee_jac @ self._robot.get_mass_matrix() @ np.transpose(ee_jac))
            np.set_printoptions( linewidth=100)
            #print(M_eff)
            
            #jtrq =  np.transpose(ee_jac) @ M_eff @ np.array([0, 0, a_ref[0], a_ref[1], a_ref[2], 0]) + self._robot.get_coriolis() + self._robot.get_gravity()

            aq_ref = np.linalg.pinv(self._robot.get_link_jacobian("ee")) @ (np.array([0, 0, a_ref[0], a_ref[1], a_ref[2], 0]) - self._robot.get_link_jacobian_dot_times_qdot("ee"))
            jtrq = self._robot.get_mass_matrix() @ aq_ref + self._robot.get_coriolis() + self._robot.get_gravity()

            # Record joint torque commands
            self.robot_traject["tau1"].append(jtrq[0])
            self.robot_traject["tau2"].append(jtrq[1])
            self.robot_traject["tau3"].append(jtrq[2])

            # Record robots true state
            self.robot_traject["time"].append(self._running_time)
            self.robot_traject["theta"].append(theta_i)
            self.robot_traject["x"].append(x_i)
            self.robot_traject["y"].append(y_i)
            self.robot_traject["theta_vel"].append(theta_vel_i)
            self.robot_traject["x_vel"].append(x_vel_i)
            self.robot_traject["y_vel"].append(y_vel_i)

        else:
            # Plot planned vs true trajectories after motion is done
            plt.figure()
            plt.title("X")
            plt.subplot(3, 1, 1)
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["pos"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["x"])
            plt.subplot(3, 1, 2)
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["vel"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["x_vel"])
            plt.subplot(3, 1, 3)
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["accel"], linestyle='-')

            plt.figure()
            plt.title("Y")
            plt.subplot(3, 1, 1)
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["pos"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["y"])
            plt.subplot(3, 1, 2)
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["vel"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["y_vel"])
            plt.subplot(3, 1, 3)
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["accel"], linestyle='-')

            plt.figure()
            plt.title("Theta")
            plt.subplot(3, 1, 1)
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["pos"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["theta"])
            plt.subplot(3, 1, 2)
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["vel"], linestyle='-')
            plt.plot(self.robot_traject["time"], self.robot_traject["theta_vel"])
            plt.subplot(3, 1, 3)
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["accel"], linestyle='-')

            # plt.figure()
            # plt.title("Joint Torques")
            # plt.subplot(3, 1, 1)
            # plt.plot(self.robot_traject["time"], self.robot_traject["tau1"])
            # plt.subplot(3, 1, 2)
            # plt.plot(self.robot_traject["time"], self.robot_traject["tau2"])
            # plt.subplot(3, 1, 3)
            # plt.plot(self.robot_traject["time"], self.robot_traject["tau3"])

            plt.show(block=True)
            sys.exit(0)

        return jtrq

    def _compute_cubic_trajectory(self, q: Tuple[float, float], q_dot: Tuple[float, float], t: Tuple[float, float]) -> np.ndarray:

        inv = np.linalg.inv(np.array([
            [t[0]**3,       t[0]**2,    t[0],   1],
            [t[1]**3,       t[1]**2,    t[1],   1],
            [3*t[0]**2,     2*t[0],     1,      0], 
            [3*t[1]**2,     2*t[1],     1,      0]]))

        [a, b, c, d] = inv @ np.transpose(np.array([q[0], q[1], q_dot[0], q_dot[1]]))
        timestamps = np.arange(t[0], t[1] + ManipulatorConfig.DT, ManipulatorConfig.DT)
        
        trajectory = dict.fromkeys({"time", "pos", "vel", "accel"}, np.array([]))
        trajectory["time"] = np.array(timestamps)
        trajectory["pos"] = np.array(a*timestamps**3 + b*timestamps**2 + c*timestamps + d)
        trajectory["vel"] = np.array(3*a*timestamps**2 + 2*b*timestamps + c)
        trajectory["accel"] = np.array(6*a*timestamps + 2*b)

        return trajectory
    
    def get_end_effector_position_2D(self, link: str) -> np.ndarray:
        link_SO3 = self._robot.get_link_iso(link)
        theta_i = math.acos(link_SO3[0, 0])
        x_i = link_SO3[0, 3]
        y_i = link_SO3[1, 3]
        return np.array([theta_i, x_i, y_i])
    
    def get_end_effector_velocity_2D(self, link: str) -> np.ndarray:
        [theta_vel, x_vel, y_vel] = self._robot.get_link_jacobian(link)[2:5, :] @ self._robot.get_q_dot()
        return np.array([theta_vel, x_vel, y_vel])