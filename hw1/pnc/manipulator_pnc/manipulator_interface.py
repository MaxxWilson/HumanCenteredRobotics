from array import array
import os
import sys
from tkinter.tix import X_REGION

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from util import liegroup

import numpy as np
from numpy.linalg import pinv
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

        self.robot_traject = dict.fromkeys({
            "time",
            "q1", "q2", "q3",
            "q1_vel", "q2_vel", "q3_vel",
            "tau1", "tau2", "tau3",
            "x", "y", "theta",
            "x_vel", "y_vel", "theta_vel",
            "a_ref", "force", "d_obs"})

        for key in self.robot_traject:
            self.robot_traject[key] = list()
            
        self.obstacle = [np.array([0.5, 1.5, 0]), np.array([3, 1.0, 0])]
            
        np.set_printoptions(linewidth=100)

    def get_command(self, sensor_data):
        # Update Robot
        self._robot.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

        # Question 2
        # jtrq_cmd = self._compute_jpos_command()
        
        # Question 3
        # jtrq_cmd = self._compute_osc_command()
        
        # Question 4
        # jtrq_cmd = self._compute_wbc_command()
        
        # Question 5
        jtrq_cmd = self._compute_obstacle_avoidance_cmd()

        jpos_cmd = np.zeros_like(jtrq_cmd)
        jvel_cmd = np.zeros_like(jtrq_cmd)

        # Compute Cmd
        command = self._robot.create_cmd_ordered_dict(jpos_cmd, jvel_cmd,
                                                      jtrq_cmd)

        # Increase time variables
        self._count += 1
        self._running_time += ManipulatorConfig.DT

        return command

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
            plt.subplot(2, 1, 1).set_title("Joint 1 Angle")
            plt.plot(self.planned_traject["j1"]["time"], self.planned_traject["j1"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q1"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 1 Velocity")
            plt.plot(self.planned_traject["j1"]["time"], self.planned_traject["j1"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q1_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 2 Angle")
            plt.plot(self.planned_traject["j2"]["time"], self.planned_traject["j2"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q2"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 2 Velocity")
            plt.plot(self.planned_traject["j2"]["time"], self.planned_traject["j2"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q2_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)
            
            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 3 Angle")
            plt.plot(self.planned_traject["j3"]["time"], self.planned_traject["j3"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q3"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 3 Velocity")
            plt.plot(self.planned_traject["j3"]["time"], self.planned_traject["j3"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q3_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(3, 1, 1).set_title("Joint 1 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau1"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplot(3, 1, 2).set_title("Joint 2 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau2"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplot(3, 1, 3).set_title("Joint 3 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau3"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplots_adjust(hspace=1.00)
            
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
            # Set desired position, velocity, and acceleration vectors
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
            
            # Calculate acceleration reference in Task Space
            a_ref = (xi_accel_osc_des + kp * (xi_osc_des - np.array([theta_i, x_i, y_i])) + kd * (xi_vel_osc_des - np.array([theta_vel_i, x_vel_i, y_vel_i])))

            aq_ref = pinv(self._robot.get_link_jacobian("ee")) @ (np.array([0, 0, a_ref[0], a_ref[1], a_ref[2], 0]) - self._robot.get_link_jacobian_dot_times_qdot("ee"))
            
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
            plt.subplot(2, 1, 1).set_title("X Position")
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["x"], label = "Actual")
            plt.ylabel("Distance (m)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("X Velocity")
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["x_vel"], label = "Actual")
            plt.ylabel("Velocity (m/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Y Position")
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["y"], label = "Actual")
            plt.ylabel("Distance (m)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Y Velocity")
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["y_vel"], label = "Actual")
            plt.ylabel("Velocity (m/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)
            
            plt.figure()
            plt.subplot(2, 1, 1).set_title("Theta Angle")
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["theta"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Theta Velocity")
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["theta_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(3, 1, 1).set_title("Joint 1 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau1"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplot(3, 1, 2).set_title("Joint 2 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau2"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplot(3, 1, 3).set_title("Joint 3 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau3"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplots_adjust(hspace=1.00)
            
            plt.show(block=True)
            sys.exit(0)

        return jtrq

    def _compute_wbc_command(self):

        # initialize
        jtrq = np.zeros(self._robot.n_a)
        kp1 = 80
        kd1 = 30
        kp2 = 50
        kd2 = 10
        
        x1_des = np.array([np.pi/2])
        x1_vel_des = np.array([0])
        
        x2_des = np.array([0, 0, 0])
        x2_vel_des = np.array([0, 0, 0])
        
        [x1_i, _, _] = self.get_end_effector_position_2D("ee")
        [x1_vel_i, _, _] = self.get_end_effector_velocity_2D("ee")
        
        x2_i = self._robot.get_q()
        x2_vel_i = self._robot.get_q_dot()
        
        # On first run, compute desired trajectory given start and end conditions for each joint
        if(self._count == 0):
            t_range = (0, 2)
            self.planned_traject["theta"] = self._compute_cubic_trajectory((x1_i, x1_des[0]), (x1_vel_i, x1_vel_des[0]), t_range)
            self.planned_traject["q1"] = self._compute_cubic_trajectory((x2_i[0], x2_des[0]), (x2_vel_i[0], x2_vel_des[0]), t_range)
            self.planned_traject["q2"] = self._compute_cubic_trajectory((x2_i[1], x2_des[1]), (x2_vel_i[1], x2_vel_des[1]), t_range)
            self.planned_traject["q3"] = self._compute_cubic_trajectory((x2_i[2], x2_des[2]), (x2_vel_i[2], x2_vel_des[2]), t_range)

        if(self._count < np.size(self.planned_traject["theta"]["time"])):
            
            # Get current setpoints for each task
            x1_des = self.planned_traject["theta"]["pos"][self._count]
            x1_vel_des = self.planned_traject["theta"]["vel"][self._count]
            x1_accel_des = self.planned_traject["theta"]["accel"][self._count]
            
            x2_des = np.array([
                self.planned_traject["q1"]["pos"][self._count],
                self.planned_traject["q2"]["pos"][self._count],
                self.planned_traject["q3"]["pos"][self._count]])
            x2_vel_des = np.array([
                self.planned_traject["q1"]["vel"][self._count],
                self.planned_traject["q2"]["vel"][self._count],
                self.planned_traject["q3"]["vel"][self._count]])
            x2_accel_des = np.array([
                self.planned_traject["q1"]["accel"][self._count],
                self.planned_traject["q2"]["accel"][self._count],
                self.planned_traject["q3"]["accel"][self._count]])
            
            A = self._robot.get_mass_matrix()
            b = self._robot.get_coriolis()
            g = self._robot.get_gravity()
            A_pinv = pinv(A)
            
            j_x1 = np.array([self._robot.get_link_jacobian("ee")[2, :]])    # Jacobian from q1, q2, q3 to theta [1x3]
            j_x1_t = np.transpose(j_x1)                                     # [3x1]
            j_x1_pinv = pinv(j_x1)                                          # [3x1]
            
            M_x1 = pinv(j_x1 @ A_pinv @ j_x1_t).reshape(1, 1)               # [1x3]*[3x3]*[3x1] = scalar
            
            j_x1_bar = A_pinv @ j_x1_t @ M_x1                               # [3x3]*[3x1]*scalar = [3x1]
            N_x1 = np.eye(3) - j_x1_bar @ j_x1                              # [3x3] - [3x1]*[1x3] = [3x3], Rank 2
            N_x1_t = np.transpose(N_x1)
            
            j_x2_x1 = N_x1  # Task 2 is already in joint space
            j_x2_x1_t = np.transpose(j_x2_x1)
            M_x2_x1 = pinv(j_x2_x1 @ A_pinv @ j_x2_x1_t)
            
            F_x1 = M_x1 @ (x1_accel_des + kp1 * (x1_des - x1_i) + kd1 * (x1_vel_des - x1_vel_i) - self._robot.get_link_jacobian_dot_times_qdot("ee")[2]).reshape(1, 1)
            F_x2 = M_x2_x1 @ (x2_accel_des + kp2 * (x2_des - x2_i) + kd2 * (x2_vel_des - x2_vel_i))

            jtrq = j_x1_t @ F_x1 + (j_x2_x1_t @ F_x2 + b+g).reshape(3, 1)
            
            # Record joint torque commands
            self.robot_traject["tau1"].append(jtrq[0])
            self.robot_traject["tau2"].append(jtrq[1])
            self.robot_traject["tau3"].append(jtrq[2])

            # Record robots true state
            self.robot_traject["time"].append(self._running_time)
            self.robot_traject["theta"].append(x1_i)
            self.robot_traject["theta_vel"].append(x1_vel_i)
            self.robot_traject["q1"].append(self._robot.get_q()[0])
            self.robot_traject["q2"].append(self._robot.get_q()[1])
            self.robot_traject["q3"].append(self._robot.get_q()[2])
            self.robot_traject["q1_vel"].append(self._robot.get_q_dot()[0])
            self.robot_traject["q2_vel"].append(self._robot.get_q_dot()[1])
            self.robot_traject["q3_vel"].append(self._robot.get_q_dot()[2])
            
        else:
            # Plot planned vs true trajectories after motion is done
            plt.figure()
            plt.subplot(2, 1, 1).set_title("Theta Angle")
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["theta"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Theta Velocity")
            plt.plot(self.planned_traject["theta"]["time"], self.planned_traject["theta"]["vel"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["theta_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 1 Angle")
            plt.plot(self.planned_traject["q1"]["time"], self.planned_traject["q1"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q1"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 1 Velocity")
            plt.plot(self.planned_traject["q1"]["time"], self.planned_traject["q1"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q1_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 2 Angle")
            plt.plot(self.planned_traject["q2"]["time"], self.planned_traject["q2"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q2"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 2 Velocity")
            plt.plot(self.planned_traject["q2"]["time"], self.planned_traject["q2"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q2_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)
            
            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 3 Angle")
            plt.plot(self.planned_traject["q3"]["time"], self.planned_traject["q3"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q3"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 3 Velocity")
            plt.plot(self.planned_traject["q3"]["time"], self.planned_traject["q3"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["q3_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            print("Final Joint Angles: ")
            print("q1:", self._robot.get_q()[0] * 180 / np.pi)
            print("q2:", self._robot.get_q()[1] * 180 / np.pi)
            print("q3:", self._robot.get_q()[2] * 180 / np.pi)

            plt.show(block=True)
            sys.exit(0)

        return jtrq
    
    def _compute_obstacle_avoidance_cmd(self):
        # initialize
        jtrq = np.zeros(self._robot.n_a)
        
        # Constants
        kp1 = 8
        kd1 = 5
        kp2 = 1
        kd2 = 1
        kd_theta = 3
        beta = 0.5

        ee_des = np.array([1.5, 2])
        ee_vel_des = np.array([0, 0])
        
        [_, x_i, y_i] = self.get_end_effector_position_2D("ee")
        [_, x_vel_i, y_vel_i] = self.get_end_effector_velocity_2D("ee")
        
        # On first run, compute desired trajectory given start and end conditions for each joint
        if(self._count == 0):
            t_range = (0, 2)
            self.planned_traject["x"] = self._compute_cubic_trajectory((x_i, ee_des[0]), (x_vel_i, ee_vel_des[0]), t_range)
            self.planned_traject["y"] = self._compute_cubic_trajectory((y_i, ee_des[1]), (y_vel_i, ee_vel_des[1]), t_range)

        if(self._count < np.size(self.planned_traject["x"]["time"])):
            # Set desired end effector position, velocity, and acceleration vectors
            xi_osc_des = np.array([
                self.planned_traject["x"]["pos"][self._count],
                self.planned_traject["y"]["pos"][self._count]])
            xi_vel_osc_des = np.array([
                self.planned_traject["x"]["vel"][self._count],
                self.planned_traject["y"]["vel"][self._count]])
            xi_accel_osc_des = np.array([
                self.planned_traject["x"]["accel"][self._count],
                self.planned_traject["y"]["accel"][self._count]])

        else:

            xi_osc_des = np.array([
                self.planned_traject["x"]["pos"][-1],
                self.planned_traject["y"]["pos"][-1]])
            xi_vel_osc_des = np.array([
                self.planned_traject["x"]["vel"][-1],
                self.planned_traject["y"]["vel"][-1]])
            xi_accel_osc_des = np.array([
                self.planned_traject["x"]["accel"][-1],
                self.planned_traject["y"]["accel"][-1]])

        xi_osc_des = ee_des
        xi_vel_osc_des = np.array([0, 0])
        xi_accel_osc_des = np.array([0, 0])

        repulsion_force = 0
        d_obs = 0
        if(self._count == 760):
            print()
        if(self.get_ee_dist_to_obstacle() > beta):
            # Operational Space Control
            # Calculate acceleration reference in Task Space
            a_ref = (xi_accel_osc_des + kp2 * (xi_osc_des - np.array([x_i, y_i])) + kd2 * (xi_vel_osc_des - np.array([x_vel_i, y_vel_i])))
            [theta_ref, _, _] = -kd_theta*self.get_end_effector_velocity_2D()

            # Convert to joint space commands
            aq_ref = pinv(self._robot.get_link_jacobian("ee")) @ (np.array([0, 0, theta_ref, a_ref[0], a_ref[1], 0]) - self._robot.get_link_jacobian_dot_times_qdot("ee"))
            jtrq = self._robot.get_mass_matrix() @ aq_ref + self._robot.get_coriolis() + self._robot.get_gravity()
        else:
            # Prioritized Obstacle Avoidance
            j_p = self._robot.get_link_jacobian("ee")[3:5, :]

            r_to_obs = self.get_rotation_to_obstacle()
            s_obs = np.array([[0, 0], [0, 1]])
            j_obs = r_to_obs @ s_obs @ r_to_obs.T @ j_p
            M_obs = pinv(j_obs @ pinv(self._robot.get_mass_matrix()) @ j_obs.T)
            
            d_obs = ((self.get_vector_ee_to_obstacle_proj() - beta * self.get_obstacle_unit_normal()).T@(self.get_vector_ee_to_obstacle_proj() - beta * self.get_obstacle_unit_normal()))[0,0]
            ee_vel_obs = r_to_obs.T@np.array([[x_vel_i], [y_vel_i]])

            a_ref_obs = r_to_obs @ ((-kp1 * np.array([0, -2*(d_obs - beta)]) - kd1 * np.array([0, ee_vel_obs[1, 0]])).reshape(2, 1))
            F_obs = M_obs @ (a_ref_obs - (self._robot.get_link_jacobian_dot_times_qdot("ee")[3:5]).reshape(2, 1))

            j_obs_bar = pinv(self._robot.get_mass_matrix()) @ j_obs.T @ M_obs
            N_obs = np.eye(3) - j_obs_bar @ j_obs
            j_p_o = j_p @ N_obs
            M_p_o = pinv(j_p_o @ pinv(self._robot.get_mass_matrix()) @ j_p_o.T)
            a_ref = (xi_accel_osc_des + kp2 * (xi_osc_des - np.array([x_i, y_i])) + kd2 * (xi_vel_osc_des - np.array([x_vel_i, y_vel_i]))).reshape(2, 1)
            F_p_o = M_p_o @ (a_ref - (self._robot.get_link_jacobian_dot_times_qdot("ee")[3:5]).reshape(2, 1))

            # Set Joint Torques
            jtrq = j_obs.T @ F_obs + j_p_o.T @ F_p_o + (self._robot.get_coriolis() + self._robot.get_gravity()).reshape(3, 1)
            repulsion_force = F_obs[1, 0]
        
        # Check for jacobian singularity
        print(np.linalg.cond(self._robot.get_link_jacobian("ee")))
        if(abs(np.linalg.cond(self._robot.get_link_jacobian("ee"))) > 30):
            jtrq = np.array([0, 0, 0])

        # Record robots true state
        self.robot_traject["time"].append(self._running_time)

        # EE States
        self.robot_traject["x"].append(x_i)
        self.robot_traject["y"].append(y_i)
        self.robot_traject["x_vel"].append(x_vel_i)
        self.robot_traject["y_vel"].append(y_vel_i)

        # Joint States
        self.robot_traject["q1"].append(self._robot.get_q()[0])
        self.robot_traject["q2"].append(self._robot.get_q()[1])
        self.robot_traject["q3"].append(self._robot.get_q()[2])
        self.robot_traject["q1_vel"].append(self._robot.get_q_dot()[0])
        self.robot_traject["q2_vel"].append(self._robot.get_q_dot()[1])
        self.robot_traject["q3_vel"].append(self._robot.get_q_dot()[2])

        # Joint torque commands
        self.robot_traject["tau1"].append(jtrq[0])
        self.robot_traject["tau2"].append(jtrq[1])
        self.robot_traject["tau3"].append(jtrq[2])

        # Repulsion force
        self.robot_traject["force"].append(repulsion_force)
        self.robot_traject["d_obs"].append(d_obs)

        if(self._count > 1000):
            # Plot
            plt.figure()
            plt.subplot(2, 1, 1).set_title("X Position")
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["x"], label = "Actual")
            plt.ylabel("Distance (m)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("X Velocity")
            plt.plot(self.planned_traject["x"]["time"], self.planned_traject["x"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["x_vel"], label = "Actual")
            plt.ylabel("Velocity (m/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Y Position")
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["pos"], linestyle='-', label="Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["y"], label = "Actual")
            plt.ylabel("Distance (m)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Y Velocity")
            plt.plot(self.planned_traject["y"]["time"], self.planned_traject["y"]["vel"], linestyle='-', label = "Reference")
            plt.plot(self.robot_traject["time"], self.robot_traject["y_vel"], label = "Actual")
            plt.ylabel("Velocity (m/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 1 Angle")
            plt.plot(self.robot_traject["time"], self.robot_traject["q1"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Joint 1 Velocity")
            plt.plot(self.robot_traject["time"], self.robot_traject["q1_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 2 Angle")
            plt.plot(self.robot_traject["time"], self.robot_traject["q2"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.subplot(2, 1, 2).set_title("Joint 2 Velocity")
            plt.plot(self.robot_traject["time"], self.robot_traject["q2_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.subplots_adjust(hspace=0.5)
            
            plt.figure()
            plt.subplot(2, 1, 1).set_title("Joint 3 Angle")
            plt.plot(self.robot_traject["time"], self.robot_traject["q3"], label = "Actual")
            plt.ylabel("Angle (rad)")
            plt.xlabel("Time (s)")
            plt.subplot(2, 1, 2).set_title("Joint 3 Velocity")
            plt.plot(self.robot_traject["time"], self.robot_traject["q3_vel"], label = "Actual")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.xlabel("Time (s)")
            plt.subplots_adjust(hspace=0.5)

            plt.figure()
            plt.subplot(3, 1, 1).set_title("Joint 1 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau1"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplot(3, 1, 2).set_title("Joint 2 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau2"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplot(3, 1, 3).set_title("Joint 3 Torque")
            plt.plot(self.robot_traject["time"], self.robot_traject["tau3"])
            plt.ylabel("Torque")
            plt.xlabel("Time (s)")
            plt.subplots_adjust(hspace=1.00)

            plt.figure()
            plt.subplot(2, 1, 1).set_title("Repulsion Force")
            plt.plot(self.robot_traject["time"], self.robot_traject["force"])
            plt.ylabel("Force (N)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.subplot(2, 1, 2).set_title("Distance to Obstacle")
            plt.plot(self.robot_traject["time"], self.robot_traject["d_obs"])
            plt.ylabel("Distance (M)")
            plt.xlabel("Time (s)")
            plt.subplots_adjust(hspace=1.00)

            plt.show(block=True)
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
    
    def get_end_effector_velocity_2D(self, link: str = "ee") -> np.ndarray:
        [theta_vel, x_vel, y_vel] = self._robot.get_link_jacobian(link)[2:5, :] @ self._robot.get_q_dot()
        return np.array([theta_vel, x_vel, y_vel])
    
    def get_ee_projected_on_obstacle(self):
        # https://en.wikipedia.org/wiki/Vector_projection
        
        [_, x, y] = self.get_end_effector_position_2D("ee")
        p1 = (np.array(self.obstacle[0][0:2])).reshape(2, 1)
        p2 = (self.obstacle[1][0:2]).reshape(2, 1)
        p3 = (np.array([x, y])).reshape(2, 1)
        
        a = p2 - p1
        b = p3 - p1
        return p1 + a @ a.T @ b / (a.T @ a)
    
    def get_vector_ee_to_obstacle_proj(self):
        [_, x, y] = self.get_end_effector_position_2D("ee")
        p = np.array([x, y]).reshape(2, 1)
        p_proj = self.get_ee_projected_on_obstacle()
        return p_proj - p
    
    def get_ee_dist_to_obstacle(self):
        return np.linalg.norm(self.get_vector_ee_to_obstacle_proj())
    
    def get_obstacle_unit_normal(self):
        return self.get_vector_ee_to_obstacle_proj() / self.get_ee_dist_to_obstacle()
    
    def get_rotation_to_obstacle(self):
        unit_normal = self.get_obstacle_unit_normal()
        unit_tangent = np.array([[0, 1], [-1, 0]]) @ unit_normal
        return np.concatenate((unit_tangent, unit_normal), axis=1)
        