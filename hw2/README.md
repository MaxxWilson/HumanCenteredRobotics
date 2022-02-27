# ASE389
Software Stacks for the class ASE 389 DCSN/CTRL HUMAN-CENT ROBOTS based on
the [PnC](https://github.com/junhyeokahn/PnC).


## Installation
- Install [Conda](https://docs.anaconda.com/anaconda/install/)
- Clone the repository
```
$ git clone https://github.com/junhyeokahn/ASE389.git
```
- Create a virtual environment and install Dependancies
```
$ conda env create -f ASE389.yml
```
- Activate the environment
```
$ conda activate ASE389
```
- Run the code
```
$ python simulator/pybullet/manipulator_main.py
```

## Update to the newest code
- Pull the latest code
```
$ git pull origin main
```
- Update conda environment
```
$ conda env update -f ASE389.yml
```

## Implement Operational Space Control
- Try to implement OSC control and fill out this [function](https://github.com/junhyeokahn/ASE389/blob/c59dcef47506383b7f99d1a4ece727023dd183ca/pnc/manipulator_pnc/manipulator_interface.py#L57).
- You could take look at [RobotSystem](https://github.com/junhyeokahn/ASE389/blob/main/pnc/robot_system/robot_system.py) to achieve relevant kinematics and dynamics properties of the robot.

