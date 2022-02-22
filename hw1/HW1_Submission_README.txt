Hello.

Enclosed is HW1 for ASE 389 Human-Centered Robotics.

HW1_Submission.pdf is a write-up with equations, explanations, and plots for each problem (including optional question 5).

File can be run from the hw1 folder with:
    python simulator/pybullet/manipulator_main.py

Each controller can be activated by uncommenting the following lines (and commenting out all others) in hw1/pnc/manipulator_pnc/manipulator_interface.py:
    Joint Space Control - Line 77
    Operational Space Control - Line 80
    Whole Body Control - Line 83
    Obstacle Avoidance with Repulsion Field - Line 87 (and line 58 for obstacle graphic)

After completion of each trajectory, plots will spawn showing performance. After the plots are closed, the program will automatically exit.

I had trouble installing matplotlib with anaconda and so I used "pip install matplotlib" which worked fine.
If there are any issues with matplotlib in the conda environment, I have added a variable on line 17 which will disable plotting and the matplotlib import.
On my computer, all controllers run without error with and without plotting at the time of submission.

Let me know if there are any troubles.

Thanks!
Maxx Wilson
JMW5966