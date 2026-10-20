"""
deploy - the pieces a phase-2 project needs to run a trained policy on a robot.

    numpy_policy.py     the trained network as plain numpy (no torch / SB3 on the robot)
    observation.py      builds the policy observation from a robot state dict
    robot_interface.py  RobotInterface + SimRobot (MuJoCo) + Go2Robot template (Unitree SDK2)
"""
