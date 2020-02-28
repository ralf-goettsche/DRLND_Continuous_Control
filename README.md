# Deep Reinforcement Learning Nanodegree
# *DRLND-Continuous-Control*

## Introduction

In this project, an agent learns to control a double-jointed arm to follow the target locations given by a light green ball volume (Unity's Reacher environment).

<p align="center">
<img src="Images/DRLND_Continuous_Control.png" height=300 />
</p>

A reward of +0.1 is provided for each step that the agent's hand is in the target location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation (state) space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a float number between -1 and 1.

The task is episodic, and in order to solve the environment, an agent must get an average score of +30 over 100 consecutive episodes.


### Getting Started

Setup the dependencies as described [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md).

Download the environment from one of the links below.
You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

For Windows users, check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
If you'd like to train the agent on AWS and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

Clone the repository and unpack the environment file in the project folder.

### Instructions

Start `Continuous_Control.ipynb` in jupyter notebook:
```
jupyter notebook Continuous_Control.ipynb &
```
Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

