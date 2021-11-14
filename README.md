# GoAI
<p align="center">
  <img src="https://github.com/brent-buffenbarger/GoAI/blob/master/docs/go.png?raw=true" />
</p>

## Overview
For this project I created an AI agent that plays Mini-Go. Mini-Go is a 5x5 version of the famous board game Go, which is played on a 19x19 board. This project was created for a homework assignment. The grading was based on the agent's performance against 6 other pre-made agents. The opponent agents consisted of a random agent, a greedy agent, an aggressive agent, an agent that also used Alpha-beta Pruning, an agent that was trained using Q-learing, and a championship agent that used an unknown strategy.

## Techniques Used
The agent uses the Minimax algorithm with a depth of 6. To optimize the Minimax algorithm, the agent also uses Alpha-beta Pruning to avoid checking unecessary branches of the Minimax tree. A refined heuristic is being used to determine the best future states.
<p align="center">
  <img src="https://github.com/brent-buffenbarger/GoAI/blob/master/docs/ab_pruning.png?raw=true" />
</p>

## Results
Against the 6 opponents, my agent was able to beat everybody except the championship agent. The agents overall win rate was 95%.

## Usage
If you want to clone the repository and play around with the agent, you can use the `build.sh` script to put the agent against a random player. Unfortunately, the more difficult opponents are not provided. The `random_player.py` file can be modified to make the opponent more difficult. Running the `build.sh` script will open up a terminal that serves as a GUI for the game. You can watch the agent play through 2 games (1 as black and 1 as white) and then view the overall results of the games.

<p align="center">
  <img src="https://github.com/brent-buffenbarger/GoAI/blob/master/docs/gameplay.PNG?raw=true" />
</p>

## Repository Contents
`host.py`: This is the driver file that provides the input file to the agents, reads the output file from the agents, verifies that game rules are being followed, and decides when a game has finished and who won the game.

`read.py` and `write.py`: These are just helper files to make it easier for the `host` to write input files and read output files.

`alpha_beta_agent.py`: This is the agents that I created. This file can be modified to try and improve the agent's performance.

`random_player.py`: This is the opponent agent that can be used for testing.

`build.sh`: This is the shell script used to run the game.

`docs`: This folder contains the images seen in this `README.md`.

`init`: This folder holds the initial input file for the `host` to use. This can be modified to change the starting state of the game.
