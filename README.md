# MPC-Reachability

This repository employs neural network reachability analysis to reason about the safety of an autonomous vehicle with an MPC controller. The vehicle aims to undertake five different maneuvers driven by the MPC controller while neural network reachability checks for intersections with obstacles to reason about safety

Maneuvers:

1. Driving in a straight lane close to a curb,
2. Taking a left turn
3. Taking a right turn
4. Making a U-turn
5. Changing lanes due to a roadblock

## Install

We checked this repo. with Ubuntu 20.04, Mac Big Sur Version 11.4, Windows 10.

```bash
git clone https://github.com/Abdu-Hekal/mpc-reachability.git
cd mpc-reachability
pip install -r requirements.txt
```

## How to use

From terminal, run

```bash
python mpc.py <MANUEVER_NUMNER>
```

 to visualise MPC with Reachability for a specified benchmark from 1 to 5.

## Example

```bash
python mpc.py 2
```
runs MPC with reachability for the second Maneuver: Taking a left turn

