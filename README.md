# MPC-Reachability

This repository employs neural network reachability analysis to reason about the safety of an autonomous vehicle with an MPC controller. The vehicle aims to undertake five different maneuvers driven by the MPC controller while neural network reachability checks for intersections with obstacles to reason about safety 

Maneuvers:

1. Driving in a straight lane close to a curb,
2. Taking a left turn
3. Taking a right turn
4. Making a U-turn
5. Changing lanes due to a roadblock


## How to use

From terminal, run "python mpc.py {Maneuver number}" to visualise MPC with Reachability for a specified benchmark from 1 to 5.

## Example

"python mpc.py 2" runs MPC with reachability for the second Maneuver: Taking a left turn

