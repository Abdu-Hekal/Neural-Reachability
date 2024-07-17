# Neural-Reachability

## Overview

This repository employs neural reachability to reason about the safety of an autonomous vehicle with an MPC controller. The vehicle aims to undertake five different maneuvers driven by the MPC controller while neural network reachability checks for intersections with obstacles to ensure safety. To see neural reachability employed as an online monitoring system in a simplex architecture, see the repository [here](https://github.com/Abdu-Hekal/Neural-Reachability-Simplex).

## Table of Contents

- [Introduction](#introduction)
- [Maneuvers](#maneuvers)
- [Compatability](#compatability)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Installation Issues](#installation-issues)

## Introduction

Neural-Reachability is a framework designed to enhance the safety of autonomous vehicles using Model Predictive Control (MPC) and neural network-based reachability analysis. The system evaluates the potential paths of the vehicle to detect any possible intersections with obstacles, thereby ensuring safe navigation through various maneuvers.

## Maneuvers

1. Driving in a straight lane close to a curb.
2. Taking a left turn.
3. Taking a right turn.
4. Making a U-turn.
5. Changing lanes due to a roadblock.

## Compatability

We checked this repository with the following operating systems:

- Ubuntu 20.04
- Mac Big Sur Version 11.4
- Windows 10

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Abdu-Hekal/Neural-Reachability.git
cd mpc-reachability
pip install -r requirements.txt
```


## Usage

From the terminal, run:

```bash
python mpc.py <MANEUVER_NUMBER>
```

 to visualise MPC with Reachability for a specified benchmark from 1 to 5.

## Example

```bash
python mpc.py 2
```

runs MPC with reachability for the second Maneuver: Taking a left turn

## Installation Issues

When installing `pycddlib`, a common issue arises when the installation process fails to locate the `gmp.h` header file. This file is essential as it contains declarations necessary for using the GNU Multiple Precision Arithmetic Library (GMP). 

To resolve this issue, you can set environment variables that point the compiler (`clang` or `gcc`) to the directory where `gmp.h` is located. Here's how you can do it:

1. **Check GMP Installation:**
   First, ensure that GMP is installed on your system. You can install it using package managers like Homebrew on macOS or `apt` on Ubuntu/Debian:
   
   **On macOS:**
   ```bash
   brew install gmp
   ```

   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgmp-dev
   ```

2. **Set Environment Variables:**
   Once GMP is installed, export the paths to the GMP header files (gmp.h) and libraries (libgmp.a or libgmp.so). This tells the compiler where to find these files during the build   process:
   
   **On macOS:**
   ```bash
   export C_INCLUDE_PATH=$(brew --prefix gmp)/include
   export LIBRARY_PATH=$(brew --prefix gmp)/lib
   ```

   **On Ubuntu/Debian:**
   ```bash
   export C_INCLUDE_PATH=/usr/include/
   export LIBRARY_PATH=/usr/lib/
   ```

4. **Reinstall pycddlib:**
   After setting the environment variables, attempt to install pycddlib again using pip:
   ```bash
   pip install pycddlib
   ```

   


   
    

