# DART-MASS

This repository contains simulation and training code for musculoskeletal models based on the **DART** (Dynamic Animation and Robotics Toolkit) physics engine.  
It supports both original training and exoskeleton-assisted training environments.

## Project Structure

- **exo training/**  
  Code for training with exoskeleton assistance.  
  Includes:
  - `EnvManager.cpp/.h` – Environment manager implementation  
  - `Environment.cpp/.h` – Simulation environment core  
  - `Model.py` – Python interface for model handling  
  - `main.py` – Training entry point  

- **original training/**  
  Baseline training environment without exoskeleton.  
  (File structure mirrors `exo training/`)

- **model/**  
  Musculoskeletal model definitions:  
  - `human.xml` – Human skeletal model  
  - `muscle284.xml` – Muscle configuration  

- **base.obj**  
  Base 3D object file for visualization.

## Requirements

- C++ compiler with CMake support  
- Python 3.9  
- DART Physics Engine  

