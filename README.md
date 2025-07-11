# Tripedal Robot RL Project

This project provides reinforcement learning (RL) and control tools for a custom tripedal robot, leveraging NVIDIA Isaac Sim for high-fidelity simulation. It is a standalone repository containing only the code and assets specific to the tripedal robot, and **depends on Isaac Sim** as an external simulator and Python API.

---

## Features
- RL training scripts and environments for a tripedal robot
- Custom robot URDFs, meshes, and configuration
- Control and deployment scripts for real and simulated robots
- IMU data processing and estimation tools
- Utilities for model conversion and deployment

---

## Project Structure
- `TLR7_0629new/`, `TLR7_0629存档/`: Robot description, URDFs, meshes, configs
- `tlr_control/`: Control, CAN, and deployment scripts for the robot
- `scripts/`: RL training, evaluation, and utility scripts
- `source/`: Custom Python modules for robot, environment, and RL integration
- `convert_pt_to_onnx.py`, `convert_simple.py`: Model conversion utilities
- `IMU_线速度估算方法.*`: IMU velocity estimation documentation and code

---

## Isaac Sim Dependency
This project **requires NVIDIA Isaac Sim**. Please follow the official [Isaac Sim installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install it.

After installing Isaac Sim, ensure its Python API is available in your environment. You may need to launch Isaac Sim's Python environment or set up your own virtual environment with the Isaac Sim Python bindings.

---

## Installation
1. **Install Isaac Sim** (see above).
2. Clone this repository:
   ```sh
   git clone <your-repo-url>
   cd new_tripedal
   ```
3. (Optional) Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   *(Create `requirements.txt` as needed for your custom code.)*

---

## Usage
- **RL Training:**
  - See `scripts/reinforcement_learning/rsl_rl/train.py` and related scripts for training workflows.
- **Robot Control:**
  - Use scripts in `tlr_control/` for CAN communication, deployment, and hardware interfacing.
- **IMU Processing:**
  - See `IMU_线速度估算方法.py` and `.tex` for IMU velocity estimation methods.
- **Model Conversion:**
  - Use `convert_pt_to_onnx.py` and `convert_simple.py` for model export and conversion.

---

## Customization
- Add your own robot models, configs, and RL environments under the respective folders.
- Update Python modules in `source/` for new tasks, robots, or controllers.

---

## License
This project is released under the BSD-3 License. See `LICENSE` for details.

---

## Acknowledgement
This project was originally based on Isaac Sim and Isaac Lab, but is now a standalone repository for the tripedal robot. Please cite Isaac Sim and Isaac Lab if you use this work in academic publications.
