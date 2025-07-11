# Tripedal Robot Workspace

This workspace is set up to work with both your custom tripedal robot project and the official Isaac Lab repository.

## Workspace Structure

```
/tripedal_ws/
├── src/
│   ├── tripedal/          # Your tripedal robot project
│   │   ├── source/        # Custom Python modules
│   │   ├── scripts/       # Training and utility scripts
│   │   ├── TLR7_0629new/  # Robot URDFs and configs
│   │   ├── tlr_control/   # Control and deployment scripts
│   │   └── setup_workspace.py  # Workspace setup script
│   └── IsaacLab/          # Official Isaac Lab repository (cloned separately)
│       ├── source/        # Isaac Lab Python modules
│       └── ...
```

## Setup Instructions

### 1. Create the Workspace Structure

```bash
# Create the workspace directory
mkdir -p /tripedal_ws/src
cd /tripedal_ws/src

# Clone Isaac Lab (if not already done)
git clone https://github.com/isaac-sim/IsaacLab.git

# Copy your tripedal project
cp -r /path/to/your/tripedal /tripedal_ws/src/
```

### 2. Set Up Python Paths

Run the workspace setup script to configure Python paths:

```bash
cd /tripedal_ws/src/tripedal
python setup_workspace.py
```

This script will:
- Add the tripedal source directory to Python path
- Add the IsaacLab source directory to Python path
- Check that all required files are present

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Isaac Sim (follow official guide)
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html
```

## Usage

### Importing Modules

With the workspace set up, you can import both your custom modules and Isaac Lab modules:

```python
# Import your custom robot configuration
from isaaclab_assets.robots.tripple_legs_robot import TLR6_CFG

# Import your custom environment
from isaaclab_tasks.direct.triiple_legs_robot import TLR6Env

# Import Isaac Lab modules
from isaaclab.app import AppLauncher
from isaaclab.envs import DirectRLEnv
```

### Running Training

```bash
# Run RL training
python scripts/reinforcement_learning/rsl_rl/train.py --task=triiple_legs_robot

# Run with joint data logging
python scripts/reinforcement_learning/rsl_rl/train.py --task=triiple_legs_robot --save_joint_data
```

### Using Robot Control

```bash
# Run robot control scripts
cd tlr_control
python src/main.py
```

## Import Strategy

The project uses a "try-except" import strategy:

1. **First Priority**: Try to import from IsaacLab workspace
2. **Fallback**: If IsaacLab is not available, use dummy classes for syntax checking
3. **Runtime**: When Isaac Sim is available, all imports work normally

This allows:
- Syntax checking without Isaac Sim installed
- Development and testing without full Isaac Sim setup
- Full functionality when Isaac Sim is available

## Key Files

### Tripedal Project Files
- `source/isaaclab_assets/isaaclab_assets/robots/tripple_legs_robot.py` - Robot configuration
- `source/isaaclab_tasks/isaaclab_tasks/direct/triiple_legs_robot/` - RL environment
- `scripts/reinforcement_learning/rsl_rl/train.py` - Training script
- `tlr_control/` - Robot control and deployment

### IsaacLab Files (External)
- All Isaac Lab modules and environments
- Isaac Sim integration
- Standard robot configurations

## Development Workflow

1. **Development**: Work on your custom code in `/tripedal_ws/src/tripedal/`
2. **Testing**: Use the workspace setup to test imports and syntax
3. **Training**: Run training scripts with Isaac Sim
4. **Deployment**: Use control scripts for real robot deployment

## Troubleshooting

### Import Errors
- Make sure IsaacLab is cloned to `/tripedal_ws/src/IsaacLab/`
- Run `python setup_workspace.py` to check paths
- Verify Isaac Sim is installed and accessible

### Path Issues
- Check that you're in the correct directory: `/tripedal_ws/src/tripedal/`
- Ensure all required files are present
- Verify Python path includes both source directories

### Isaac Sim Issues
- Follow the official Isaac Sim installation guide
- Make sure Isaac Sim Python API is available
- Check that Isaac Lab is compatible with your Isaac Sim version

## Benefits of This Structure

1. **Clean Separation**: Your code is separate from Isaac Lab
2. **Easy Updates**: Update Isaac Lab independently
3. **Version Control**: Manage your project separately
4. **Development**: Work on your code without Isaac Sim
5. **Deployment**: Easy to package and distribute your project 