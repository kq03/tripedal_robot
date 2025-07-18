#!/usr/bin/env python3
"""
Workspace setup script for tripedal robot project.

This script sets up the Python path to include both the tripedal project
and the IsaacLab repository in the workspace structure:

/tripedal_ws/
├── src/
│   ├── tripedal/          # Your tripedal robot project
│   └── IsaacLab/          # Official Isaac Lab repository
"""

import os
import sys
from pathlib import Path

def setup_workspace_paths():
    """Set up Python paths for the workspace."""
    
    # Get the current directory (should be /tripedal_ws/src/tripedal)
    current_dir = Path(__file__).parent.absolute()
    
    # Check if we're in the expected workspace structure
    if current_dir.name != "tripedal":
        print(f"Warning: Expected to be in 'tripedal' directory, but found '{current_dir.name}'")
        print("Make sure this script is run from /tripedal_ws/src/tripedal/")
    
    # Add the tripedal source directory to Python path
    tripedal_source = current_dir / "source"
    if tripedal_source.exists():
        sys.path.insert(0, str(tripedal_source))
        print(f"Added tripedal source to Python path: {tripedal_source}")
    else:
        print(f"Warning: tripedal source directory not found at {tripedal_source}")
    
    # Add the IsaacLab source directory to Python path
    isaaclab_source = current_dir.parent / "IsaacLab" / "source"
    if isaaclab_source.exists():
        sys.path.insert(0, str(isaaclab_source))
        print(f"Added IsaacLab source to Python path: {isaaclab_source}")
    else:
        print(f"Warning: IsaacLab source directory not found at {isaaclab_source}")
        print("Make sure IsaacLab is cloned to /tripedal_ws/src/IsaacLab/")
    
    # Add the current directory to Python path for scripts
    sys.path.insert(0, str(current_dir))
    print(f"Added current directory to Python path: {current_dir}")
    
    return True

def check_workspace_structure():
    """Check if the workspace structure is correct."""
    current_dir = Path(__file__).parent.absolute()
    
    print("Checking workspace structure...")
    print(f"Current directory: {current_dir}")
    
    # Check tripedal project structure
    tripedal_source = current_dir / "source"
    if tripedal_source.exists():
        print(f"✓ tripedal source found: {tripedal_source}")
    else:
        print(f"✗ tripedal source missing: {tripedal_source}")
    
    # Check IsaacLab repository
    isaaclab_dir = current_dir.parent / "IsaacLab"
    if isaaclab_dir.exists():
        print(f"✓ IsaacLab repository found: {isaaclab_dir}")
        
        isaaclab_source = isaaclab_dir / "source"
        if isaaclab_source.exists():
            print(f"✓ IsaacLab source found: {isaaclab_source}")
        else:
            print(f"✗ IsaacLab source missing: {isaaclab_source}")
    else:
        print(f"✗ IsaacLab repository missing: {isaaclab_dir}")
        print("Please clone IsaacLab to /tripedal_ws/src/IsaacLab/")
    
    # Check for key files
    key_files = [
        "source/tripedal_robot_assets/tripedal_robot_assets/tlr7_config.py",
        "source/tripedal_robot_tasks/direct/tripedal_robot/tripple_legs_robot_env.py",
        "scripts/reinforcement_learning/rsl_rl/train.py",
    ]
    
    print("\nChecking key files:")
    for file_path in key_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")

def main():
    """Main function to set up the workspace."""
    print("=== Tripedal Robot Workspace Setup ===")
    
    # Check workspace structure
    check_workspace_structure()
    
    # Set up Python paths
    print("\nSetting up Python paths...")
    setup_workspace_paths()
    
    print("\nWorkspace setup complete!")
    print("\nYou can now import your tripedal modules and IsaacLab modules.")
    print("Example:")
    print("  from isaaclab_assets.robots.tripple_legs_robot import TLR6_CFG")
    print("  from isaaclab_tasks.direct.triiple_legs_robot import TLR6Env")

if __name__ == "__main__":
    main() 