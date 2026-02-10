"""
Setup script to initialize the Almaty Air Quality Forecasting & Analysis project.

Run this script once to create all necessary directories.
"""

from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
    SRC_DIR,
    NOTEBOOKS_DIR,
)
from src.utils import safe_mkdir


def create_project_structure() -> None:
    """
    Create all necessary directories for the project.
    """
    print("=" * 70)
    print("Setting up Almaty Air Quality Forecasting & Analysis")
    print("=" * 70)
    
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
        SRC_DIR,
        NOTEBOOKS_DIR,
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        safe_mkdir(directory)
        print(f"  ✓ Created: {directory}")
    
    # Create .gitkeep files to track empty directories
    print("\nCreating .gitkeep files...")
    for directory in directories:
        gitkeep_file = directory / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"  ✓ Created: {gitkeep_file}")
    
    print("\n" + "=" * 70)
    print("Project structure created successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review src/config.py for configuration settings")
    print("  2. Install dependencies: pip install -r requirements.txt")
    print("  3. Follow notebooks in order (see README.md)")


if __name__ == "__main__":
    create_project_structure()