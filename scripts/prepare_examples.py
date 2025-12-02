#!/usr/bin/env python3
"""
Script to prepare validation examples for the web interface.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from web.backend.example_manager import ExampleManager
from web.backend.config import ExampleConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    valid_txt_path = project_root / "dataset" / "valid.txt"
    target_dir = project_root / "dataset" / "validation_examples"
    num_examples = 12  # Number of examples to prepare
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Target directory: {target_dir}")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read validation list
    if not valid_txt_path.exists():
        logger.error(f"Validation list not found: {valid_txt_path}")
        return
        
    with open(valid_txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(lines)} files in validation list")
    
    # Copy files
    count = 0
    for file_path in lines:
        if count >= num_examples:
            break
            
        src_path = Path(file_path)
        if not src_path.exists():
            logger.warning(f"File not found: {src_path}")
            continue
            
        # Create a nice name
        # If path is .../POP909/687/687.mid, name it pop909_687.mid
        # Or just use the filename
        
        # Try to extract dataset name if possible
        name_parts = src_path.parts
        if "POP909" in name_parts:
            idx = name_parts.index("POP909")
            if idx + 2 < len(name_parts):
                 # e.g. POP909/687/687.mid -> pop909_687.mid
                 dst_name = f"pop909_{name_parts[-1]}"
            else:
                dst_name = src_path.name
        else:
            dst_name = src_path.name
            
        dst_path = target_dir / dst_name
        
        try:
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {src_path} to {dst_path}")
            count += 1
        except Exception as e:
            logger.error(f"Error copying {src_path}: {e}")
            
    logger.info(f"Copied {count} examples")
    
    # Generate metadata and thumbnails using ExampleManager
    logger.info("Generating metadata and thumbnails...")
    
    # We pass the absolute path to ensure it works correctly
    manager = ExampleManager(examples_dir=target_dir, max_examples=num_examples)
    
    # Force recreation to ensure thumbnails are generated
    manager._create_metadata()
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

