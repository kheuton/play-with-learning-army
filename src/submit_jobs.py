#!/usr/bin/env python
import glob
import os
import argparse
import subprocess
from pathlib import Path

def create_slurm_script(problem_config, hyper_config):
    """Create the content of the Slurm script."""
    return f"""#!/bin/bash
#SBATCH --job-name=frozen
#SBATCH -o /cluster/tufts/hugheslab/kheuto01/slurmlog/out/log_%j.out       # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/kheuto01/slurmlog/err/log_%j.err       # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --partition=batch
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --export=ALL
##SBATCH --gres=gpu:1

export PYTHONPATH=$PYTHONPATH:/cluster/home/kheuto01/code/play-with-learning-army/src

# Run the training script
python src/clear_thinking_train_loop.py --problem_config {problem_config} --hyper_config {hyper_config}
"""

def submit_job(problem_config, hyper_config):
    """Create and submit a Slurm job."""
    # Create a temporary Slurm script
    hyper_config_dir = os.path.dirname(hyper_config)
    slurm_script_path = os.path.join(hyper_config_dir, "slurm_script.sh")
    
    # Write the Slurm script
    with open(slurm_script_path, "w") as f:
        f.write(create_slurm_script(problem_config, hyper_config))
    
    # Submit the job
    try:
        result = subprocess.run(["sbatch", slurm_script_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(f"Submitted job for {hyper_config}: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job for {hyper_config}: {e.stderr}")
    
    # Clean up the temporary script
    os.remove(slurm_script_path)

def main():
    parser = argparse.ArgumentParser(description='Submit Slurm jobs for all config combinations')
    parser.add_argument('--problem_config', required=True,
                        help='Path to the problem config file')
    parser.add_argument('--base_dir', required=True,
                        help='Base directory containing the generated hyper configs')
    parser.add_argument('--dryrun', action='store_true',
                        help='If set, only report the number of jobs that would be submitted without submitting them')
    args = parser.parse_args()

    # Verify problem config exists
    if not os.path.exists(args.problem_config):
        raise FileNotFoundError(f"Problem config not found: {args.problem_config}")
    
    # Find all config files
    config_pattern = os.path.join(args.base_dir, "**/config.yaml")
    config_files = glob.glob(config_pattern, recursive=True)

    # Exclude config files if the directory also contains a best_model.pth
    filtered_config_files = []
    for config_file in config_files:
        dir_path = os.path.dirname(config_file)
        if not os.path.exists(os.path.join(dir_path, "final_model.pth")):
            filtered_config_files.append(config_file)

    config_files = filtered_config_files

    if not config_files:
        print(f"No config files found in {args.base_dir} or all directories contain best_model.pth")
        return

    print(f"Found {len(config_files)} config files")

    if args.dryrun:
        print(f"Dry run: {len(config_files)} jobs would be submitted")
        return
    
    # Submit a job for each config file
    for config_file in config_files:
        submit_job(args.problem_config, config_file)
        print(f"Submitted job for {config_file}")

if __name__ == "__main__":
    main()