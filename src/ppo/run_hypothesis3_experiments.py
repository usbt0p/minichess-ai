import os
import sys
import argparse
import subprocess

def run_dry_run(results_base: str, supervised_checkpoint: str, variant_filter: str):
    """
    Perform a quick sanity check of the training pipeline using small dimensions,
    few environments, and a low number of PPO iterations.
    """
    print("========================================")
    print("=== DRY RUN MODE: Quick sanity check ===")
    print("========================================")
    
    # Only run 1 seed, 2 iterations, small dimensions
    dry_seeds = [42]
    dry_iterations = 2
    dry_num_envs = 8
    dry_rollout_steps = 16
    dry_batch_size = 64
    dry_epochs = 1
    dry_workers = 2
    dry_eval_games = 2
    
    variants = ["tabula_rasa", "pretrained"] if variant_filter == "all" else [variant_filter]
    
    for variant in variants:
        for seed in dry_seeds:
            print(f"\n>>> Running DRY-RUN for variant={variant}, seed={seed} <<<")
            
            run_dir = os.path.join(results_base, f"{variant}_seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            
            log_file = os.path.join(run_dir, "train.log")
            save_dir = os.path.join(run_dir, "save")
            pool_dir = os.path.join(run_dir, "pool")
            tb_dir = os.path.join(run_dir, "tb")
            
            cmd = [
                ".venv/bin/python3",
                "src/ppo/train_ppo.py",
                "--iterations", str(dry_iterations),
                "--log_file", log_file,
                "--save_dir", save_dir,
                "--pool_dir", pool_dir,
                "--tb_dir", tb_dir,
                "--seed", str(seed),
                "--opponent_mode", "self_play_pool",
                "--eval_interval", "1",
                "--checkpoint_save_interval", "1",
                "--num_workers", str(dry_workers),
                "--num_envs", str(dry_num_envs),
                "--rollout_steps", str(dry_rollout_steps),
                "--batch_size", str(dry_batch_size),
                "--epochs", str(dry_epochs),
                "--eval_games", str(dry_eval_games)
            ]
            
            if variant == "pretrained":
                cmd.extend(["--checkpoint", supervised_checkpoint])
                
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            
            print(f"Executing: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
            proc.communicate()
            if proc.returncode != 0:
                print(f"[ERROR] Run failed for seed {seed} variant {variant}")
                sys.exit(proc.returncode)
            print(f"[SUCCESS] Finished dry-run for variant={variant}, seed={seed}")


def run_experiments(results_base: str, supervised_checkpoint: str, variant_filter: str, seeds: list[int], iterations: int):
    """
    Run standard Hypothesis 3 experiments for the selected variant across all specified seeds.
    """
    print("========================================")
    print("=== EXPERIMENT RUNNER: PPO Hypothesis 3 ===")
    print("========================================")
    
    variants = ["tabula_rasa", "pretrained"] if variant_filter == "all" else [variant_filter]
    
    for variant in variants:
        for seed in seeds:
            print(f"\n>>> Starting Experiment Run for variant={variant}, seed={seed} <<<")
            
            run_dir = os.path.join(results_base, f"{variant}_seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            
            log_file = os.path.join(run_dir, "train.log")
            save_dir = os.path.join(run_dir, "save")
            pool_dir = os.path.join(run_dir, "pool")
            tb_dir = os.path.join(run_dir, "tb")
            
            cmd = [
                ".venv/bin/python3",
                "src/ppo/train_ppo.py",
                "--iterations", str(iterations),
                "--log_file", log_file,
                "--save_dir", save_dir,
                "--pool_dir", pool_dir,
                "--tb_dir", tb_dir,
                "--seed", str(seed),
                "--opponent_mode", "self_play_pool",
                "--eval_interval", "5",
                "--checkpoint_save_interval", "5",
                "--num_workers", "12"
            ]
            
            if variant == "pretrained":
                cmd.extend(["--checkpoint", supervised_checkpoint])
                
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            
            print(f"Executing: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
            proc.communicate()
            if proc.returncode != 0:
                print(f"[ERROR] Run failed for seed {seed} variant {variant}")
                sys.exit(proc.returncode)
            print(f"[SUCCESS] Finished experiment run for variant={variant}, seed={seed}")


def main():
    parser = argparse.ArgumentParser(description="Run Gardner Minichess PPO Hypothesis 3 experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 1-iteration dry-run of a single seed to verify the pipeline")
    parser.add_argument("--variant", type=str, choices=["all", "tabula_rasa", "pretrained"], default="all",
                        help="Choose which experiment variant to run (default: 'all' runs both tabula rasa and pretrained)")
    parser.add_argument("--seeds", type=str, default="42,100,2026,999,12345", help="Comma-separated list of random seeds")
    parser.add_argument("--iterations", type=int, default=60, help="Number of PPO training iterations per run")
    parser.add_argument("--supervised_checkpoint", type=str, 
                        default="results/ablations/ablations_dk64_n3/h2_ablation_bigbatch_seed_1/20260620_170835_d4_val_h2_abl_seed_1_spatial_nofact_dk64_depth3_lr2.84e-03_bs16384/best_model.pth",
                        help="Path to winning supervised model checkpoint")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Base directories
    results_base = "results/hypothesis3"
    os.makedirs(results_base, exist_ok=True)

    if args.dry_run:
        run_dry_run(results_base, args.supervised_checkpoint, args.variant)
    else:
        run_experiments(results_base, args.supervised_checkpoint, args.variant, seeds, args.iterations)

if __name__ == "__main__":
    main()
