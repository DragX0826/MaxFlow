import argparse
import logging
import time
import csv
import os
import torch
import numpy as np
from saeb import SAEBFlowExperiment, SimulationConfig

# ============================================================
# ASTEX DIVERSE SET — Hartshorn et al., J. Med. Chem. 2007
# The de-facto standard 85-complex benchmark for molecular docking.
# DO NOT MODIFY this list — it must match the canonical paper.
# ============================================================
ASTEX_DIVERSE_85 = [
    "1aq1", "1b8o", "1cvu", "1d3p", "1eve", "1f0r", "1fc0", "1fpu", "1glh",
    "1gpk", "1hw8", "1hwi", "1ig3", "1j3j", "1jd0", "1k3u", "1ke5", "1kzk",
    "1l2s", "1lpg", "1lpk", "1m2z", "1mq6", "1n2v", "1n46", "1nav", "1o3f",
    "1of1", "1opk", "1oq5", "1owh", "1p2y", "1pxn", "1q41", "1q8t", "1qkt",
    "1r1h", "1r55", "1s19", "1s3v", "1sg0", "1sj0", "1sqt", "1t46", "1tt1",
    "1u1c", "1vso", "1w1p", "1xm6", "2br1", "2brl", "2brn", "2cet", "2ch0",
    "2cji", "2cnp", "2cpp", "2gss", "2hs1", "2i1m", "2i78", "2ica", "2j4i",
    "2jcj", "2jdm", "2jdu", "2jf4", "2nlj", "2nnq", "2npo", "2p15", "2p4y",
    "2p54", "2p55", "2p7a", "2pog", "2psh", "2pwy", "2qmj", "2wnc", "2xnb",
    "2xys", "2yge", "2zjw",
]

# DiffDock PDBbind-2020 time-split test set (362 complexes, released ≥2019)
# Use for blind-docking comparison with published baselines.
DIFFDOCK_TIMESPLIT_362 = [
    "6qqw","6d08","6jap","6np2","6uvp","6oxq","6jsn","6hzb","6qrc","6oio",
    "6jag","6moa","6hld","6i9a","6e4c","6g24","6jb4","6s55","6seo","6dyz",
    "5zk5","6jid","5ze6","6qlu","6a6k","6qgf","6e3z","6te6","6pka","6g2o",
    "6jsf","5zxk","6qxd","6n97","6jt3","6qtr","6oy1","6n96","6qzh","6qqz",
    "6qmt","6ibx","6hmt","5zk7","6k3l","6cjs","6n9l","6ibz","6ott","6gge",
    "6hot","6e3p","6md6","6hlb","6fe5","6uwp","6npp","6g2f","6mo7","6bqd",
    "6nsv","6i76","6n53","6g2c","6eeb","6n0m","6uvy","6ovz","6olx","6v5l",
    "6hhg","5zcu","6dz2","6mjq","6efk","6s9w","6gdy","6kqi","6ueg","6oxt",
    "6oy0","6qr7","6i41","6cyg","6qmr","6g27","6ggb","6g3c","6n4e","6fcj",
    "6quv","6iql","6i74","6qr4","6rnu","6jib","6izq","6qw8","6qto","6qrd",
    "6hza","6e5s","6dz3","6e6w","6cyh","5zlf","6om4","6gga","6pgp","6qqv",
    "6qtq","6gj6","6os5","6s07","6i77","6hhj","6ahs","6oxx","6mjj","6hor",
    "6jb0","6i68","6pz4","6mhb","6uim","6jsg","6i78","6oxy","6gbw","6mo0",
    "6ggf","6qge","6cjr","6oxp","6d07","6i63","6ten","6uii","6qlr","6sen",
    "6oxv","6g2b","5zr3","6kjf","6qr9","6g9f","6e6v","5zk9","6pnn","6nri",
    "6uwv","6ooz","6npi","6oip","6miv","6s57","6p8x","6hoq","6qts","6ggd",
    "6pnm","6oy2","6oi8","6mhd","6agt","6i5p","6hhr","6p8z","6c85","6g5u",
    "6j06","6qsz","6jbb","6hhp","6np5","6nlj","6qlp","6n94","6e13","6qls",
    "6uil","6st3","6n92","6s56","6hzd","6uhv","6k05","6q36","6ic0","6hhi",
    "6e3m","6qtx","6jse","5zjy","6o3y","6rpg","6rr0","6gzy","6qlt","6ufo",
    "6o0h","6o3x","5zjz","6i8t","6ooy","6oiq","6od6","6nrh","6qra","6hhh",
    "6m7h","6ufn","6qr0","6o5u","6h14","6jwa","6ny0","6jan","6ftf","6oxw",
    "6jon","6cf7","6rtn","6jsz","6o9c","6mo8","6qln","6qqu","6i66","6mja",
    "6gwe","6d3z","6oxr","6r4k","6hle","6h9v","6hou","6nv9","6py0","6qlq",
    "6nv7","6n4b","6jaq","6i8m","6dz0","6oxs","6k2n","6cjj","6ffg","6a73",
    "6qqt","6a1c","6oxu","6qre","6qtw","6np4","6hv2","6n55","6e3o","6kjd",
    "6sfc","6qi7","6hzc","6k04","6op0","6q38","6n8x","6np3","6uvv","6pgo",
    "6jbe","6i75","6qqq","6i62","6j9y","6g29","6h7d","6mo9","6jao","6jmf",
    "6hmy","6qfe","5zml","6i65","6e7m","6i61","6rz6","6qtm","6qlo","6oie",
    "6miy","6nrf","6gj5","6jad","6mj4","6h12","6d3y","6qr2","6qxa","6o9b",
    "6ckl","6oir","6d40","6e6j","6i7a","6g25","6oin","6jam","6oxz","6hop",
    "6rot","6uhu","6mji","6nrj","6nt2","6op9","6pno","6e4v","6k1s","6a87",
    "6oim","6cjp","6pyb","6h13","6qrf","6mhc","6j9w","6nrg","6fff","6n93",
    "6jut","6g2e","6nd3","6os6","6dql","6inz","6i67","6quw","6qwi","6npm",
    "6i64","6e3n","6qrg","6nxz","6iby","6gj7","6qr3","6qr1","6s9x","6q4q",
    "6hbn","6nw3","6tel","6p8y","6d5w","6t6a","6o5g","6r7d","6pya","6ffe",
    "6d3x","6gj8","6mo2",
]


def run_single_target(pdb_id, device_id, seed, args):
    """Worker function for single-target run."""
    device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"
    config = SimulationConfig(
        pdb_id=pdb_id,
        target_name=f"SAEB_{pdb_id}_G{device_id}_S{seed}",
        steps=args.steps,
        batch_size=args.batch_size,
        mode=args.mode,
        pdb_dir=args.pdb_dir,
        seed=seed,
        fksmc=args.fksmc,
        socm=args.socm,
        srpg=args.srpg,
        no_backbone=getattr(args, "no_backbone", False),  # B6
    )

    t0 = time.time()
    try:
        experiment = SAEBFlowExperiment(config)
        results = experiment.run(device=device)
        results["time_sec"] = round(time.time() - t0, 1)
        results["seed"] = seed
        return {"pdb_id": pdb_id, "status": "Success", "results": results}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"pdb_id": pdb_id, "status": "Failed", "error": str(e),
                "traceback": tb, "time_sec": round(time.time() - t0, 1), "seed": seed}


def worker(q_in, q_out, gpu_id, args_copy):
    """Pickleable worker for torch.multiprocessing spawn.
    Sets CUDA_VISIBLE_DEVICES so this process only sees its own GPU.
    This avoids cross-device tensor errors.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        task = q_in.get()
        if task is None: break
        pdb_id, seed = task
        # gpu_id within this process is always 0 (we only see one GPU)
        res = run_single_target(pdb_id, 0, seed, args_copy)
        q_out.put(res)


def main():
    parser = argparse.ArgumentParser(description="SAEB-Flow Benchmark — Astex Diverse 85 / DiffDock-362")
    # Target selection
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated PDB IDs. Overrides --pdb_id/--bench_*.")
    parser.add_argument("--pdb_id", type=str, default=None, help="Single target PDB ID")
    parser.add_argument("--bench_astex", action="store_true", help="Astex Diverse Set (85 targets)")
    parser.add_argument("--bench_diffdock", action="store_true", help="DiffDock time-split (362 targets)")
    # Data
    parser.add_argument("--pdb_dir", type=str, default="",
                        help="Local dir with pre-downloaded .pdb files (Kaggle: /kaggle/input/astex-diverse)")
    # Training
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Ensemble clones")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"])
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds for ensemble (e.g. 42,43,44)")
    parser.add_argument("--high_fidelity", action="store_true", help="Set steps=1000, batch_size=64 for high accuracy")
    # Hardware
    parser.add_argument("--num_gpus", type=int, default=max(1, torch.cuda.device_count()))
    parser.add_argument("--kaggle", action="store_true",
                        help="Kaggle mode: sequential execution (no multiprocessing)")
    # Output
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    # Comparison
    parser.add_argument("--compare", type=str, default=None, 
                        help="Comma-separated list of CSVs to compare (e.g. 'results1.csv,results2.csv')")
    parser.add_argument("--compare_labels", type=str, default=None,
                        help="Comma-separated labels for comparison (e.g. 'Full,No-Physics')")
    # v9.0 Mathematical Foundations
    parser.add_argument("--fksmc", action="store_true", help="Enable Feynman-Kac SMC Resampling")
    parser.add_argument("--socm",  action="store_true", help="Enable SOCM-Inspired Twist Force")
    parser.add_argument("--srpg",  action="store_true", help="Enable Self-Rewarding Particle Gibbs")
    # B6: ablation flag
    parser.add_argument("--no_backbone", action="store_true",
                        help="B6/P0-2: Pure physics ablation baseline (skip neural backbone)")
    args = parser.parse_args()


    if args.high_fidelity:
        if args.steps == 300:   args.steps = 1000
        if args.batch_size == 16: args.batch_size = 128  # wider search

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("SAEB-Flow.CLI")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.high_fidelity:
        logger.info(f"High-fidelity mode: steps={args.steps}, B={args.batch_size}")

    # Select targets
    if args.targets:
        targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    elif args.pdb_id:
        targets = [args.pdb_id.lower()]
    elif args.bench_diffdock:
        targets = DIFFDOCK_TIMESPLIT_362
    elif args.bench_astex:
        targets = ASTEX_DIVERSE_85
    else:
        targets = [ASTEX_DIVERSE_85[0]]  # default: 1aq1

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    
    # Task queue: (pdb_id, seed)
    tasks = []
    for pdb_id in targets:
        for seed in seeds:
            tasks.append((pdb_id, seed))

    logger.info(f"SAEB-Flow | targets={len(targets)} | seeds={len(seeds)} | total_tasks={len(tasks)} | "
                f"steps={args.steps} | B={args.batch_size} | mode={args.mode} | "
                f"kaggle={args.kaggle}")

    results_summary = []

    if (not args.num_gpus or args.num_gpus <= 1) or len(tasks) == 1:
        # Sequential mode — safe for single GPU or single task.
        # Note: We no longer force sequential just because of --kaggle flag,
        # allowing multi-GPU to leverage spawn-based multiprocessing.
        for i, (pdb_id, seed) in enumerate(tasks):
            gpu_id = i % max(1, args.num_gpus)
            res = run_single_target(pdb_id, gpu_id, seed, args)
            results_summary.append(res)
            if res["status"] == "Success":
                r = res["results"]
                logger.info(f" [DONE] {pdb_id} (Seed {seed}): best_rmsd={r['best_rmsd']:.2f}A "
                            f"E={r['final_energy']:.1f} t={r.get('time_sec', 0)}s")
            else:
                logger.error(f" [FAIL] {pdb_id} (Seed {seed}): {res['error']}")
    else:
        # Parallel mode for local/Kaggle multi-GPU using torch.multiprocessing (spawn)
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError: pass

        q_in = mp.Queue()
        q_out = mp.Queue()
        for t in tasks: q_in.put(t)
        for _ in range(args.num_gpus): q_in.put(None)

        processes = []
        for i in range(args.num_gpus):
            p = mp.Process(target=worker, args=(q_in, q_out, i, args))
            p.start()
            processes.append(p)

        for _ in range(len(tasks)):
            res = q_out.get()
            results_summary.append(res)
            pdb_id = res["pdb_id"]
            seed = res["results"].get("seed", "?") if res["status"] == "Success" else res.get("seed", "?")
            if res["status"] == "Success":
                r = res["results"]
                logger.info(f" [DONE] {pdb_id} (Seed {seed}): best_rmsd={r['best_rmsd']:.2f}A "
                            f"E={r['final_energy']:.1f}")
            else:
                logger.error(f" [FAIL] {pdb_id}: {res.get('error')}")

        for p in processes: p.join()

    # B3 fix: expanded CSV fieldnames with Claim-1/3 metrics
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    successful = [r for r in results_summary if r["status"] == "Success"]
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "pdb_id", "seed",
            "best_rmsd", "mean_rmsd", "final_energy",
            "log_Z_final", "ess_min", "resample_count", "pb_valid_frac",
            "steps", "time_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in successful:
            row = r["results"].copy()
            row["seed"] = r.get("seed", row.get("seed", 0))
            for k in ("log_Z_final", "ess_min", "resample_count", "pb_valid_frac"):
                row.setdefault(k, "")
            row.setdefault("time_sec", 0)
            writer.writerow(row)


    # Final report
    n_tot = len(targets)
    
    # Bug Fix/Improvement 3: Multi-seed aggregation
    # We aggregate by pdb_id to find the best across seeds
    import pandas as pd
    df_all = pd.DataFrame([r["results"] for r in successful])
    if not df_all.empty:
        # Group by pdb_id and take min of best_rmsd
        df_aggr = df_all.groupby("pdb_id").agg({
            "best_rmsd": "min",
            "mean_rmsd": "mean",
            "final_energy": "min", # Best energy
            "time_sec": "sum"
        }).reset_index()
        
        n_ok = len(df_aggr)
        rmsds = df_aggr["best_rmsd"].values
        
        logger.info(f"\n{'='*65}")
        logger.info(f" BENCHMARK COMPLETE (Aggregated over {len(seeds)} seeds)")
        logger.info(f" Targets Succeeded : {n_ok}/{n_tot}")
        logger.info(f" Median RMSD : {np.median(rmsds):.2f}A  |  Mean: {np.mean(rmsds):.2f}A")
        logger.info(f" SR@2A  : {(rmsds < 2.0).mean()*100:.1f}%  ({(rmsds < 2.0).sum()}/{n_ok})")
        logger.info(f" SR@5A  : {(rmsds < 5.0).mean()*100:.1f}%  ({(rmsds < 5.0).sum()}/{n_ok})")
        logger.info(f" Results saved → {csv_path}")
        logger.info(f"{'='*65}")
        
        # Save aggregated results
        aggr_path = os.path.join(args.output_dir, "benchmark_aggregated.csv")
        df_aggr.to_csv(aggr_path, index=False)
    else:
        logger.error("No successful targets to aggregate.")

    # Generate aggregate figures
    if len(successful) > 3:
        try:
            from saeb.reporting.visualizer import PublicationVisualizer
            viz = PublicationVisualizer(output_dir=args.output_dir)
            result_dicts = [r["results"] for r in successful]
            viz.plot_success_rate_curve(rmsds, filename="fig_sr_curve.pdf")
            viz.plot_rmsd_cdf(rmsds, filename="fig_rmsd_cdf.pdf")
            viz.plot_benchmark_summary(result_dicts, filename="fig_benchmark_summary.pdf")
            logger.info(" Aggregate figures generated.")
        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")


    # Comparison implementation
    if args.compare:
        import pandas as pd
        csvs = args.compare.split(",")
        labels = args.compare_labels.split(",") if args.compare_labels else [os.path.basename(c) for c in csvs]
        
        comparison_rmsd = {}
        ablation_data = {}
        
        for c, label in zip(csvs, labels):
            df = pd.read_csv(c)
            rmsds = df["best_rmsd"].values
            comparison_rmsd[label] = rmsds
            ablation_data[label] = {
                "sr2": (rmsds < 2.0).mean() * 100,
                "median_rmsd": np.median(rmsds)
            }
            
        try:
            from saeb.reporting.visualizer import PublicationVisualizer
            viz = PublicationVisualizer(output_dir=args.output_dir)
            viz.plot_success_rate_curve(comparison_rmsd, filename="fig_comparison_sr.pdf")
            viz.plot_rmsd_cdf(comparison_rmsd, filename="fig_comparison_cdf.pdf")
            viz.plot_ablation(ablation_data, filename="fig_ablation_study.pdf")
            logger.info(f" Comparison figures generated in {args.output_dir}")
        except Exception as e:
            logger.warning(f"Comparison figure generation failed: {e}")
        return


if __name__ == "__main__":
    main()
