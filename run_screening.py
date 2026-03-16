# run_screening.py  —  background screening script
# Usage:   nohup python run_screening.py > data/output/screening.log 2>&1 &
# Monitor: tail -f data/output/screening.log

import os, time, json, pickle, ray
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

from utils import (
    load, save,
    disease_to_filename as _disease_to_filename,
    simulate_parallel_runtime,
    fit_empirical_model,
)


def disease_to_filename(name):
    return _disease_to_filename(name, CHECKPOINT_DIR)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

BENCH_PATH     = "data/output/benchmark.json"
PPI_PATH       = "data/processed/ppi/ppi_network.pkl"
DIST_PATH      = "data/output/sp_distance.pkl"
DISEASE_PATH   = "data/processed/gene_modules/gene_modules.pkl"
RESULTS_CACHE  = "data/output/disease_separation_results.pkl"
CHECKPOINT_DIR = "data/output/checkpoints"
RUN_LOG_PATH   = "data/output/run_timelog.json"
QUERY_KEY      = "spaceflight"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Load benchmark config ─────────────────────────────────────────────────────

DEFAULT_WORKERS = max(1, int(os.cpu_count() * 0.5))
DEFAULT_ITER    = 1000

if os.path.exists(BENCH_PATH):
    with open(BENCH_PATH) as f:
        cfg = json.load(f)
    N_WORKERS      = cfg["N_WORKERS"]
    N_ITER         = cfg["N_ITER"]
    runtime_alpha  = cfg.get("runtime_alpha")
    runtime_beta   = cfg.get("runtime_beta")
    runtime_alpha2 = cfg.get("runtime_alpha2", None)
    runtime_model  = cfg.get("runtime_model", "linear")
    log("Config loaded from benchmark.json")
else:
    log("WARNING: benchmark.json not found — using safe defaults.")
    N_WORKERS, N_ITER = DEFAULT_WORKERS, DEFAULT_ITER
    runtime_alpha = runtime_beta = runtime_alpha2 = None
    runtime_model = "linear"

log(f"N_WORKERS={N_WORKERS}  N_ITER={N_ITER}  model={runtime_model}")

# ── Load data ─────────────────────────────────────────────────────────────────

log("Loading interactome...")
ppi = load(PPI_PATH)
log("Loading disease gene sets...")
disease_genes_raw = load(DISEASE_PATH)
ppi_nodes = set(ppi.nodes())
disease_genes = {
    d: set(g) & ppi_nodes
    for d, g in disease_genes_raw.items()
    if len(set(g) & ppi_nodes) >= 20
}

if QUERY_KEY not in disease_genes:
    raise KeyError(f"{QUERY_KEY} not found.")

query_genes     = disease_genes[QUERY_KEY]
target_diseases = {d: g for d, g in disease_genes.items() if d != QUERY_KEY}
log(f"Query: {QUERY_KEY} ({len(query_genes)} genes) | Targets: {len(target_diseases)}")

# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_runtime_benchmark(disease_name, empirical_log=None):
    if runtime_alpha is None:
        return 1
    size = len(target_diseases[disease_name])
    t100 = (runtime_alpha2 * size**2 + runtime_alpha * size + runtime_beta) if runtime_alpha2            else (runtime_alpha * size + runtime_beta)
    pred = max(t100 * (N_ITER / 100), 1)
    if empirical_log and len(empirical_log) >= 5:
        floor = min(t for _, t in empirical_log) * 0.80
        return max(pred, floor)
    return pred

# fit_empirical_model imported from utils

# ── Run log helpers ───────────────────────────────────────────────────────────

def load_run_log():
    if os.path.exists(RUN_LOG_PATH):
        with open(RUN_LOG_PATH) as f:
            return json.load(f)
    return {"runs": []}

def save_run_log(data):
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ── Resume logic ──────────────────────────────────────────────────────────────

done = {d for d in target_diseases if os.path.exists(disease_to_filename(d))}
todo = [d for d in target_diseases if d not in done]
log(f"Completed: {len(done)} / {len(target_diseases)} | Remaining: {len(todo)}")

if not todo:
    log("All diseases complete — proceeding to merge.")
else:
    todo.sort(key=lambda d: predict_runtime_benchmark(d), reverse=True)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=N_WORKERS, ignore_reinit_error=True,
             log_to_driver=False, logging_level="WARNING")
    log("Ray initialized.")

    @ray.remote
    def ray_separation(disease_name, disease_genes_set,
                       source_genes, ppi_path, dist_path, n_iter):
        import pickle, time
        from netmedpy import separation_z_score
        with open(ppi_path,  "rb") as f: ppi_w  = pickle.load(f)
        with open(dist_path, "rb") as f: dist_w = pickle.load(f)
        t0     = time.time()
        result = separation_z_score(ppi_w, source_genes, disease_genes_set,
                                    dist_w, null_model="log_binning", n_iter=n_iter)
        return disease_name, result, time.time() - t0

    remaining, running = todo.copy(), {}
    completed_count    = len(done)
    empirical_log      = []
    t_wall             = time.time()

    run_log   = load_run_log()
    run_entry = {
        "run_start": time.strftime("%Y-%m-%d %H:%M:%S"), "run_end": None,
        "n_iter": N_ITER, "n_workers": N_WORKERS,
        "diseases_total": len(target_diseases), "diseases_start": len(done),
        "diseases_end": None, "wall_seconds": None, "wall_human": None,
        "per_disease": [],
    }

    for _ in range(min(N_WORKERS, len(remaining))):
        d = remaining.pop(0)
        running[ray_separation.remote(d, target_diseases[d], query_genes,
                                       PPI_PATH, DIST_PATH, N_ITER)] = d

    while running:
        ready, _ = ray.wait(list(running.keys()), num_returns=1)
        ref      = ready[0]
        disease_name, result, elapsed = ray.get(ref)
        save(result, disease_to_filename(disease_name))
        del running[ref]
        completed_count += 1

        gene_size = len(target_diseases[disease_name])
        empirical_log.append((gene_size, elapsed))

        run_entry["per_disease"].append({
            "disease": disease_name, "gene_size": gene_size,
            "elapsed_s": round(elapsed, 1),
            "z_score":   round(result["z_score"], 4),
            "p_value":   round(result["p_value_double_tail"], 6),
        })

        eta_bench = simulate_parallel_runtime(
            [predict_runtime_benchmark(d, empirical_log) for d in remaining], N_WORKERS
        ) if remaining else 0

        if remaining:
            pred_fn, emp_label = fit_empirical_model(empirical_log)
            eta_live = simulate_parallel_runtime(
                [pred_fn(len(target_diseases[d])) for d in remaining], N_WORKERS
            )
        else:
            eta_live, emp_label = 0, "done"

        log(f"[{completed_count:>3}/{len(target_diseases)}] "
            f"{disease_name:<40} size={gene_size:<5} "
            f"z={result['z_score']:+.3f} p={result['p_value_double_tail']:.4f} {elapsed:.0f}s"
            f" | ETA bench={eta_bench/60:.0f}min live={eta_live/60:.0f}min [{emp_label}]")

        if remaining:
            d = remaining.pop(0)
            running[ray_separation.remote(d, target_diseases[d], query_genes,
                                           PPI_PATH, DIST_PATH, N_ITER)] = d

    ray.shutdown()
    wall_total = time.time() - t_wall
    run_entry.update({
        "run_end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "diseases_end": completed_count,
        "wall_seconds": round(wall_total, 1),
        "wall_human": f"{wall_total/3600:.2f} hrs ({wall_total/60:.0f} min)",
    })
    run_log["runs"].append(run_entry)
    save_run_log(run_log)
    log(f"Total wall time: {wall_total/3600:.2f} hrs")

# ── Merge checkpoints ─────────────────────────────────────────────────────────

log("Merging checkpoints...")
separation_results = {
    "z_score": {}, "p_value_double_tail": {}, "raw_separation": {},
    "d_mu": {}, "d_sigma": {}, "null_distribution": {},
}
missing = []
for d in target_diseases:
    cp = disease_to_filename(d)
    if os.path.exists(cp):
        r = load(cp)
        separation_results["z_score"][d]             = r["z_score"]
        separation_results["p_value_double_tail"][d] = r["p_value_double_tail"]
        separation_results["raw_separation"][d]      = r["raw_separation"]
        separation_results["d_mu"][d]                = r["d_mu"]
        separation_results["d_sigma"][d]             = r["d_sigma"]
        separation_results["null_distribution"][d]   = r.get("dist")
    else:
        missing.append(d)

if missing:
    log(f"WARNING: {len(missing)} diseases missing checkpoints: {missing[:5]}")
else:
    save(separation_results, RESULTS_CACHE)
    log(f"Saved: {RESULTS_CACHE}")
    log("Done.")
