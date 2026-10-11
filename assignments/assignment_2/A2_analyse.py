"""Analysis for A2_experiment.py; also runnable on an existing output folder.

  uv run assignments/assignment_2/A2_analyse.py __data__/a2_pilot
"""

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import rankdata


METHODS = ("fixed_small", "fixed_large", "adaptive", "random")
LABELS = {"fixed_small": "Fixed sigma = 0.05", "fixed_large": "Fixed sigma = 0.20",
          "adaptive": "Self-adaptive sigma", "random": "Random search (fresh batch)"}
COLORS = {"fixed_small": "#2471A3", "fixed_large": "#D35400", "adaptive": "#17805C", "random": "#6C7075"}


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def signed_rank_permutation(differences):
    """Two-sided paired signed-rank test, exact sign enumeration up to 18 pairs.

    Zero differences are discarded; ties receive average ranks. Enumerating
    signs explicitly handles tied absolute differences without using a false
    no-ties exact Wilcoxon distribution. H0 requires sign symmetry of the
    paired differences, with independent experimental runs.
    """
    differences = np.round(np.asarray(differences, dtype=float), 12)
    differences = differences[differences != 0]
    n = len(differences)
    if n == 0:
        return 1.0, 0.0, "all_zero", 0
    ranks = rankdata(np.abs(differences), method="average")
    signed_sum = float(np.dot(np.sign(differences), ranks))
    threshold = abs(signed_sum) - 1e-12
    if n <= 18:
        extreme = sum(abs(np.dot(signs, ranks)) >= threshold
                      for signs in itertools.product((-1, 1), repeat=n))
        pvalue = extreme / 2 ** n
        mode = "exact_sign_enumeration"
    else:
        rng = np.random.default_rng(3807)
        extreme, total = 0, 100_000
        for _ in range(100):
            signs = rng.choice([-1, 1], size=(1000, n))
            extreme += int(np.count_nonzero(np.abs(signs @ ranks) >= threshold))
        pvalue = (extreme + 1) / (total + 1)
        mode = "100000_sign_permutations"
    return float(pvalue), signed_sum / float(ranks.sum()), mode, n


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def analyse(root):
    root = Path(root)
    setup = json.loads((root / "setup.json").read_text(encoding="utf-8"))
    cfg = setup["settings"]
    labels = dict(LABELS)
    labels["fixed_small"] = f"Fixed sigma = {cfg['sigma_small']:g}"
    labels["fixed_large"] = f"Fixed sigma = {cfg['sigma_large']:g}"
    history = {}
    for method in METHODS:
        for seed in setup["seeds"]:
            path = root / method / f"seed_{seed}" / "history.csv"
            if not path.exists():
                raise ValueError(f"Missing run: {path}. This output is incomplete.")
            with path.open(encoding="utf-8") as handle:
                history[method, seed] = [{k: (v if k == "method" else float(v)) for k, v in row.items()}
                                         for row in csv.DictReader(handle)]
    lengths = {len(rows) for rows in history.values()}
    if len(lengths) != 1 or setup.get("status") != "completed":
        raise ValueError("Runs are incomplete or have unequal lengths. Do not compare their final results.")
    n = len(setup["seeds"])
    plot_title = "PILOT ONLY" if cfg["pilot"] else "Neuroevolution: mutation strength"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    generations = np.array([row["generation"] for row in next(iter(history.values()))])
    curve_rows = []
    metrics = ("best_so_far", "population_mean", "population_worst", "weight_diversity", "sigma_mean")
    arrays = {}
    for method in METHODS:
        arrays[method] = {}
        for metric in metrics:
            arrays[method][metric] = np.array([[row[metric] for row in history[method, seed]]
                                              for seed in setup["seeds"]])
        for index, generation in enumerate(generations):
            row = {"method": method, "generation": int(generation), "runs": n,
                   "evaluations": int(history[method, setup["seeds"][0]][index]["evaluations"])}
            for metric in metrics:
                values = arrays[method][metric][:, index]
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_sample_sd"] = sample_sd(values)
            curve_rows.append(row)
    save_csv(root / "curves_summary.csv", curve_rows)

    def draw(axis, metric, methods=METHODS):
        for method in methods:
            values = arrays[method][metric]
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
            axis.plot(generations, mean, color=COLORS[method],
                      linestyle="--" if method == "random" else "-", label=labels[method])
            axis.fill_between(generations, mean - sd, mean + sd, color=COLORS[method], alpha=.14)
        axis.set_xlabel("Generation / sampling batch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    draw(axes[0], "best_so_far")
    draw(axes[1], "population_mean")
    axes[0].set(title="Best-so-far fitness", ylabel="Remaining target distance (m; lower is better)")
    axes[1].set(title="Mean surviving-population / fresh-batch fitness", ylabel="Remaining target distance (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle(plot_title)
    fig.text(.5, .01, f"{n} independent runs per method; bands: sample SD across runs. "
             "Random search uses fresh batches, not selected populations.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"fitness_curves.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    draw(axes[0], "weight_diversity")
    draw(axes[1], "sigma_mean", METHODS[:-1])
    axes[0].set(title="Genotypic diversity", ylabel="Mean pairwise RMS weight difference")
    axes[1].set(title="Mutation strength", ylabel="Mean sigma in surviving population")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle(plot_title)
    fig.text(.5, .01, "Diversity measures network parameters, not behavioural differences. "
             "Bands: sample SD across runs.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"diversity_and_sigma.{suffix}", dpi=180)
    plt.close(fig)

    summary = []
    finals = {m: arrays[m]["best_so_far"][:, -1] for m in METHODS}
    for method in METHODS:
        summary.append({"method": method, "runs": n, "final_best_mean": float(finals[method].mean()),
                        "final_best_sample_sd": sample_sd(finals[method]),
                        "final_population_or_batch_mean": float(arrays[method]["population_mean"][:, -1].mean()),
                        "final_weight_diversity_mean": float(arrays[method]["weight_diversity"][:, -1].mean())})
    save_csv(root / "summary.csv", summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(METHODS))
    for index, seed in enumerate(setup["seeds"]):
        ax.plot(x, [finals[m][index] for m in METHODS], color="#aaaaaa", alpha=.5, linewidth=.8)
    for index, method in enumerate(METHODS):
        ax.scatter(np.full(n, index), finals[method], color=COLORS[method], s=35, zorder=3)
        ax.scatter(index, finals[method].mean(), marker="_", color="black", s=350, zorder=4)
    ax.set_xticks(x, [labels[m].replace(" (fresh batch)", "") for m in METHODS], fontsize=8)
    ax.set(ylabel="Final best target distance (m)", title=f"{plot_title}: individual run outcomes")
    ax.grid(axis="y", alpha=.2)
    fig.text(.5, .015, "Dots: independent runs. Grey lines: matched seeds. Black marks: means.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"final_fitness.{suffix}", dpi=180)
    plt.close(fig)

    print("\nFinal best distance: mean +/- sample SD across runs (metres; lower is better)")
    for row in summary:
        print(f"  {row['method']:12s} {row['final_best_mean']:.4f} +/- {row['final_best_sample_sd']:.4f}")
    if not cfg["pilot"]:
        # Primary tests use the preset min_generations checkpoint, reached by
        # every run before plateau stopping is allowed. This avoids choosing
        # the inferential endpoint from the observed convergence curves.
        primary_generation = cfg["min_generations"]
        primary = {m: arrays[m]["best_so_far"][:, primary_generation] for m in METHODS}
        primary_rows = [{"method": m, "seed": seed, "generation": primary_generation,
                         "evaluations": cfg["population"] + primary_generation * cfg["offspring"],
                         "best_distance_m": float(primary[m][i])}
                        for m in METHODS for i, seed in enumerate(setup["seeds"])]
        save_csv(root / "primary_endpoint_runs.csv", primary_rows)
        # Predeclared family: adaptive versus each of the three references.
        tests = []
        for index, reference in enumerate(("fixed_small", "fixed_large", "random")):
            differences = primary["adaptive"] - primary[reference]
            p, effect, mode, nonzero = signed_rank_permutation(differences)
            rng = np.random.default_rng(7300 + index)
            means = rng.choice(differences, size=(10_000, n), replace=True).mean(axis=1)
            tests.append({"comparison": "adaptive_minus_" + reference,
                          "primary_generation": primary_generation, "pairs": n,
                          "nonzero_pairs": nonzero, "mean_difference_m": float(differences.mean()),
                          "median_difference_m": float(np.median(differences)),
                          "bootstrap_mean_ci_low": float(np.quantile(means, .025)),
                          "bootstrap_mean_ci_high": float(np.quantile(means, .975)),
                          "signed_rank_biserial": effect, "p_raw": p, "test_method": mode})
        adjusted = holm_adjust([row["p_raw"] for row in tests])
        for row, p in zip(tests, adjusted):
            row["p_holm"] = float(p)
            row["reject_at_0_05"] = bool(p < .05)
        save_csv(root / "paired_tests_primary.csv", tests)
        print(f"Primary paired tests at preset generation {primary_generation} saved; "
              "negative differences favour adaptive. Final-at-stop results are descriptive.")
    print(f"Results and plots: {root.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyse(parser.parse_args().output)
