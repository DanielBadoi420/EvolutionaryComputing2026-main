"""
Assignment 1: tournament selection (k=2 versus k=5) and random search.

To run write: 
uv run assignments/assignment_1/A1_experiment.py --output __data__/a1_full
in the main directory.

It saves the results in __data__/a1_full
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.operators import (
    add_node, crossover_subtree, mutate_replace_node, random_tree, remove_subtree,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES, ALLOWED_ROTATIONS, IDX_OF_CORE, ModuleType,
)
from tree_edit_distance import distances_to_targets

HERE = Path(__file__).resolve().parent
METHODS = ("ea_k2", "ea_k5", "random")
LABELS = {"ea_k2": "EA: tournament k=2", "ea_k5": "EA: tournament k=5",
          "random": "Random search"}
COLORS = {"ea_k2": "#176A9B", "ea_k5": "#CC5529", "random": "#626970"}


@dataclass(frozen=True)
class Settings:
    population: int = 50
    offspring: int = 50
    generations: int = 100
    max_modules: int = 20  #TOTAL modules, including the core
    crossover_rate: float = 0.7
    mutation_rate: float = 0.8

    @property
    def budget(self) -> int:
        return self.population + self.generations * self.offspring


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_targets() -> tuple[list[nx.DiGraph], list[Path]]:
    paths = sorted((HERE / "target_bodies").glob("*.json"))
    if not paths:
        raise FileNotFoundError("Place this script in assignments/assignment_1, beside target_bodies.")
    graphs = [nx.node_link_graph(json.loads(p.read_text(encoding="utf-8")), edges="edges")
              for p in paths]
    for graph in graphs:
        if not nx.is_arborescence(graph):
            raise ValueError("A target is not a rooted directed tree.")
    return graphs, paths


def body_signature(genome: TreeGenome) -> tuple:
    """Exact rooted graph identity, including labels/faces but ignoring node IDs.
    This measures phenotypic richness, not the magnitude of pairwise differences.
    It does not identify physically symmetric bodies under global rotations.
    """
    children = defaultdict(list)
    for edge in genome.edges:
        children[edge["parent"]].append((edge["face"], edge["child"]))

    def visit(node):
        attrs = genome.nodes[node]
        branches = tuple((face, visit(child)) for face, child in sorted(children[node]))
        return attrs["type"], attrs["rotation"], branches

    return visit(IDX_OF_CORE)


def valid_body(genome: TreeGenome, limit: int) -> bool:
    if not 1 <= len(genome.nodes) <= limit:
        return False
    try:
        validate_genome_dict(genome.to_dict())
        graph = genome.to_networkx()
        return (nx.is_arborescence(graph) and graph.in_degree(IDX_OF_CORE) == 0
                and genome.nodes[IDX_OF_CORE]["type"] == "CORE"
                and sum(n["type"] == "CORE" for n in genome.nodes.values()) == 1)
    except (ValueError, KeyError, nx.NetworkXException):
        return False


def sample_body(limit: int) -> TreeGenome:
    """Same generator for both EAs' initial populations and all random-search draws.
    Uniform total size 1...limit, followed by ARIEL's random topology generator.
    This is not uniform sampling over the complete space of tree phenotypes.
    In the supplied ARIEL version random_tree(n) adds n nodes to a core.
    """
    total = random.randint(1, limit)
    genome = random_tree(total - 1)
    if len(genome.nodes) != total or not valid_body(genome, limit):
        raise RuntimeError("Unexpected random_tree behaviour; check the supplied ARIEL version.")
    return genome


def tournament(parents: list[Individual], k: int) -> Individual:
    """Sample with replacement; randomise fitness ties among sampled contestants."""
    contestants = random.choices(parents, k=k)
    best = min(ind.fitness for ind in contestants)
    return random.choice([ind for ind in contestants if ind.fitness == best])


def mutate_body(genome: TreeGenome, settings: Settings) -> TreeGenome:
    """Four equally likely mutation attempts; mutation probability is per child.
    Add one module; delete a non-core subtree; change one rotation; or apply
    ARIEL's type-replacement operator (which prunes incompatible children).
    Impossible operations are no-ops, never repeated until improvement.
    """
    child = copy.deepcopy(genome)
    operation = random.choice(("add", "delete", "rotate", "type"))
    noncore = [node for node in child.nodes if node != IDX_OF_CORE]

    if operation == "add" and len(child.nodes) < settings.max_modules:
        occupied = {(e["parent"], e["face"]) for e in child.edges}
        free = [(node, face.name) for node, data in child.nodes.items()
                for face in ALLOWED_FACES[ModuleType[data["type"]]]
                if (node, face.name) not in occupied]
        if free:
            parent, face = random.choice(free)
            module_type = random.choice((ModuleType.BRICK, ModuleType.HINGE))
            rotation = random.choice(ALLOWED_ROTATIONS[module_type]).name
            add_node(child, parent, face, max(child.nodes) + 1, module_type.name, rotation)
    elif operation == "delete" and noncore:
        remove_subtree(child, random.choice(noncore))
    elif operation == "rotate" and noncore:
        node = random.choice(noncore)
        data = child.nodes[node]
        choices = [r.name for r in ALLOWED_ROTATIONS[ModuleType[data["type"]]]
                   if r.name != data["rotation"]]
        if choices:
            data["rotation"] = random.choice(choices)
    elif operation == "type" and noncore:
        mutate_replace_node(child)

    return child if valid_body(child, settings.max_modules) else copy.deepcopy(genome)


class Experiment:
    """One independent run; actual EA persistence is handled by ariel.ec.EA."""

    def __init__(self, settings: Settings, method: str, seed: int,
                 targets: list[nx.DiGraph], folder: Path):
        self.settings, self.method, self.seed = settings, method, seed
        self.targets, self.folder = targets, folder
        self.evaluations, self.generation = 0, 0
        self.history: list[dict] = []
        self.best: dict | None = None

    def evaluate_body(self, genome: TreeGenome) -> Individual:
        if not valid_body(genome, self.settings.max_modules):
            raise ValueError("An invalid or over-budget body reached evaluation.")
        distances = list(distances_to_targets(genome.to_networkx(), self.targets))
        #exact arithmetic and population-SD convention from tree_edit_distance.py.
        mean = sum(distances) / len(distances)
        spread = (sum((d - mean) ** 2 for d in distances) / len(distances)) ** 0.5
        ind = Individual()
        ind.genotype = copy.deepcopy(genome.to_dict())
        ind.fitness = mean + spread
        ind.tags = {"per_target": distances, "modules": len(genome.nodes),
                    "evaluation_index": self.evaluations + 1}
        #every proposed valid candidate is scored, including identical copies.
        self.evaluations += 1
        if self.best is None or ind.fitness < self.best["fitness"]:
            self.best = {"fitness": ind.fitness, "mean_distance": mean,
                         "std_across_targets": spread, "per_target": distances,
                         "modules": len(genome.nodes), "generation": self.generation,
                         "evaluation_index": self.evaluations,
                         "genotype": copy.deepcopy(genome.to_dict())}
        return ind

    def reproduce(self, population: Population) -> Population:
        self.generation += 1
        parents = population.alive.to_list()
        if len(parents) != self.settings.population:
            raise RuntimeError("Population size changed unexpectedly.")
        self.parent_ids = {ind.id for ind in parents}
        children = []
        k = int(self.method.removeprefix("ea_k"))
        while len(children) < self.settings.offspring:
            p1, p2 = tournament(parents, k), tournament(parents, k)
            g1 = TreeGenome.from_dict(copy.deepcopy(p1.genotype))
            g2 = TreeGenome.from_dict(copy.deepcopy(p2.genotype))
            if random.random() < self.settings.crossover_rate:
                c1, c2 = crossover_subtree(g1, g2)
            else:
                c1, c2 = copy.deepcopy(g1), copy.deepcopy(g2)
            for candidate, parent in ((c1, g1), (c2, g2)):
                if len(children) == self.settings.offspring:
                    break
                #revert an invalid/oversized crossover child before mutation.
                if not valid_body(candidate, self.settings.max_modules):
                    candidate = copy.deepcopy(parent)
                if random.random() < self.settings.mutation_rate:
                    candidate = mutate_body(candidate, self.settings)
                children.append(self.evaluate_body(candidate))
        population.extend(children)
        return population

    def survive(self, population: Population) -> Population:
        """One parental elite plus the best mu-1 offspring; random fitness ties."""
        parents = [ind for ind in population if ind.id in self.parent_ids]
        offspring = [ind for ind in population if ind.id not in self.parent_ids]
        #fresh children have id=None until EA commits them; parents have DB ids.
        random.shuffle(parents)
        random.shuffle(offspring)
        elite = min(parents, key=lambda ind: ind.fitness)
        offspring.sort(key=lambda ind: ind.fitness)
        survivors = [elite] + offspring[: self.settings.population - 1]
        identities = {id(ind) for ind in survivors}
        for ind in population:
            ind.alive = id(ind) in identities
        if sum(ind.alive for ind in population) != self.settings.population:
            raise RuntimeError("Survivor selection did not preserve population size.")
        return population

    def record(self, population: Population) -> Population:
        individuals = population.alive.to_list()
        values = [ind.fitness for ind in individuals]
        signatures = {body_signature(TreeGenome.from_dict(ind.genotype))
                      for ind in individuals}
        self.history.append({
            "generation": self.generation,
            "evaluations": self.evaluations,
            "best_so_far": self.best["fitness"],
            "population_best": min(values),
            "population_mean": statistics.mean(values),
            "population_worst": max(values),
            "unique_fraction": len(signatures) / len(individuals),
            "mean_modules": statistics.mean(ind.tags["modules"] for ind in individuals),
            "best_modules": self.best["modules"],
        })
        write_csv(self.folder / "history.csv", self.history)
        if self.generation % 10 == 0 or self.generation == self.settings.generations:
            print(f"  {self.method} seed={self.seed} gen={self.generation} "
                  f"evaluations={self.evaluations} best={self.best['fitness']:.4f}", flush=True)
        return population

    def run(self) -> dict:
        self.folder.mkdir(parents=True, exist_ok=False)
        #reset immediately before initialisation, after all module imports.
        random.seed(self.seed)
        started = time.perf_counter()
        initial = Population([self.evaluate_body(sample_body(self.settings.max_modules))
                              for _ in range(self.settings.population)])
        write_json(self.folder / "initial_population.json",
                   [{"genotype": ind.genotype, "fitness": ind.fitness} for ind in initial])
        self.record(initial)
        if self.method == "random":
            #no selection, inheritance or mutation. Keep a best-so-far record only.
            for generation in range(1, self.settings.generations + 1):
                self.generation = generation
                batch = Population([self.evaluate_body(sample_body(self.settings.max_modules))
                                    for _ in range(self.settings.offspring)])
                self.record(batch)
        else:
            operations = [EAOperation(self.reproduce), EAOperation(self.survive),
                          EAOperation(self.record)]
            ea = EA(initial, operations=operations, num_steps=self.settings.generations,
                    first_generation_id=0, is_maximisation=False,
                    db_file_path=self.folder / "population.db", db_handling="halt", quiet=True)
            try:
                ea.run()
            finally:
                ea.engine.dispose()
        if self.evaluations != self.settings.budget:
            raise RuntimeError("Evaluation count differs from the declared budget.")
        elapsed = time.perf_counter() - started
        write_json(self.folder / "best_genome.json", self.best["genotype"])
        graph = TreeGenome.from_dict(self.best["genotype"]).to_networkx()
        write_json(self.folder / "best_body.json", nx.node_link_data(graph, edges="edges"))
        result = {key: value for key, value in self.best.items() if key != "genotype"}
        result.update({"method": self.method, "seed": self.seed,
                       "evaluations": self.evaluations, "seconds": elapsed,
                       "final_unique_fraction": self.history[-1]["unique_fraction"],
                       "final_mean_modules": self.history[-1]["mean_modules"],
                       #best-so-far area: average over the post-initialisation budget.
                       #lower means good fitness was found earlier, as well as finally.
                       "convergence_auc": sum(
                           (a["best_so_far"] + b["best_so_far"]) / 2
                           * (b["evaluations"] - a["evaluations"])
                           for a, b in zip(self.history, self.history[1:]))
                           / (self.settings.generations * self.settings.offspring)})
        write_json(self.folder / "result.json", result)
        return result


def aggregate(output: Path) -> None:
    """Summarise independent runs. Error bands use sample SD (ddof=1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup = json.loads((output / "setup.json").read_text(encoding="utf-8"))
    results, histories = [], {}
    for method in setup["methods"]:
        histories[method] = []
        for seed in setup["seeds"]:
            folder = output / method / f"seed_{seed}"
            if not (folder / "result.json").exists():
                raise FileNotFoundError(f"Run is incomplete: {folder}")
            results.append(json.loads((folder / "result.json").read_text(encoding="utf-8")))
            with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
                histories[method].append([{k: float(v) for k, v in row.items()}
                                          for row in csv.DictReader(stream)])
    write_csv(output / "all_runs.csv",
              [{k: v for k, v in row.items() if k != "per_target"} for row in results])
    summary, targets, curve_rows = [], [], []
    for method in setup["methods"]:
        group = [row for row in results if row["method"] == method]
        row = {"method": method, "runs": len(group), "evaluations_per_run": group[0]["evaluations"]}
        for metric in ("fitness", "convergence_auc", "modules", "seconds",
                       "final_unique_fraction", "final_mean_modules"):
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(row)
        for index, target in enumerate(setup["targets"]):
            values = [r["per_target"][index] for r in group]
            targets.append({"method": method, "target": target["name"],
                            "mean": statistics.mean(values),
                            "sd": statistics.stdev(values) if len(values) > 1 else ""})
        for points in zip(*histories[method], strict=True):
            if len({p["evaluations"] for p in points}) != 1:
                raise ValueError("Evaluation checkpoints do not align across seeds.")
            curve = {"method": method, "generation": int(points[0]["generation"]),
                     "evaluations": int(points[0]["evaluations"])}
            for metric in ("best_so_far", "unique_fraction", "mean_modules"):
                values = [p[metric] for p in points]
                curve[metric + "_mean"] = statistics.mean(values)
                curve[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
            curve_rows.append(curve)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "per_target_summary.csv", targets)
    write_csv(output / "curves_summary.csv", curve_rows)
    #within-seed differences acknowledge the matched initial population design.
    paired = []
    if all(method in setup["methods"] for method in ("ea_k2", "ea_k5")):
        for seed in setup["seeds"]:
            low = next(r for r in results if r["method"] == "ea_k2" and r["seed"] == seed)
            high = next(r for r in results if r["method"] == "ea_k5" and r["seed"] == seed)
            paired.append({"seed": seed, "fitness_k5_minus_k2": high["fitness"] - low["fitness"],
                           "auc_k5_minus_k2": high["convergence_auc"] - low["convergence_auc"],
                           "diversity_k5_minus_k2": high["final_unique_fraction"]
                           - low["final_unique_fraction"]})
        write_csv(output / "paired_differences.csv", paired)

    def draw(ax, metric, xlabel, ylabel, methods):
        for method in methods:
            rows = [row for row in curve_rows if row["method"] == method]
            x = [row[xlabel] for row in rows]
            y = [row[metric + "_mean"] for row in rows]
            ax.plot(x, y, color=COLORS[method], label=LABELS[method], linewidth=1.8)
            if len(setup["seeds"]) > 1:
                sd = [row[metric + "_sd"] for row in rows]
                ax.fill_between(x, [a-b for a,b in zip(y,sd)], [a+b for a,b in zip(y,sd)],
                                color=COLORS[method], alpha=0.14)
        ax.set_xlabel("Candidate evaluations" if xlabel == "evaluations" else "Generation")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    mode = "PILOT - checking the implementation" if setup["pilot"] else "Tournament selection experiment"
    caption = (f"{len(setup['seeds'])} independent runs per method; bands: sample SD across runs"
               if len(setup["seeds"]) > 1 else "One run per method; variability is not estimated")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    draw(ax, "best_so_far", "evaluations", "Best-so-far fitness (lower is better)", setup["methods"])
    ax.set_title(mode, loc="left")
    ax.legend(frameon=False)
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "convergence.png", dpi=180)
    fig.savefig(output / "convergence.pdf")
    plt.close(fig)
    ea_methods = [m for m in setup["methods"] if m != "random"]
    if ea_methods:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        draw(axes[0], "best_so_far", "generation", "Best-so-far fitness", ea_methods)
        draw(axes[1], "unique_fraction", "generation", "Fraction of distinct body graphs", ea_methods)
        draw(axes[2], "mean_modules", "generation", "Mean modules (including core)", ea_methods)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(mode, fontsize=12)
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(output / "ea_diagnostics.png", dpi=180)
        fig.savefig(output / "ea_diagnostics.pdf")
        plt.close(fig)
    print("\nFinal fitness (mean +/- sample SD across runs; lower is better):")
    for row in summary:
        sd = row["fitness_sd"]
        spread = f"{sd:.4f}" if sd != "" else "not estimated"
        print(f"  {row['method']:8s}  {row['fitness_mean']:.4f} +/- {spread}")


def provenance(settings: Settings, methods, seeds, pilot, target_paths) -> dict:
    source_root = Path(sys.modules["ariel"].__file__).resolve().parent
    files = [Path(__file__).resolve(), HERE / "tree_edit_distance.py"]
    files += sorted((source_root / "ec").rglob("*.py"))
    files += [source_root / "body_phenotypes" / "robogen_lite" / "config.py"]
    versions = {}
    for name in ("networkx", "numpy", "sqlmodel", "sqlalchemy", "pydantic",
                 "pydantic-settings", "rich", "matplotlib"):
        versions[name] = importlib.metadata.version(name)
    return {"settings": asdict(settings), "methods": list(methods), "seeds": list(seeds),
            "pilot": pilot, "budget_per_run": settings.budget,
            "created": datetime.now().isoformat(), "python": sys.version,
            "platform": platform.platform(), "packages": versions,
            "targets": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in target_paths],
            "source_sha256": {str(p.relative_to(source_root)) if p.is_relative_to(source_root)
                              else p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "initialisation": "Uniform total size 1..max_modules; ARIEL random_tree(total-1)",
            "parent_selection": "k-tournament with replacement; random ties",
            "survivor_selection": "One parental elite plus best population-1 offspring; random ties",
            "mutation": "Per-child probability; equal add/delete-subtree/rotation/type attempts",
            "constraint_handling": "Invalid or oversized variation reverts to preceding valid body",
            "diversity": "Fraction of distinct rooted labelled graphs, including faces, ignoring node IDs",
            "fitness_sd": "Population SD across targets (ddof=0)",
            "reported_run_sd": "Sample SD across independent runs (ddof=1)",
            "random_population_statistics": "Fresh candidate batches, not selected EA populations"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="10 parents, 10 children, 5 generations, seeds 11/22")
    parser.add_argument("--output", type=Path, help="New output folder; must be empty")
    parser.add_argument("--analyse-only", action="store_true", help="Regenerate summaries/plots from --output")
    parser.add_argument("--render-best", type=Path, metavar="BEST_BODY_JSON", help="Render a saved body using the original template")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--offspring", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-modules", type=int, default=20)
    parser.add_argument("--crossover-rate", type=float, default=0.7)
    parser.add_argument("--mutation-rate", type=float, default=0.8)
    args = parser.parse_args()
    if args.render_best:
        from A1_template_2026 import show_body
        graph = nx.node_link_graph(json.loads(args.render_best.read_text(encoding="utf-8")), edges="edges")
        show_body(graph, mode="frame", file_name=args.render_best.parent.parent.name + "_" + args.render_best.parent.name)
        return
    if args.analyse_only:
        if args.output is None:
            parser.error("--analyse-only requires --output")
        aggregate(args.output)
        return
    if args.pilot:
        args.population, args.offspring, args.generations, args.seeds = 10, 10, 5, [11, 22]
    if (args.population < 2 or args.offspring < args.population - 1
            or args.generations < 1 or not 2 <= args.max_modules <= 100
            or not 0 <= args.crossover_rate <= 1 or not 0 <= args.mutation_rate <= 1):
        parser.error("Require population>=2, offspring>=population-1, generations>=1, "
                     "2<=max-modules<=100, and probabilities in [0,1].")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("Seeds and methods must be unique.")
    settings = Settings(args.population, args.offspring, args.generations, args.max_modules,
                        args.crossover_rate, args.mutation_rate)
    output = args.output or Path("__data__") / (
        ("a1_pilot_" if args.pilot else "a1_full_") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output is not empty: {output}. Choose a new folder; existing runs are preserved.")
    targets, target_paths = load_targets()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "setup.json", provenance(settings, args.methods, args.seeds, args.pilot, target_paths))
    print(f"{'PILOT' if args.pilot else 'EXPERIMENT'} | targets: {[len(g) for g in targets]} | "
          f"evaluations/run: {settings.budget} | runs: {len(args.methods) * len(args.seeds)}", flush=True)
    print(f"Output: {output}", flush=True)
    for method in args.methods:
        for seed in args.seeds:
            Experiment(settings, method, seed, targets, output / method / f"seed_{seed}").run()
    aggregate(output)
    print(f"\nDone. Results and plots: {output}")
    if args.pilot:
        print("PILOT ONLY: use this to check the code and runtime, not to draw research conclusions.")
    elif len(args.seeds) < 5:
        print("This batch has fewer than five runs/method; the assignment requires at least five.")


if __name__ == "__main__":
    main()
