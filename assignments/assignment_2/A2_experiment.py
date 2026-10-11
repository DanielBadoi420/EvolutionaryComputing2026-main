"""A2: fixed versus self-adaptive Gaussian mutation of neural controllers.

Run from the ARIEL repository root (Python >=3.12):
  uv run assignments/assignment_2/A2_experiment.py --pilot --output __data__/a2_pilot

The full preset is provisional: inspect the pilot and its timings first.
All variation uses explicit NumPy generators; ariel.ec.set_seed is not needed.
No changes to src/ariel are required. See A2_README.md for exact methodology.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.simulation.environments import SimpleFlatWorld


METHODS = ("fixed_small", "fixed_large", "adaptive", "random")
METHOD_CODE = dict(zip(METHODS, (101, 102, 103, 104)))
BAD_FITNESS = 1_000_000.0


@dataclass(frozen=True)
class Settings:
    pilot: bool = True
    population: int = 8
    offspring: int = 16
    max_generations: int = 5
    min_generations: int = 5
    patience: int = 20
    plateau_tolerance: float = 0.01  # metres of best-so-far improvement
    duration: float = 15.0
    hidden: int = 6
    tournament: int = 3
    sigma_small: float = 0.05
    sigma_large: float = 0.20
    sigma_min: float = 0.001
    sigma_max: float = 1.0
    weight_bound: float = 5.0
    initial_sd: float = 0.5
    rhythm_hz: float = 1.0
    target_x: float = 2.0
    target_y: float = 0.0
    spawn_z: float = 0.1
    target_scale: float = 2.0
    velocity_scale: float = 10.0


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_row(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class Simulator:
    """One fixed compiled model, reset completely for each candidate.

    The controller is stateless: repeated MuJoCo callback calls at a physics
    step do not accumulate an action. Every evaluation detaches the callback.
    This evaluator is serial; do not share its global callback across threads.
    """

    def __init__(self, settings):
        self.settings = settings
        mj.set_mjcb_control(None)
        world = SimpleFlatWorld()
        world.spawn(gecko().spec, position=[0, 0, settings.spawn_z],
                    correct_collision_with_floor=True)
        self.model_xml = world.spec.to_xml()
        self.model = world.spec.compile()
        self.data = mj.MjData(self.model)
        self.xml_hash = hashlib.sha256(self.model_xml.encode()).hexdigest()
        self.outputs = self.model.nu
        joint_ids = self.model.actuator_trnid[:, 0]
        if not np.all(self.model.jnt_type[joint_ids] == mj.mjtJoint.mjJNT_HINGE):
            raise ValueError("Expected one hinge joint per actuator.")
        self.q_addresses = self.model.jnt_qposadr[joint_ids]
        self.v_addresses = self.model.jnt_dofadr[joint_ids]
        free = np.flatnonzero(self.model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected exactly one free joint for the robot core.")
        self.core_body = int(self.model.jnt_bodyid[free[0]])
        self.core_q = int(self.model.jnt_qposadr[free[0]])
        self.inputs = 2 * self.outputs + 7  # q, qdot, up(3), target(2), clock(2)
        h = settings.hidden
        self.dimension = self.inputs * h + h + h * self.outputs + self.outputs
        self.steps = int(math.ceil(settings.duration / self.model.opt.timestep))
        self.total_evaluation_seconds = 0.0
        self.calls = 0
        self.executor = None
        self.batch_wall_seconds = 0.0
        self.batch_evaluations = 0
        self.lower = np.full(self.outputs, -np.pi / 2)
        self.upper = np.full(self.outputs, np.pi / 2)
        limited = self.model.actuator_ctrllimited.astype(bool)
        self.lower[limited] = np.maximum(self.lower[limited],
                                        self.model.actuator_ctrlrange[limited, 0])
        self.upper[limited] = np.minimum(self.upper[limited],
                                        self.model.actuator_ctrlrange[limited, 1])

    def unpack(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.dimension,) or not np.isfinite(weights).all():
            raise ValueError("Incorrect or nonfinite controller genotype.")
        h, n, u = self.settings.hidden, self.inputs, self.outputs
        a, b, c = n * h, n * h + h, n * h + h + h * u
        return weights[:a].reshape(n, h), weights[a:b], weights[b:c].reshape(h, u), weights[c:]

    def reset(self):
        mj.set_mjcb_control(None)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def position(self):
        return self.data.qpos[self.core_q:self.core_q + 3].copy()

    def actions(self, data, matrices):
        cfg = self.settings
        q = np.clip(data.qpos[self.q_addresses] / (np.pi / 2), -1, 1)
        v = np.clip(data.qvel[self.v_addresses] / cfg.velocity_scale, -1, 1)
        rotation = data.xmat[self.core_body].reshape(3, 3)
        position = data.qpos[self.core_q:self.core_q + 3]
        relative = rotation.T @ np.array([cfg.target_x - position[0],
                                         cfg.target_y - position[1], 0.0])
        target = np.clip(relative[:2] / cfg.target_scale, -1, 1)
        phase = 2 * np.pi * cfg.rhythm_hz * data.time
        inputs = np.concatenate((q, v, rotation[2, :], target,
                                 [np.sin(phase), np.cos(phase)]))
        w1, b1, w2, b2 = matrices
        actions = np.tanh(np.tanh(inputs @ w1 + b1) @ w2 + b2) * (np.pi / 2)
        return np.clip(actions, self.lower, self.upper)

    def evaluate(self, weights, record=False):
        started = time.perf_counter()
        matrices = self.unpack(weights)
        self.reset()
        start = self.position()
        trajectory = []

        def callback(model, data):
            data.ctrl[:] = self.actions(data, matrices)

        invalid = False
        mj.set_mjcb_control(callback)
        try:
            if record:
                trajectory.append([0.0, *start.tolist()])
            for offset in range(0, self.steps, 100):
                before = self.data.time
                mj.mj_step(self.model, self.data, nstep=min(100, self.steps - offset))
                invalid = (not np.isfinite(self.data.qpos).all()
                           or not np.isfinite(self.data.qvel).all()
                           or not np.isfinite(self.data.ctrl).all()
                           or self.data.time <= before)
                for warning in (mj.mjtWarning.mjWARN_BADQPOS,
                                mj.mjtWarning.mjWARN_BADQVEL,
                                mj.mjtWarning.mjWARN_BADQACC,
                                mj.mjtWarning.mjWARN_BADCTRL):
                    invalid = invalid or self.data.warning[warning].number > 0
                if invalid:
                    break
                if record:
                    trajectory.append([float(self.data.time), *self.position().tolist()])
        finally:
            mj.set_mjcb_control(None)
        end = self.position()
        fitness = BAD_FITNESS if invalid else float(np.linalg.norm(
            end[:2] - [self.settings.target_x, self.settings.target_y]))
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.total_evaluation_seconds += elapsed
        # Position is omitted for invalid simulations to keep JSON finite.
        return {"fitness": fitness, "invalid": bool(invalid),
                "start": start.tolist(), "end": None if invalid else end.tolist(),
                "seconds": elapsed, "trajectory": trajectory}

    def start_workers(self, workers):
        if workers > 1:
            # Spawn works on Windows too. Each process owns its own MuJoCo
            # model/data and GLOBAL control callback; no simulation threads.
            self.executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialise_worker,
                initargs=(asdict(self.settings), self.xml_hash))

    def evaluate_many(self, weights):
        started = time.perf_counter()
        if self.executor is None:
            results = [self.evaluate(w) for w in weights]
        else:
            # map preserves input order, keeping selection/tie handling and
            # best-controller logging independent of worker completion order.
            results = list(self.executor.map(evaluate_worker, weights, chunksize=1))
            self.calls += len(results)
            self.total_evaluation_seconds += sum(r["seconds"] for r in results)
        self.batch_wall_seconds += time.perf_counter() - started
        self.batch_evaluations += len(results)
        return results

    def close_workers(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


_WORKER_SIMULATOR = None


def initialise_worker(settings, expected_hash):
    global _WORKER_SIMULATOR
    _WORKER_SIMULATOR = Simulator(Settings(**settings))
    if _WORKER_SIMULATOR.xml_hash != expected_hash:
        raise RuntimeError("Worker robot/world differs from parent process.")


def evaluate_worker(weights):
    return _WORKER_SIMULATOR.evaluate(weights)


def make_individual(weights, sigma):
    individual = Individual()
    individual.genotype = np.asarray(weights, dtype=float).tolist()
    individual.tags = {"sigma": float(sigma)}
    return individual


def weight_diversity(individuals):
    """Mean pairwise Euclidean distance / sqrt(number of NN parameters)."""
    weights = np.array([ind.genotype for ind in individuals], dtype=float)
    if len(weights) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(weights) - 1):
        total += float(np.sqrt(np.mean((weights[i + 1:] - weights[i]) ** 2, axis=1)).sum())
        count += len(weights) - i - 1
    return total / count


class SearchRun:
    """Custom ordered EA operations executed and persisted by ariel.ec.EA."""

    def __init__(self, method, seed, settings, simulator, root):
        self.method, self.seed = method, seed
        self.settings, self.simulator = settings, simulator
        self.directory = root / method / f"seed_{seed}"
        self.directory.mkdir(parents=True)
        # A shared initialisation stream makes all methods begin identically.
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        # Separate streams ensure method order cannot affect any result.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, METHOD_CODE[method]]))
        self.adaptation_rng = np.random.default_rng(np.random.SeedSequence([seed, 900]))
        self.generation = 0
        self.evaluations = 0
        self.invalid_evaluations = 0
        self.history = []
        self.best = None
        self.parents = []
        initial = Population([make_individual(self.random_weights(initial_rng), self.initial_sigma())
                              for _ in range(settings.population)])
        self.evaluate_population(initial)
        self.initial_weights_hash = hashlib.sha256(np.array(
            [ind.genotype for ind in initial], dtype='<f8').tobytes()).hexdigest()
        self.log(initial)
        operations = [EAOperation(self.parent_selection),
                      EAOperation(self.mutation),
                      EAOperation(self.evaluate_population),
                      EAOperation(self.survivor_selection),
                      EAOperation(self.log)]
        self.ea = EA(initial, operations, num_steps=settings.max_generations,
                     first_generation_id=0, is_maximisation=False, quiet=True,
                     db_file_path=self.directory / "population.sqlite", db_handling="halt")

    def initial_sigma(self):
        return self.settings.sigma_small if self.method == "fixed_small" else self.settings.sigma_large

    def random_weights(self, rng):
        s = self.settings
        return np.clip(rng.normal(0, s.initial_sd, self.simulator.dimension),
                       -s.weight_bound, s.weight_bound)

    def parent_selection(self, population: Population) -> Population:
        self.generation += 1
        self.parents = []
        if self.method != "random":
            alive = list(population.alive)
            for _ in range(self.settings.offspring):
                indices = self.rng.integers(0, len(alive), self.settings.tournament)
                entries = [alive[int(i)] for i in indices]
                best = min(ind.fitness for ind in entries)
                ties = [ind for ind in entries if ind.fitness == best]
                self.parents.append(ties[int(self.rng.integers(len(ties)))])
        return population

    def mutation(self, population: Population) -> Population:
        cfg = self.settings
        if self.method == "random":
            children = [make_individual(self.random_weights(self.rng), cfg.sigma_large)
                        for _ in range(cfg.offspring)]
        else:
            children = []
            tau = 1 / math.sqrt(self.simulator.dimension)
            for parent in self.parents:
                sigma = float(parent.tags["sigma"])
                if self.method == "adaptive":
                    sigma = float(np.clip(sigma * np.exp(tau * self.adaptation_rng.normal()),
                                          cfg.sigma_min, cfg.sigma_max))
                weights = np.asarray(parent.genotype) + sigma * self.rng.normal(
                    size=self.simulator.dimension)
                weights = np.clip(weights, -cfg.weight_bound, cfg.weight_bound)
                child = make_individual(weights, sigma)
                child.tags = {"parent_id": parent.id}
                children.append(child)
        population.extend(children)
        return population

    def evaluate_population(self, population: Population) -> Population:
        pending = list(population.unevaluated)
        results = self.simulator.evaluate_many([list(ind.genotype) for ind in pending])
        for individual, result in zip(pending, results):
            individual.fitness = result["fitness"]
            individual.tags = {"invalid_simulation": result["invalid"],
                               "final_position": result["end"],
                               "evaluation_generation": self.generation}
            self.evaluations += 1
            self.invalid_evaluations += int(result["invalid"])
            if self.best is None or individual.fitness < self.best["fitness"]:
                self.best = {"fitness": individual.fitness,
                             "weights": list(individual.genotype),
                             "sigma": float(individual.tags["sigma"]),
                             "method": self.method, "seed": self.seed,
                             "generation_found": self.generation,
                             "evaluation_found": self.evaluations,
                             "final_position": result["end"],
                             "settings": asdict(self.settings),
                             "model_xml_sha256": self.simulator.xml_hash,
                             "inputs": self.simulator.inputs,
                             "outputs": self.simulator.outputs}
                write_json(self.directory / "best.json", self.best)
        return population

    def survivor_selection(self, population: Population) -> Population:
        individuals = list(population)
        if self.method == "random":
            # Keep the complete latest fresh batch; it is not an EA population.
            for ind in individuals:
                ind.alive = ind.tags["evaluation_generation"] == self.generation
        else:
            # (mu + lambda) replacement, with random tie order and elitism.
            order = self.rng.permutation(len(individuals))
            ranked = sorted((individuals[int(i)] for i in order), key=lambda ind: ind.fitness)
            keep = {id(ind) for ind in ranked[:self.settings.population]}
            for ind in individuals:
                ind.alive = id(ind) in keep
        return population

    def log(self, population: Population) -> Population:
        alive = list(population.alive)
        fitness = np.array([ind.fitness for ind in alive])
        sigmas = np.array([ind.tags["sigma"] for ind in alive], dtype=float)
        row = {"method": self.method, "seed": self.seed,
               "generation": self.generation, "evaluations": self.evaluations,
               "best_so_far": self.best["fitness"], "current_best": float(fitness.min()),
               "population_mean": float(fitness.mean()),
               "population_worst": float(fitness.max()),
               "within_population_sd": float(fitness.std(ddof=1)),
               "weight_diversity": weight_diversity(alive),
               "sigma_mean": float(sigmas.mean()),
               "sigma_min": float(sigmas.min()), "sigma_max": float(sigmas.max()),
               "population_n": len(alive),
               "invalid_evaluations": self.invalid_evaluations}
        self.history.append(row)
        append_row(self.directory / "history.csv", row)
        # Save states for audit. This version does not implement interrupted-run resume.
        write_json(self.directory / "rng_state.json", {
            "generation": self.generation, "variation": self.rng.bit_generator.state,
            "adaptation": self.adaptation_rng.bit_generator.state})
        return population

    def step(self):
        self.ea.step()
        expected = self.settings.population + self.generation * self.settings.offspring
        if self.evaluations != expected:
            raise AssertionError(f"Budget mismatch: {self.evaluations} != {expected}")

    def plateau(self):
        window = self.settings.patience
        if len(self.history) <= window:
            return False
        improvement = self.history[-1 - window]["best_so_far"] - self.history[-1]["best_so_far"]
        return improvement <= self.settings.plateau_tolerance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--replay", type=Path, help="Replay one saved best.json headlessly.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--population", type=int)
    parser.add_argument("--offspring", type=int)
    parser.add_argument("--generations", type=int, help="Maximum generations, not evidence of convergence.")
    parser.add_argument("--min-generations", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--workers", type=int, default=1,
                        help="Separate simulation processes (e.g. 4); selection and logging remain serial.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.replay:
        best = json.loads(args.replay.read_text(encoding="utf-8"))
        simulator = Simulator(Settings(**best["settings"]))
        if simulator.xml_hash != best["model_xml_sha256"]:
            raise RuntimeError("Robot/world XML differs from training; check your ARIEL version/settings.")
        result = simulator.evaluate(best["weights"], record=True)
        path = args.replay.with_name("best_trajectory.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "x", "y", "z"])
            writer.writerows(result["trajectory"])
        print(f"Stored fitness: {best['fitness']:.8f}; replay: {result['fitness']:.8f}")
        print(f"Trajectory saved: {path.resolve()}")
        return

    options = asdict(Settings())
    if args.full:
        options.update(pilot=False, population=20, offspring=40,
                       max_generations=150, min_generations=50)
    for arg, key in [("population", "population"), ("offspring", "offspring"),
                     ("generations", "max_generations"), ("min_generations", "min_generations"),
                     ("patience", "patience"), ("duration", "duration")]:
        value = getattr(args, arg)
        if value is not None:
            options[key] = value
    if args.pilot:
        options["min_generations"] = options["max_generations"]
    cfg = Settings(**options)
    seeds = args.seeds if args.seeds is not None else ([11, 22] if args.pilot else list(range(101, 111)))
    if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    if min(cfg.population, cfg.offspring) < 2 or cfg.max_generations < 1 or cfg.duration <= 0:
        raise ValueError("Population/offspring >=2, generations >=1 and duration >0 required.")
    if cfg.patience < 1 or not 1 <= cfg.min_generations <= cfg.max_generations:
        raise ValueError("Require patience>=1 and 1<=min_generations<=max_generations.")
    if args.full and len(seeds) < 5:
        raise ValueError("Final experiments require at least five independent seeds.")
    root = (args.output or Path("__data__") / ("a2_pilot" if args.pilot else "a2_full")).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output is not empty: {root}. Use a NEW output folder.")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    simulator = Simulator(cfg)
    (root / "model.xml").write_text(simulator.model_xml, encoding="utf-8")
    shutil.copyfile(__file__, root / "experiment_source.py")
    analysis_source = Path(__file__).with_name("A2_analyse.py")
    if analysis_source.exists():
        shutil.copyfile(analysis_source, root / "analysis_source.py")
    versions = {}
    for name in ("numpy", "mujoco", "ariel", "scipy", "sqlmodel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown/source checkout"
    manifest = {"settings": asdict(cfg), "seeds": seeds, "methods": METHODS,
                "workers": args.workers, "logical_cpus": os.cpu_count(),
                "python": sys.version, "platform": platform.platform(), "versions": versions,
                "controller_inputs": simulator.inputs, "controller_outputs": simulator.outputs,
                "neural_parameters": simulator.dimension,
                "adaptive_tau": 1 / math.sqrt(simulator.dimension),
                "physics_timestep": simulator.model.opt.timestep,
                "physics_steps_per_evaluation": simulator.steps,
                "model_xml_sha256": simulator.xml_hash,
                "body": "john_set.gecko", "world": "SimpleFlatWorld",
                "algorithm": "tournament parents -> Gaussian mutation -> evaluate -> (mu+lambda) survivors",
                "primary_comparison_generation": cfg.min_generations,
                "primary_comparison_evaluations": cfg.population + cfg.min_generations * cfg.offspring,
                "stopping": "pilot: fixed short run; full: all EA runs plateau, or common generation cap",
                "status": "running"}
    write_json(root / "setup.json", manifest)
    print(f"{'PILOT ONLY' if cfg.pilot else 'EXPERIMENT'} | {len(seeds)} seeds x {len(METHODS)} methods | "
          f"workers={args.workers}", flush=True)
    print(f"Population={cfg.population}, offspring={cfg.offspring}, "
          f"maximum budget/run={cfg.population + cfg.max_generations * cfg.offspring}", flush=True)
    print(f"Output: {root}", flush=True)

    # Technical repeatability check, excluded from all search budgets/results.
    probe = np.random.default_rng(8301).normal(0, cfg.initial_sd, simulator.dimension)
    probe_a, probe_b = simulator.evaluate(probe), simulator.evaluate(probe)
    if probe_a["invalid"] or probe_b["invalid"] or abs(probe_a["fitness"] - probe_b["fitness"]) > 1e-9:
        raise RuntimeError("Repeatability check failed: identical controller did not reproduce valid fitness.")
    manifest["repeatability_check"] = {"fitness_a": probe_a["fitness"], "fitness_b": probe_b["fitness"],
                                        "excluded_from_search_budget": True}
    write_json(root / "setup.json", manifest)

    runs = []
    stop_reason = "pilot_generation_limit" if cfg.pilot else "maximum_generations_reached"
    try:
        simulator.start_workers(args.workers)
        for seed in seeds:
            for method in METHODS:
                run = SearchRun(method, seed, cfg, simulator, root)
                runs.append(run)
        for seed in seeds:
            matching = [r for r in runs if r.seed == seed]
            if len({r.initial_weights_hash for r in matching}) != 1:
                raise AssertionError("Initial populations were not matched.")
            if not all(r.history[0]["best_so_far"] == matching[0].history[0]["best_so_far"] for r in matching):
                raise AssertionError("Identical initial controllers gave different fitness.")

        means = {m: float(np.mean([r.best["fitness"] for r in runs if r.method == m])) for m in METHODS}
        print(f"generation=  0 evaluations/run={cfg.population} | "
              + " | ".join(f"{m}={v:.4f}" for m, v in means.items()), flush=True)

        for generation in range(1, cfg.max_generations + 1):
            for run in runs:
                run.step()
            if cfg.pilot or generation % 5 == 0:
                means = {m: float(np.mean([r.best["fitness"] for r in runs if r.method == m])) for m in METHODS}
                print(f"generation={generation:3d} evaluations/run={cfg.population + generation * cfg.offspring} | "
                      + " | ".join(f"{m}={v:.4f}" for m, v in means.items()), flush=True)
            if (not cfg.pilot and generation >= cfg.min_generations
                    and all(r.plateau() for r in runs if r.method != "random")):
                stop_reason = "all_ea_runs_plateaued"
                break
        manifest.update(status="completed", stop_reason=stop_reason,
                        generations_completed=generation,
                        evaluations_per_run=cfg.population + generation * cfg.offspring,
                        elapsed_seconds=time.perf_counter() - started,
                        mean_evaluation_seconds=simulator.total_evaluation_seconds / simulator.calls,
                        search_batch_wall_seconds=simulator.batch_wall_seconds,
                        search_evaluation_count=simulator.batch_evaluations,
                        wall_seconds_per_search_evaluation=simulator.batch_wall_seconds / simulator.batch_evaluations,
                        total_invalid_evaluations=sum(r.invalid_evaluations for r in runs))
        for run in runs:
            append_row(root / "final_runs.csv", {
                **run.history[-1], "plateau_at_stop": run.plateau(),
                "best_generation": run.best["generation_found"],
                "initial_weights_sha256": run.initial_weights_hash})
        write_json(root / "setup.json", manifest)
    except BaseException as exc:
        manifest.update(status="interrupted_or_failed", error_type=type(exc).__name__,
                        elapsed_seconds=time.perf_counter() - started)
        write_json(root / "setup.json", manifest)
        raise
    finally:
        mj.set_mjcb_control(None)
        simulator.close_workers()
        for run in runs:
            run.ea.engine.dispose()

    print(f"\nStop: {stop_reason} | generation={generation} | "
          f"evaluations/run={manifest['evaluations_per_run']} | "
          f"search elapsed={manifest['elapsed_seconds'] / 60:.1f} min", flush=True)
    from A2_analyse import analyse
    analyse(root)
    if cfg.pilot:
        estimate = manifest["wall_seconds_per_search_evaluation"] * (15.0 / cfg.duration) * (20 + 150 * 40) * 40 / 3600
        print(f"Estimated full evaluation time at cap: 8 seeds ~{estimate * 0.8:.2f}h; "
              f"10 seeds ~{estimate:.2f}h (plus logging/analysis).")
    elif stop_reason == "maximum_generations_reached":
        print("Generation cap reached; some EA runs were still improving.")


if __name__ == "__main__":
    main()
