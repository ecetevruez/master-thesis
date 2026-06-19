"""
NSGA-II vs AWS (single-layer McCormick) — Hypervolume comparison for MOUC.

AWS HV: computed from the single Pareto point per n in Table 7
        (point-wise HV) using the thesis reference point scaled by n.
        This covers n = 540–11340 without requiring new MATLAB/Gurobi runs.

NSGA-II HV: reported at time checkpoints.
        Table 1 checkpoints: t = 5, 15, 25, 35, 70 s  (n = 540–7020)
        Table 2 checkpoints: t = 45, 55, 65, 75, 85, 95, 105, 200 s  (n = 7560–11340)
        The AWS runtime always falls between two consecutive checkpoints.

Reference point: thesis formula scaled by n.
Fixed seed: 42.
"""

import numpy as np
import time
import re
import csv
from datetime import datetime

from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV


# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------

def parse_mod_file(filename, T_override=None):
    with open(filename) as f:
        lines = [l.rstrip('\n') for l in f]

    data = {}
    i = 0
    thermal_rows, hydro_rows, hydro_ts_rows = [], [], []
    parsing_thermal = parsing_hydro = False

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('HorizonLen'):
            data['T'] = int(re.search(r'\d+', line).group())
        elif line.startswith('NumThermal'):
            data['NumThermal'] = int(re.search(r'\d+', line).group())
        elif line.startswith('NumHydro'):
            data['NumHydro'] = int(re.search(r'\d+', line).group())
        elif line.startswith('Loads'):
            data['Loads'] = list(map(float, lines[i + 1].split()))
            i += 1
        elif line.startswith('ThermalSection'):
            parsing_thermal, parsing_hydro = True, False
        elif line.startswith('HydroSection'):
            parsing_thermal, parsing_hydro = False, True
        elif line.startswith('HydroCascadeSection'):
            parsing_hydro = False
        elif parsing_thermal and not line.startswith('RampConstraints'):
            nums = list(map(float, line.split()))
            if nums:
                thermal_rows.append(nums)
        elif parsing_hydro:
            nums = list(map(float, line.split()))
            if len(nums) == 8:
                hydro_rows.append(nums)
            elif len(nums) > 8:
                hydro_ts_rows.append(nums)
        i += 1

    T_full = data['T']
    T = T_override if T_override else T_full

    thermal  = np.array(thermal_rows)
    hydro    = np.array(hydro_rows)
    hydro_ts = np.array(hydro_ts_rows)

    if T > T_full:
        reps = (T // T_full) + 1
        data['Loads'] = list(np.tile(np.array(data['Loads']), reps)[:T])
        hydro_ts = np.tile(hydro_ts, (1, reps))[:, :T]

    quad_c  = thermal[:, 1];  lin_c   = thermal[:, 2]
    minP_t  = thermal[:, 4];  maxP_t  = thermal[:, 5]
    minUp   = thermal[:, 7].astype(int)
    minDown = thermal[:, 8].astype(int)
    coolFC  = thermal[:, 9];  hotFC   = thermal[:, 10]
    tau     = thermal[:, 11]; fixedC  = thermal[:, 13]

    startup_cost = (coolFC * (1 - np.exp(-minDown / np.maximum(tau, 1e-9)))
                    + fixedC + hotFC * minDown + fixedC)

    vtp         = hydro[:, 1]
    mean_inflow = hydro_ts[:, :T].mean(axis=1)
    minP_h = np.zeros(hydro.shape[0])
    maxP_h = vtp * mean_inflow

    I_t, I_h = data['NumThermal'], data['NumHydro']
    I = I_t + I_h

    maxP = np.concatenate([maxP_t, maxP_h])
    minP = np.concatenate([minP_t, minP_h])

    beta_s  = np.array([2.06,2.09,2.14,2.25,2.11,3.45,2.62,5.18,5.38,5.40]*2)
    gamma_s = np.array([0.00019,0.00018,0.00220,0.00220,0.00210,0.00250,
                        0.00220,0.00420,0.00540,0.00550]*2)
    beta_c  = np.array([-2.86,-2.72,-2.94,-2.35,-2.36,-2.28,-2.36,-1.29,-1.14,-2.14]*2)
    gamma_c = np.array([0.022,0.020,0.044,0.058,0.065,0.080,0.075,0.082,0.090,0.084]*2)

    beta_em_full  = np.concatenate([beta_s + beta_c,   np.zeros(I_h)])
    gamma_em_full = np.concatenate([gamma_s + gamma_c, np.zeros(I_h)])

    return {
        'T': T, 'I': I, 'I_t': I_t, 'I_h': I_h,
        'maxP': maxP, 'minP': minP,
        'quad': np.concatenate([quad_c, np.zeros(I_h)]),
        'lin':  np.concatenate([lin_c,  np.zeros(I_h)]),
        'startup': np.concatenate([startup_cost, np.zeros(I_h)]),
        'beta_em': beta_em_full, 'gamma_em': gamma_em_full,
        'minUp':   np.concatenate([minUp,   np.zeros(I_h, dtype=int)]),
        'minDown': np.concatenate([minDown, np.zeros(I_h, dtype=int)]),
        'loads': np.array(data['Loads'][:T]),
        'Fscal': 899.0,
    }


# ---------------------------------------------------------------------------
# Demand repair with min-up / min-down
# ---------------------------------------------------------------------------

class DemandRepair(Repair):
    def __init__(self, params):
        super().__init__()
        self.p = params

    def _do(self, problem, X, **kwargs):
        p = self.p
        T, I = p['T'], p['I']
        n_g = I * T
        minUp, minDown = p['minUp'], p['minDown']

        for k in range(X.shape[0]):
            g = X[k, :n_g].reshape(I, T).copy()

            for t in range(T):
                deficit = p['loads'][t] - np.sum(g[:, t])
                if deficit > 1e-6:
                    headroom = p['maxP'] - g[:, t]
                    total_hw = np.sum(headroom)
                    if total_hw > 1e-9:
                        g[:, t] += headroom * min(1.0, deficit / total_hw)

            z = (g > 1e-6).astype(float)

            for i in range(I):
                mu, md = minUp[i], minDown[i]
                for t in range(T):
                    prev = z[i, t - 1] if t > 0 else 0.0
                    if z[i, t] == 1.0 and prev == 0.0 and mu >= 2:
                        z[i, t:min(t + mu, T)] = 1.0
                    elif z[i, t] == 0.0 and prev == 1.0 and md >= 2:
                        z[i, t:min(t + md, T)] = 0.0

            g = np.clip(g, p['minP'][:, None] * z, p['maxP'][:, None] * z)

            for t in range(T):
                deficit = p['loads'][t] - np.sum(g[:, t])
                if deficit > 1e-6:
                    headroom = p['maxP'] * z[:, t] - g[:, t]
                    total_hw = np.sum(headroom)
                    if total_hw > 1e-9:
                        g[:, t] += headroom * min(1.0, deficit / total_hw)

            X[k, :n_g] = g.reshape(n_g)
            X[k, n_g:] = z.reshape(n_g)
        return X


# ---------------------------------------------------------------------------
# MOUC problem
# ---------------------------------------------------------------------------

class MOUCProblem(Problem):
    def __init__(self, params, penalty=1e5):
        self.p = params
        self.penalty = penalty
        T, I = params['T'], params['I']
        n_var = 2 * I * T
        lb = np.zeros(n_var)
        ub = np.concatenate([np.repeat(params['maxP'], T), np.ones(I * T)])
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=T, xl=lb, xu=ub)

    def _evaluate(self, X, out, *args, **kwargs):
        p = self.p
        T, I = p['T'], p['I']
        n_g = I * T
        G  = X[:, :n_g].reshape(-1, I, T)
        Z  = np.round(X[:, n_g:].reshape(-1, I, T))
        n_pop = X.shape[0]

        q  = p['quad'][:, None];  l  = p['lin'][:, None]
        ge = p['gamma_em'][:, None]; be = p['beta_em'][:, None]

        f1 = np.sum(q * G**2 + l * G, axis=(1, 2))
        Gn = G / p['Fscal']
        f2 = np.sum(ge * Gn**2 + be * Gn, axis=(1, 2))

        if T > 1:
            dZ = np.maximum(0, Z[:, :, 1:] - Z[:, :, :-1])
            f1 += np.sum(dZ * p['startup'][:, None], axis=(1, 2))

        pen = self.penalty * (
            np.sum(np.maximum(0, G - p['maxP'][:, None] * Z), axis=(1, 2)) +
            np.sum(np.maximum(0, p['minP'][:, None] * Z - G), axis=(1, 2))
        )
        supply   = np.sum(G, axis=1)
        G_constr = p['loads'][None, :] - supply

        out['F'] = np.column_stack([f1 + pen, f2 + pen])
        out['G'] = G_constr


# ---------------------------------------------------------------------------
# HV checkpoint callback
# ---------------------------------------------------------------------------

class HVCheckpointCallback(Callback):
    """Snapshots the non-dominated front at each requested time checkpoint."""

    def __init__(self, checkpoints, t0):
        super().__init__()
        self.checkpoints = sorted(checkpoints)
        self._fronts    = {}
        self._triggered = set()
        self.t0         = t0

    def notify(self, algorithm):
        elapsed = time.time() - self.t0

        try:
            F = algorithm.opt.get("F")
        except Exception:
            F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        for t in self.checkpoints:
            if t not in self._triggered and elapsed >= t:
                self._fronts[t] = F.copy()
                self._triggered.add(t)

    def compute_hv(self, ref_point, final_F):
        """Compute HV at each checkpoint using the given reference point."""
        ind = HV(ref_point=ref_point)
        return {t: ind(self._fronts.get(t, final_F)) for t in self.checkpoints}


# ---------------------------------------------------------------------------
# Run NSGA-II with time budget
# ---------------------------------------------------------------------------

def thesis_ref_point(n):
    """Reference point from the thesis: (factor × 3307000, factor × 125)
    where factor = n/2160.  Confirmed from PyGMO script: n=2160 uses factor=1."""
    factor = n / 2160.0
    return np.array([factor * 3307000.0, factor * 125.0])


def compute_aws_single_hv(n, f1, f2):
    """Point-wise HV from the single balanced Pareto point in Table 7."""
    ref = thesis_ref_point(n)
    return (ref[0] - f1) * (ref[1] - f2)


INSTANCES_TABLE1 = [
    {'n': 540,  'aws_time':  6.87, 'aws_f1': 735203.203,   'aws_f2':   8.361},
    {'n': 1080, 'aws_time':  7.36, 'aws_f1': 1472770.926,  'aws_f2':  15.530},
    {'n': 1620, 'aws_time': 15.77, 'aws_f1': 2204594.811,  'aws_f2':  25.593},
    {'n': 2160, 'aws_time': 15.86, 'aws_f1': 2939702.566,  'aws_f2':  34.002},
    {'n': 2700, 'aws_time': 17.20, 'aws_f1': 3674927.273,  'aws_f2':  42.352},
    {'n': 3240, 'aws_time': 20.41, 'aws_f1': 4410151.981,  'aws_f2':  50.702},
    {'n': 3780, 'aws_time': 23.07, 'aws_f1': 5145376.688,  'aws_f2':  59.052},
    {'n': 4320, 'aws_time': 24.54, 'aws_f1': 5880601.472,  'aws_f2':  67.401},
    {'n': 4860, 'aws_time': 25.09, 'aws_f1': 6615826.182,  'aws_f2':  75.751},
    {'n': 5400, 'aws_time': 28.34, 'aws_f1': 7351050.874,  'aws_f2':  84.101},
    {'n': 5940, 'aws_time': 29.75, 'aws_f1': 8086275.514,  'aws_f2':  92.451},
    {'n': 6480, 'aws_time': 31.29, 'aws_f1': 8821500.273,  'aws_f2': 100.800},
    {'n': 7020, 'aws_time': 31.90, 'aws_f1': 9556725.012,  'aws_f2': 109.150},
]
CHECKPOINTS_TABLE1 = [5, 15, 25, 35, 70]

INSTANCES_TABLE2 = [
    {'n':  7560, 'aws_time':  40.56, 'aws_f1': 10291949.721, 'aws_f2': 117.500},
    {'n':  8100, 'aws_time':  50.91, 'aws_f1': 11027174.431, 'aws_f2': 125.850},
    {'n':  8640, 'aws_time':  61.70, 'aws_f1': 11761157.401, 'aws_f2': 134.826},
    {'n':  9180, 'aws_time':  67.20, 'aws_f1': 12497623.824, 'aws_f2': 142.549},
    {'n':  9720, 'aws_time':  73.70, 'aws_f1': 13232028.710, 'aws_f2': 151.313},
    {'n': 10260, 'aws_time':  86.96, 'aws_f1': 13968073.186, 'aws_f2': 159.249},
    {'n': 10800, 'aws_time':  95.01, 'aws_f1': 14701927.560, 'aws_f2': 168.290},
    {'n': 11340, 'aws_time':  99.40, 'aws_f1': 15437081.978, 'aws_f2': 176.675},
]
CHECKPOINTS_TABLE2 = [45, 55, 65, 75, 85, 95, 105, 200]

TABLE = 1
CHECKPOINTS = CHECKPOINTS_TABLE1 if TABLE == 1 else CHECKPOINTS_TABLE2


def run_nsga2_hv(params, n, checkpoints=None, pop_size=200, seed=42):
    """Run NSGA-II for max(checkpoints) seconds; return HV dict at each checkpoint."""
    if checkpoints is None:
        checkpoints = CHECKPOINTS
    problem  = MOUCProblem(params)
    repair   = DemandRepair(params)
    callback = HVCheckpointCallback(checkpoints, t0=None)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=repair,
        eliminate_duplicates=True,
    )

    budget = max(checkpoints)
    t0     = time.time()
    callback.t0 = t0
    res = minimize(problem, algorithm,
                   ('time', budget),
                   seed=seed,
                   verbose=False,
                   callback=callback)
    runtime = time.time() - t0

    ref     = thesis_ref_point(n)
    hv_dict = callback.compute_hv(ref, res.F)
    return hv_dict, runtime

if __name__ == '__main__':
    MOD_FILE = '20_10_1_w.mod'
    SEED     = 42

    print("NSGA-II vs AWS — Hypervolume comparison  (seed=42)")
    instances   = INSTANCES_TABLE1 if TABLE == 1 else INSTANCES_TABLE2
    checkpoints = CHECKPOINTS_TABLE1 if TABLE == 1 else CHECKPOINTS_TABLE2
    print(f"AWS HV: point-wise from Table 7.  NSGA-II HV: at checkpoints {checkpoints} s.")
    est_min     = len(instances) * max(checkpoints) // 60
    print(f"Table {TABLE}: {len(instances)} instances, "
          f"checkpoints={checkpoints}, "
          f"budget={max(checkpoints)}s (~{est_min} min)\n")

    cw = 13
    cp_headers = "".join(f"  {'NSGA@'+str(t)+'s':>{cw}}" for t in checkpoints)
    header = f"  {'n':>6}  {'AWS Gurobi(s)':>{cw}}  {'AWS HV(×10^7)':>{cw}}" + cp_headers
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_rows = []

    for inst in instances:
        n        = inst['n']
        aws_time = inst['aws_time']
        aws_hv   = compute_aws_single_hv(n, inst['aws_f1'], inst['aws_f2'])
        T        = n // (3 * 30)
        params   = parse_mod_file(MOD_FILE, T_override=T)

        print(f"  n={n} (T={T}): running NSGA-II for {max(checkpoints)}s ...", end=' ', flush=True)
        hv_dict, rt = run_nsga2_hv(params, n=n, checkpoints=checkpoints, seed=SEED)
        print(f"done in {rt:.1f}s")

        nsga_hvs = [hv_dict[t] / 1e7 for t in checkpoints]
        print(f"  {n:>6}  {aws_time:>{cw}.2f}  {aws_hv/1e7:>{cw}.4f}"
              + "".join(f"  {v:>{cw}.4f}" for v in nsga_hvs))

        row = {
            'n':          n,
            'aws_time_s': aws_time,
            'aws_hv':     f"{aws_hv:.6e}",
        }
        for t in checkpoints:
            row[f'nsga2_hv_{t}s'] = f"{hv_dict[t]:.6e}"
        all_rows.append(row)

    print(sep)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f'nsga2_hv_table{TABLE}_{ts}.csv'
    with open(csv_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nResults saved to {csv_file}")
