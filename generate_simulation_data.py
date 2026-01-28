"""
Generate simulation data using dynstall_combined models.
Outputs data in the same format as the Bladed results for use with fourier_analysis.py.

Output format: # time[s], AoA[rad], cl[-], cd[-], cm[-]
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add the mdm3-Project2 folder to path to import dynstall_combined
MDM3_DIR = Path(__file__).resolve().parent.parent / "mdm3-Project2"
sys.path.insert(0, str(MDM3_DIR))

from dynstall_combined import (
    AeroDynPolar,
    NoDynamicStall,
    OyeDynamicStallCL,
    LeishmanBeddoesFirstOrder,
    IAGDynamicStallCL,
)

# -----------------------------
# PATHS
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = (
    REPO_ROOT
    / "Dataset"
    / "EdgewiseExcited_IEA15MW_S809"
    / "AeroParameters_at_93m"
)

# Our model data goes into a parallel structure (in fourier analysis repo)
OUR_MODEL_DIR = (
    REPO_ROOT
    / "Dataset"
    / "Our_DynStall_Model"
    / "AeroParameters_at_93m"
)

# Also save to mdm3-Project2 folder
OUR_MODEL_DIR_MDM3 = (
    MDM3_DIR
    / "Our_DynStall_Model"
    / "AeroParameters_at_93m"
)

# Polar file location
POLAR_FILE = MDM3_DIR / "S809_Cln.dat"

# -----------------------------
# SIMULATION PARAMETERS
# (Extracted from the existing Bladed data)
# -----------------------------

# Time parameters (matching Bladed output: 10003 points at dt=0.01s = 100.02s)
DT = 0.01  # seconds
T_END = 100.0  # seconds

# IEA 15MW turbine blade chord at 93m span station (approximate)
# Typical chord at 93% span for IEA 15MW is ~2-3m
CHORD = 2.5  # meters

# Case configurations
# Each case has: (excitation_freq_hz, alpha_mean_deg, alpha_amp_deg)


CASE_CONFIGS = {
    # OSU_Frequency case: lower frequencies, moderate alpha oscillation
    "Polar_Instability_OSU_Frequency": {
        # Frequency scales with wind speed: k = pi*f*c/U is constant ~0.079 (OSU reduced frequency)
        # For S809 at Re=1M: k=0.079 -> f = k*U/(pi*c)
        "10ms": {
            "U": 10.0,  # m/s wind speed
            "freq_hz": 0.09,  # ~0.09 Hz from FFT
            "alpha_mean_deg": 18.5,  # mean AoA ~18.5 deg
            "alpha_amp_deg": 10.5,  # amplitude ~10.5 deg (range 8-29 deg)
        },
        "30ms": {
            "U": 30.0,
            "freq_hz": 0.27,  # scales with wind
            "alpha_mean_deg": 18.5,
            "alpha_amp_deg": 10.5,
        },
        "50ms": {
            "U": 50.0,
            "freq_hz": 0.45,
            "alpha_mean_deg": 18.5,
            "alpha_amp_deg": 10.5,
        },
    },
    # Resonance case: blade edgewise resonance frequency ~0.69 Hz, larger amplitude
    "Polar_Instability_Resonance": {
        "10ms": {
            "U": 10.0,
            "freq_hz": 0.69,  # edgewise natural frequency
            "alpha_mean_deg": 25.0,  # higher mean, larger amplitude
            "alpha_amp_deg": 33.0,  # range ~3-70 deg in flex case
        },
        "30ms": {
            "U": 30.0,
            "freq_hz": 0.69,
            "alpha_mean_deg": 25.0,
            "alpha_amp_deg": 33.0,
        },
        "50ms": {
            "U": 50.0,
            "freq_hz": 0.69,
            "alpha_mean_deg": 25.0,
            "alpha_amp_deg": 33.0,
        },
    },
}

# Map our model names to fourier_analysis naming convention
# fourier_analysis uses: "None", "Oye2", "IAGModel", "BeddoesIncomp"
# Our models map to:
MODEL_MAP = {
    "None": lambda c: NoDynamicStall(),
    "Oye2": lambda c: OyeDynamicStallCL(c=c, Tf0=6.0, use_bergami_cd=True),
    "BeddoesIncomp": lambda c: LeishmanBeddoesFirstOrder(c=c, alpha_crit_rad=np.deg2rad(15.0)),
    "IAGModel": lambda c: IAGDynamicStallCL(c=c, ks=0.06, alpha_crit_rad=np.deg2rad(15.0)),
}

# Our model doesn't simulate structural dynamics (blade flexibility)
# so we only output "Rigid" - prescribed kinematics only
STRUCTS = ["Rigid"]


def run_simulation(
    model,
    polar,
    dt: float,
    t_end: float,
    alpha_mean_deg: float,
    alpha_amp_deg: float,
    freq_hz: float,
    U: float,
):
    """
    Run a sinusoidal pitch simulation and return time series.
    
    Returns: dict with keys t, alpha_rad, cl, cd, cm
    """
    t = np.arange(0.0, t_end + dt, dt)
    n = len(t)
    
    omega = 2.0 * np.pi * freq_hz
    
    # Sinusoidal angle of attack
    alpha_deg = alpha_mean_deg + alpha_amp_deg * np.sin(omega * t)
    alpha_dot_deg = omega * alpha_amp_deg * np.cos(omega * t)
    
    alpha_rad = np.deg2rad(alpha_deg)
    alpha_dot_rad = np.deg2rad(alpha_dot_deg)
    
    # Reset model state
    model.reset()
    
    # Preallocate
    cl = np.zeros(n)
    cd = np.zeros(n)
    cm = np.zeros(n)
    
    # Run simulation
    for i in range(n):
        out = model.step(alpha_rad[i], alpha_dot_rad[i], U, dt, polar)
        cl[i] = out["cl"]
        cd[i] = out["cd"]
        cm[i] = out.get("cm", 0.0)
    
    return {
        "t": t,
        "alpha_rad": alpha_rad,
        "cl": cl,
        "cd": cd,
        "cm": cm,
    }


def save_dat_file(filepath: Path, t, alpha_rad, cl, cd, cm):
    """
    Save data in the format expected by fourier_analysis.py:
    # time[s], AoA[rad], cl[-], cd[-], cm[-]
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = np.column_stack([t, alpha_rad, cl, cd, cm])
    np.savetxt(
        filepath,
        data,
        fmt="%.6f",
        header="time[s], AoA[rad], cl[-], cd[-], cm[-]"
    )
    print(f"  Saved: {filepath}")


def main():
    print("=" * 60)
    print("Dynamic Stall Simulation Data Generator")
    print("=" * 60)
    print()
    
    # Load polar
    if not POLAR_FILE.exists():
        print(f"ERROR: Polar file not found: {POLAR_FILE}")
        print("Please ensure S809_Cln.dat exists in mdm3-Project2 folder.")
        return
    
    polar = AeroDynPolar.from_file(str(POLAR_FILE))
    print(f"Loaded polar: {POLAR_FILE.name}")
    print(f"  Alpha range: {polar.alpha_deg.min():.1f} to {polar.alpha_deg.max():.1f} deg")
    print(f"  CL range: {polar.cl_data.min():.3f} to {polar.cl_data.max():.3f}")
    print()
    
    print(f"Simulation parameters:")
    print(f"  dt = {DT} s")
    print(f"  t_end = {T_END} s")
    print(f"  chord = {CHORD} m")
    print()
    
    print(f"Output directory: {OUR_MODEL_DIR}")
    print()
    
    # Count files
    total_files = 0
    
    for case_name, wind_configs in CASE_CONFIGS.items():
        print(f"\nCase: {case_name}")
        print("-" * 40)
        
        for wind_name, cfg in wind_configs.items():
            print(f"\n  Wind: {wind_name} (U={cfg['U']} m/s, f={cfg['freq_hz']:.3f} Hz)")
            print(f"    Alpha: {cfg['alpha_mean_deg']:.1f} ± {cfg['alpha_amp_deg']:.1f} deg")
            
            for struct in STRUCTS:
                for model_name, model_factory in MODEL_MAP.items():
                    # Create model instance
                    model = model_factory(CHORD)
                    
                    # Run simulation
                    result = run_simulation(
                        model=model,
                        polar=polar,
                        dt=DT,
                        t_end=T_END,
                        alpha_mean_deg=cfg["alpha_mean_deg"],
                        alpha_amp_deg=cfg["alpha_amp_deg"],
                        freq_hz=cfg["freq_hz"],
                        U=cfg["U"],
                    )
                    
                    # Build output path (matches Bladed structure)
                    fname = f"{struct}_{model_name}.dat"
                    fpath = OUR_MODEL_DIR / case_name / wind_name / fname
                    
                    # Save
                    save_dat_file(
                        fpath,
                        result["t"],
                        result["alpha_rad"],
                        result["cl"],
                        result["cd"],
                        result["cm"],
                    )
                    
                    # Also save to mdm3-Project2 folder
                    fpath_mdm3 = OUR_MODEL_DIR_MDM3 / case_name / wind_name / fname
                    save_dat_file(
                        fpath_mdm3,
                        result["t"],
                        result["alpha_rad"],
                        result["cl"],
                        result["cd"],
                        result["cm"],
                    )
                    
                    total_files += 1
    
    print()
    print("=" * 60)
    print(f"Done! Generated {total_files} files (saved to both locations).")
    print(f"Output directory 1: {OUR_MODEL_DIR}")
    print(f"Output directory 2: {OUR_MODEL_DIR_MDM3}")
    print()
    print("To use with fourier_analysis.py, run:")
    print("  python fourier_analysis.py --source our_model")


if __name__ == "__main__":
    main()
