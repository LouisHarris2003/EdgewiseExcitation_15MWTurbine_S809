from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks

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

OUT_DIR = REPO_ROOT / "moving_fft_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ_CSV = OUT_DIR / "excitation_frequencies.csv"


# -----------------------------
# CASES / OPTIONS
# -----------------------------
CASES = [
    "Polar_Instability_OSU_Frequency",
    "Polar_Instability_Resonance",
]
WINDS = ["10ms", "30ms", "50ms"]
STRUCTS = ["Flex", "Rigid"]
MODELS = ["None", "Oye2", "IAGModel", "BeddoesIncomp"]

# FFT pick range (Hz) for f_e detection from alpha
FMAX_FFT_PICK = 5.0

# Plot cap fallback if auto cap fails
FMAX_PLOT_FALLBACK = 10.0

#stft settings defined in seconds 
# Needs optimization for values
WINDOW_SEC = 15 #seconds per stft window
OVERLAP_SEC = 14 #seconds overlap
# WINDOW_SEC = 5 #seconds per stft window
# OVERLAP_SEC = 4.5 #seconds overlap


#log power epsilon floor
EPS = 1e-12

#fixed colour scale for comparisons (edit so use -70 and -10 for no epsilon)
COLOUR_LIM_MIN = -70
COLOUR_LIM_MAX = -10


# -----------------------------
# NORMALISATION CONSTANTS
# -----------------------------
ALPHA_AMPLITUDE_DEG = 10.45  # degrees
ALPHA_AMPLITUDE_RAD = ALPHA_AMPLITUDE_DEG * np.pi / 180
ALPHA_NORM = ALPHA_AMPLITUDE_RAD
CL_NORM = 2 * np.pi * ALPHA_AMPLITUDE_RAD
CD_NORM = CL_NORM * np.sin(ALPHA_AMPLITUDE_RAD)


# -----------------------------
# DATA LOADING
# -----------------------------
def load_dat_file(filepath: Path):
    """
    File format:
    # time[s], AoA[rad], cl[-], cd[-], cm[-]
    """
    data = np.loadtxt(filepath, comments="#")
    t = data[:, 0]
    alpha = data[:, 1]
    cl = data[:, 2]
    cd = data[:, 3]
    return t, alpha, cl, cd


def infer_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError(f"Bad time step in data (dt={dt})")
    return 1.0 / dt

# -----------------------------
# NORMALIsATION
# -----------------------------
def normalise_data(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray):
    """
    Normalise alpha, cl, and cd by their respective normalisation constants.
    
    Args:
        alpha: Angle of attack in radians
        cl: Lift coefficient
        cd: Drag coefficient
    
    Returns:
        Normalised alpha, cl, cd
    """
    alpha_normalised = alpha / ALPHA_NORM
    cl_normalised = cl / CL_NORM
    cd_normalised = cd / CD_NORM
    
    return alpha_normalised, cl_normalised, cd_normalised
# -----------------------------
# EXCITATION FREQUENCY PICKER
# -----------------------------
def estimate_excitation_frequency_from_alpha(
    t: np.ndarray,
    alpha: np.ndarray,
    fmax: float = 5.0,
    min_prom_frac: float = 0.1,
):
    """
    f_e estimate from alpha FFT:
      - Computes FFT magnitude
      - Finds significant peaks using prominence threshold
      - Picks the LOWEST-frequency significant peak (fundamental forcing)- check by galih this isnt the highest but seems like a more robust method
      - Fallbacks to max peak if no peaks found

    Returns:
      fe (float) or None if cannot be determined
    """
    dt = np.median(np.diff(t))
    if dt <= 0:
        return None

    # de-mean to reduce DC dominance
    a = alpha - np.mean(alpha)

    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(len(a), d=dt)
    mag = np.abs(A)

    mask = (f > 0) & (f <= fmax)
    f2 = f[mask]
    m2 = mag[mask]

    if len(f2) < 3:
        return None

    prom = min_prom_frac * np.max(m2) if np.max(m2) > 0 else 0.0
    peaks, _ = find_peaks(m2, prominence=prom)

    if len(peaks) == 0:
        # fallback: strongest peak
        return float(f2[np.argmax(m2)])

    # choose lowest significant peak = fundamental forcing frequency
    return float(f2[peaks[0]])


def auto_fmax_plot(fe: float | None) -> float:
    """
    Auto-scale plot frequency cap:
      - show up to ~6 harmonics or at least 1 Hz
      - cap at fallback max
    """
    if fe is None or fe <= 0:
        return FMAX_PLOT_FALLBACK
    return min(FMAX_PLOT_FALLBACK, max(1.0, 6.0 * fe))


# -----------------------------
# STFT PARAM CONVERSION FROM SECONDS TO SAMPLES
# -----------------------------

def stft_params_from_seconds(fs: float, window_sec: float, overlap_sec: float):
    """
    Convert time-based STFT settings to sample counts.
    Ensures:
      - nperseg >= 8
      - 0 <= noverlap < nperseg
    """
    nperseg = int(round(window_sec * fs))
    noverlap = int(round(overlap_sec * fs))

    nperseg = max(nperseg, 8)
    noverlap = max(noverlap, 0)
    if noverlap >= nperseg:
        noverlap = nperseg - 1
        
    return nperseg, noverlap

# -----------------------------
# MOVING FFT (STFT)
# -----------------------------
def moving_fft(x: np.ndarray, fs: float, nperseg: int, noverlap: int):
    """
    Returns:
      f (Hz), tt (s), Z (log power dB)
    """
    f, tt, Zxx = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        boundary=None,
        padded=False,
        nfft = nperseg*5,
    )

    P = np.abs(Zxx) ** 2
    Z = 10.0 * np.log10(P + EPS)
    
    return f, tt, Z


def plot_time_freq(
    f: np.ndarray,
    tt: np.ndarray,
    Z: np.ndarray,
    title: str,
    outpath: Path,
    fmax: float,
    freq_lines=None,
    vmin: float = COLOUR_LIM_MIN,
    vmax: float = COLOUR_LIM_MAX,
):
    """
    freq_lines: list of (frequency_hz, label) tuples
    """
    if fmax is not None:
        m = f <= fmax
        f = f[m]
        Z = Z[m, :]

    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    # cmap = 'YlOrBr'
    # cmap = 'YlOrBr_r'
    # cmap = 'hot'
    # cmap = 'cividis'
    # cmap = 'pink'
    cmap = 'magma'
    plt.pcolormesh(tt, f, Z, shading="auto",cmap=cmap, vmin = vmin, vmax = vmax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)

    cbar = plt.colorbar()
    cbar.set_label("Log Power (dB)")

    if freq_lines:
        y_max = f[-1] if len(f) else None
        x0 = tt[0] if len(tt) else 0.0
        for fr, label in freq_lines:
            if y_max is None or fr > y_max:
                continue
            plt.axhline(fr, linestyle="--", linewidth=1)
            plt.text(x0, fr, f" {label}", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_combined_time_freq(
    f: np.ndarray,
    tt: np.ndarray,
    Z_alpha: np.ndarray,
    Z_cl: np.ndarray,
    Z_cd: np.ndarray,
    case: str,
    wind: str,
    struct: str,
    model: str,
    outpath: Path,
    fmax: float,
    freq_lines=None,
    vmin: float = COLOUR_LIM_MIN,
    vmax: float = COLOUR_LIM_MAX,
):
    """
    Create a combined plot with alpha, cl, and cd time-frequency plots stacked vertically.
    All three parameters are normalized.
    
    freq_lines: list of (frequency_hz, label) tuples
    """
    if fmax is not None:
        m = f <= fmax
        f = f[m]
        Z_alpha = Z_alpha[m, :]
        Z_cl = Z_cl[m, :]
        Z_cd = Z_cd[m, :]

    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    cmap = 'magma'
    
    # Alpha plot
    im0 = axes[0].pcolormesh(tt, f, Z_alpha, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title(f"{case} | {wind} | {struct} | {model} | alpha (normalised)")
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label("Log Power (dB)")
    
    # Cl plot
    im1 = axes[1].pcolormesh(tt, f, Z_cl, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"cl (normalised)")
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label("Log Power (dB)")
    
    # Cd plot
    im2 = axes[2].pcolormesh(tt, f, Z_cd, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title(f"cd (normalised)")
    cbar2 = plt.colorbar(im2, ax=axes[2])
    cbar2.set_label("Log Power (dB)")
    
    # Add frequency lines to all subplots
    if freq_lines:
        y_max = f[-1] if len(f) else None
        x0 = tt[0] if len(tt) else 0.0
        for ax in axes:
            for fr, label in freq_lines:
                if y_max is None or fr > y_max:
                    continue
                ax.axhline(fr, linestyle="--", linewidth=1, color='white', alpha=0.5)
                # Only add label to first subplot
                if ax == axes[0]:
                    ax.text(x0, fr, f" {label}", va="bottom", color='white')

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Current working directory:", Path.cwd())
    print("Dataset dir:", DATASET_DIR.resolve())
    print("Output dir:", OUT_DIR.resolve())
    print()
    print(f"STFT window_sec={WINDOW_SEC:.3f}s overlap_sec={OVERLAP_SEC:.3f}s EPS={EPS:g}")
    print(f"ColoUr scale vmin={COLOUR_LIM_MIN} vmax={COLOUR_LIM_MAX}")
    print()
    print(f"Normalisation constants:")
    print(f"  ALPHA_NORM = {ALPHA_NORM:.6f} rad ({ALPHA_AMPLITUDE_DEG}°)")
    print(f"  CL_NORM = {CL_NORM:.6f}")
    print(f"  CD_NORM = {CD_NORM:.6f}")
    print()

    csv_rows = [["case", "wind", "struct", "model", "file", "fe_hz", "fmax_plot_hz"]]

    wrote_anything = False
    missing_count = 0

    for case in CASES:
        for wind in WINDS:
            for struct in STRUCTS:
                for model in MODELS:
                    fname = f"{struct}_{model}.dat"
                    fpath = DATASET_DIR / case / wind / fname

                    if not fpath.exists():
                        missing_count += 1
                        continue

                    print("Processing:", fpath)

                    t, alpha, cl, cd = load_dat_file(fpath)
                    fs = infer_fs(t)

                    # NORMALIsE the data
                    alpha, cl, cd = normalise_data(alpha, cl, cd)

                    # reduce 0 Hz dominance by minusing the mean
                    alpha = alpha - np.mean(alpha)
                    cl = cl - np.mean(cl)
                    cd = cd - np.mean(cd)

                    # robust f_e by finding lowest significant FFT peak
                    fe = estimate_excitation_frequency_from_alpha(
                        t, alpha, fmax=FMAX_FFT_PICK, min_prom_frac=0.1
                    )

                    # auto frequency cap per file (so 0.09 Hz cases are readable)
                    fmax_plot = auto_fmax_plot(fe)

                    # Lines at f_e, 2f_e, 3f_e
                    freq_lines = None
                    if fe is not None:
                        freq_lines = [
                            (fe, f"f_e={fe:.3f}"),
                            (2 * fe, "2 f_e"),
                            (3 * fe, "3 f_e"),
                        ]

                    # time-based STFT parameters
                    nperseg, noverlap = stft_params_from_seconds(
                        fs=fs, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC
                    )

                    # STFT
                    f, tt, Z_alpha = moving_fft(alpha, fs, nperseg=nperseg, noverlap=noverlap)
                    _, _, Z_cl = moving_fft(cl, fs, nperseg=nperseg, noverlap=noverlap)
                    _, _, Z_cd = moving_fft(cd, fs, nperseg=nperseg, noverlap=noverlap)

                    base = OUT_DIR / case / wind / struct / model

                    plot_time_freq(
                        f, tt, Z_alpha,
                        title=f"{case} | {wind} | {struct} | {model} | alpha (normalised)",
                        outpath=base / "alpha_timefreq.png",
                        fmax=fmax_plot,
                        freq_lines=freq_lines
                    )
                    plot_time_freq(
                        f, tt, Z_cl,
                        title=f"{case} | {wind} | {struct} | {model} | cl (normalised)",
                        outpath=base / "cl_timefreq.png",
                        fmax=fmax_plot,
                        freq_lines=freq_lines
                    )
                    plot_time_freq(
                        f, tt, Z_cd,
                        title=f"{case} | {wind} | {struct} | {model} | cd (normalised)",
                        outpath=base / "cd_timefreq.png",
                        fmax=fmax_plot,
                        freq_lines=freq_lines
                    )

                    # Create combined plot with all three parameters
                    plot_combined_time_freq(
                        f, tt, Z_alpha, Z_cl, Z_cd,
                        case=case,
                        wind=wind,
                        struct=struct,
                        model=model,
                        outpath=base / "combined_timefreq.png",
                        fmax=fmax_plot,
                        freq_lines=freq_lines
                    )

                    csv_rows.append([
                        case,
                        wind,
                        struct,
                        model,
                        str(fpath),
                        "" if fe is None else f"{fe:.6f}",
                        f"{fmax_plot:.3f}",
                        f"{fs:.6f}",
                        str(nperseg),
                        str(noverlap),
                        f"{WINDOW_SEC:.3f}",
                        f"{OVERLAP_SEC:.3f}",
                    ])

                    wrote_anything = True

    with open(FREQ_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerows(csv_rows)

    print()
    print(f"Missing files skipped: {missing_count}")
    if wrote_anything:
        print("Done. Outputs written to:", OUT_DIR.resolve())
        print("Excitation frequency summary CSV:", FREQ_CSV.resolve())
    else:
        print("No outputs written (all files were missing or skipped).")


if __name__ == "__main__":
    main()
