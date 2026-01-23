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

# STFT settings (tune if galih says)
NPERSEG = 1024
NOVERLAP = 768


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
# MOVING FFT (STFT)
# -----------------------------
def moving_fft(x: np.ndarray, fs: float, nperseg: int = 1024, noverlap: int = 768):
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
    )

    P = np.abs(Zxx) ** 2
    Z = 10.0 * np.log10(P + 1e-12)
    return f, tt, Z


def plot_time_freq(
    f: np.ndarray,
    tt: np.ndarray,
    Z: np.ndarray,
    title: str,
    outpath: Path,
    fmax: float,
    freq_lines=None,
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
    plt.pcolormesh(tt, f, Z, shading="auto")
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


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Current working directory:", Path.cwd())
    print("Dataset dir:", DATASET_DIR.resolve())
    print("Output dir:", OUT_DIR.resolve())
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

                    # STFT
                    f, tt, Z_alpha = moving_fft(alpha, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
                    _, _, Z_cl = moving_fft(cl, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
                    _, _, Z_cd = moving_fft(cd, fs, nperseg=NPERSEG, noverlap=NOVERLAP)

                    base = OUT_DIR / case / wind / struct / model

                    plot_time_freq(
                        f, tt, Z_alpha,
                        title=f"{case} | {wind} | {struct} | {model} | alpha",
                        outpath=base / "alpha_timefreq.png",
                        fmax=fmax_plot,
                        freq_lines=freq_lines
                    )
                    plot_time_freq(
                        f, tt, Z_cl,
                        title=f"{case} | {wind} | {struct} | {model} | cl",
                        outpath=base / "cl_timefreq.png",
                        fmax=fmax_plot,
                        freq_lines=freq_lines
                    )
                    plot_time_freq(
                        f, tt, Z_cd,
                        title=f"{case} | {wind} | {struct} | {model} | cd",
                        outpath=base / "cd_timefreq.png",
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
