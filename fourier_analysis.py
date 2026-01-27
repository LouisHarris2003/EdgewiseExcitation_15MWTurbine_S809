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


INTERACTIVE_DEBUG = True

NORMALIZE_DATA = True

DEBUG_CASES = {"Polar_Instability_Resonance"}   # only resonance
DEBUG_WINDS = {"10ms", "30ms", "50ms"}          # all winds
DEBUG_STRUCTS = {"Flex", "Rigid"}               # both
DEBUG_MODELS = {"None", "Oye2", "IAGModel", "BeddoesIncomp"}  # all


def is_debug_case(case: str, wind: str, struct: str, model: str) -> bool:
    return (
        INTERACTIVE_DEBUG
        and (case in DEBUG_CASES)
        and (wind in DEBUG_WINDS)
        and (struct in DEBUG_STRUCTS)
        and (model in DEBUG_MODELS)
    )

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
# NORMALIZATION FUNCTION
# -----------------------------
def normalize_by_range(x: np.ndarray):
    """
    Normalize by range (peak-to-peak amplitude).
    
    Args:
        x: Input signal (already demeaned)
    
    Returns:
        normalized signal, normalization_scale (the range value)
    """
    x_range = np.max(x) - np.min(x)
    if x_range > 1e-12:
        return x / x_range, x_range
    else:
        return x, 1.0

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


def plot_time_freq_ax(
    ax,
    f: np.ndarray,
    tt: np.ndarray,
    Z: np.ndarray,
    title: str,
    fmax: float,
    freq_lines=None,
    vmin: float = COLOUR_LIM_MIN,
    vmax: float = COLOUR_LIM_MAX,
    cmap: str = "magma",
):
    if fmax is not None:
        m = f <= fmax
        f = f[m]
        Z = Z[m, :]

    im = ax.pcolormesh(
        tt, f, Z,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")

    if freq_lines:
        y_max = f[-1] if len(f) else None
        x0 = tt[0] if len(tt) else 0.0
        for fr, label in freq_lines:
            if y_max is None or fr > y_max:
                continue
            ax.axhline(fr, linestyle="--", linewidth=1)
            ax.text(x0, fr, f" {label}", va="bottom")

    return im

def save_combined_timefreq_figure(
    outpath: Path,
    case: str, wind: str, struct: str, model: str,
    f: np.ndarray, tt: np.ndarray,
    Z_alpha: np.ndarray, Z_cl: np.ndarray, Z_cd: np.ndarray,
    fmax_plot: float,
    freq_lines=None,
    vmin: float = COLOUR_LIM_MIN,
    vmax: float = COLOUR_LIM_MAX,
    cmap: str = "magma",
    figsize=(10, 18),
    dpi=200,
    show: bool = False,

):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)

    # Make room for the colorbar on the right (so it doesn't cover plots)
    fig.subplots_adjust(right=0.86, hspace=0.25)

    im = plot_time_freq_ax(
        axs[0], f, tt, Z_alpha, title="alpha",
        fmax=fmax_plot, freq_lines=freq_lines,
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    plot_time_freq_ax(
        axs[1], f, tt, Z_cl, title="cl",
        fmax=fmax_plot, freq_lines=freq_lines,
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    plot_time_freq_ax(
        axs[2], f, tt, Z_cd, title="cd",
        fmax=fmax_plot, freq_lines=freq_lines,
        vmin=vmin, vmax=vmax, cmap=cmap
    )

    axs[2].set_xlabel("Time (s)")

    fig.suptitle(f"{case} | {wind} | {struct} | {model}", fontsize=14)

    # Put the colorbar OUTSIDE the axes
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Log Power (dB)")

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()  
    plt.close(fig)


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
    print(f"Normalization: {NORMALIZE_DATA} (dividing by range)")
    print()

    #csv_rows = [["case", "wind", "struct", "model", "file", "fe_hz", "fmax_plot_hz"]]


    csv_rows = [[
    "case", "wind", "struct", "model", "file",
    "fe_hz", "fmax_plot_hz",
    "fs_hz", "nperseg", "noverlap",
    "window_sec", "overlap_sec", "alpha_norm_scale",
     "cl_norm_scale", "cd_norm_scale"
    ]]

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

                    # Normalize each signal independently by its range
                    if NORMALIZE_DATA:
                        alpha_norm, alpha_scale = normalize_by_range(alpha)
                        cl_norm, cl_scale = normalize_by_range(cl)
                        cd_norm, cd_scale = normalize_by_range(cd)
                        
                        print(f"  Normalization scales: alpha={alpha_scale:.6f}, cl={cl_scale:.6f}, cd={cd_scale:.6f}")
                    else:
                        alpha_norm, cl_norm, cd_norm = alpha, cl, cd
                        alpha_scale = cl_scale = cd_scale = 1.0

                    

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
                    f, tt, Z_alpha = moving_fft(alpha_norm, fs, nperseg=nperseg, noverlap=noverlap)
                    _, _, Z_cl = moving_fft(cl_norm, fs, nperseg=nperseg, noverlap=noverlap)
                    _, _, Z_cd = moving_fft(cd_norm, fs, nperseg=nperseg, noverlap=noverlap)


                    # ---- combined figure (saved + optional show) ----
                    base = OUT_DIR / case / wind / struct / model
                    
                    save_combined_timefreq_figure(
                        outpath=base / "combined_timefreq.png",
                        case=case, wind=wind, struct=struct, model=model,
                        f=f, tt=tt,
                        Z_alpha=Z_alpha, Z_cl=Z_cl, Z_cd=Z_cd,
                        fmax_plot=fmax_plot,
                        freq_lines=freq_lines,
                        figsize=(10, 18),   # change to (10, 6) if you want same dimension as singles
                        dpi=200,
                        show=is_debug_case(case, wind, struct, model),

                        )



                    
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
                        f"{fs:.6f}",
                        str(nperseg),
                        str(noverlap),
                        f"{WINDOW_SEC:.3f}",
                        f"{OVERLAP_SEC:.3f}",
                        f"{alpha_scale:.6f}",
                        f"{cl_scale:.6f}",
                        f"{cd_scale:.6f}",
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
