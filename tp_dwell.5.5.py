import os
import glob
import argparse
import numpy as np
import pandas as pd
import pims
from nd2reader import ND2Reader
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent pop-ups
import matplotlib.pyplot as plt
import trackpy as tp
import numba

numba.config.THREADING_LAYER = 'workqueue'  # Optimized threading
numba.set_num_threads(32)  # Adjust based on CPU cores
import scipy.stats as stats
import re
import concurrent.futures
from trackpy.linking.utils import SubnetOversizeException

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Batch process ND2/TIFF files using Trackpy with adaptive minmass filtering.")
parser.add_argument("input_directory", type=str, help="Path to the folder containing image files")
parser.add_argument("-notrack", action="store_true", help="Skip Trackpy tracking and only generate MSD plots")
parser.add_argument("-minmsd", type=float, default=0, help="Minimum MSD threshold for filtering tracks")
parser.add_argument("-mpp", type=float, default=0.15, help="Microns per pixel")
parser.add_argument("-fps", type=float, default=33.33, help="Frames per second")
parser.add_argument("-diam", type=int, default=11, help="Particle diameter (pixels)")
parser.add_argument("-range", type=int, default=5, help="Particle linking range (pixels)")
parser.add_argument("-memory", type=int, default=1, help="Particle linking memory (frames)")
parser.add_argument("-minmass", type=float, default=70, help="Minimum mass threshold for particle detection")
parser.add_argument("-percentile", type=float, default=20, help="Percentile threshold for particle detection")
args = parser.parse_args()

input_directory = args.input_directory
min_msd_threshold = args.minmsd
MPP = args.mpp
FPS = args.fps
DIAMETER = args.diam
SEARCH_RANGE = args.range
MEMORY = args.memory

def compute_track_alpha(track_df, fps, mpp):
    """
    Compute the slope (alpha) for a single track using a log-log linear fit to the MSD versus lag time.
    If the computed slope is negative (indicating very little lateral diffusion), return 0.
    """
    if "frame" in track_df.index.names:
        track_df = track_df.reset_index()
    track_df = track_df.sort_values("frame")
    positions = track_df[["x", "y"]].to_numpy() * mpp
    n = len(positions)
    if n < 2:
        return np.nan

    lags = np.arange(1, n)
    msd_values = [np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=1))
                  for lag in lags]
    lag_times = lags / fps  # Convert lag (in frames) to seconds
    if len(lag_times) < 2:
        return np.nan

    log_lag = np.log(lag_times)
    log_msd = np.log(msd_values)
    slope = np.polyfit(log_lag, log_msd, 1)[0]
    if slope < 0:
        slope = 0
    return slope

def compute_ensemble_alpha(emsd_df, fps):
    """
    Compute the ensemble alpha (slope) from the ensemble MSD.
    If the computed slope is negative, return 0.
    """
    emsd_valid = emsd_df[emsd_df.index > 0]  # Exclude lag time 0
    if emsd_valid.empty:
        return np.nan
    emsd_valid = emsd_valid.squeeze()  # Ensure it's a Series
    lag_times = emsd_valid.index.values / fps  # Convert frames to seconds
    msd_values = emsd_valid.values
    if len(lag_times) < 2:
        return np.nan
    log_lag = np.log(lag_times)
    log_msd = np.log(msd_values)
    slope = np.polyfit(log_lag, log_msd, 1)[0]
    if slope < 0:
        slope = 0
    return slope

def process_file(file_path):
    """
    Process a single file: load frames (using the first channel if multi-channel), perform tracking and linking,
    compute per-track and ensemble MSD and alpha values, save results to disk, and output an overlay image of tracks.
    Returns a tuple (condition_name, msd_values) for later histogramming.
    
    Files with "BF_" in the title are skipped.
    """
    movie_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Skip files containing "BF_" in their title.
    if "BF_" in movie_name:
        print(f"Skipping file {file_path} because it contains 'BF_' in the title.")
        return (None, [])
    
    file_extension = os.path.splitext(file_path)[1].lower()
    condition_name = os.path.basename(os.path.dirname(file_path))
    print(f"Processing {file_path} (Condition: {condition_name})...")

    # Load frames
    frames = []
    if file_extension == '.nd2':
        with ND2Reader(file_path) as images:
            if hasattr(images, "axes") and 'c' in images.axes:
                images.default_coords['c'] = 0
            for frame in images:
                arr = np.array(frame, dtype=np.float32)
                if arr.ndim > 2:  # Multi-channel image; use first channel
                    arr = arr[0]
                frames.append(arr)
    elif file_extension in ['.tif', '.tiff']:
        frames_seq = pims.open(file_path)
        for frame in frames_seq:
            arr = np.array(frame, dtype=np.float32)
            if arr.ndim > 2:  # Multi-channel image; use first channel
                arr = arr[0]
            frames.append(arr)
    else:
        print(f"Unsupported file format: {file_extension}. Skipping.")
        return (condition_name, [])

    # Create output folder (inside the input folder)
    output_directory = os.path.join(input_directory, "trackpy-results")
    movie_output_dir = os.path.join(output_directory, movie_name)
    os.makedirs(movie_output_dir, exist_ok=True)

    # If the output files already exist, skip processing this movie.
    particles_file = os.path.join(movie_output_dir, f"{movie_name}_particles.csv")
    tracks_file = os.path.join(movie_output_dir, f"{movie_name}_tracks.csv")
    imsd_file = os.path.join(movie_output_dir, f"{movie_name}_IMSD.csv")
    emsd_file = os.path.join(movie_output_dir, f"{movie_name}_EMSD.csv")
    if (os.path.exists(particles_file) and os.path.exists(tracks_file) and
            os.path.exists(imsd_file) and os.path.exists(emsd_file)):
        print(f"Skipping processing for {movie_name} as output files already exist.")
        try:
            imsd = pd.read_csv(imsd_file, index_col=0)
            msd_vals = imsd.values.flatten().tolist()
        except Exception:
            msd_vals = [0]
        return (condition_name, msd_vals)

    # Run particle tracking
    separation_value = 5
    particles = tp.batch(frames, diameter=DIAMETER, minmass=args.minmass,
                         separation=separation_value, noise_size=1, percentile=args.percentile, engine='numba')
    particles.to_csv(particles_file, index=False)

    # Link particles using all detected features
    try:
        tracks = tp.link(particles, search_range=SEARCH_RANGE, memory=MEMORY)
    except SubnetOversizeException as e:
        print(f"Warning: {e} for movie {movie_name}. Skipping linking and setting MSD=0.")
        return (condition_name, [0])

    # Filter tracks: remove trajectories shorter than 10 frames.
    filtered_tracks = tp.filter_stubs(tracks, 10)
    if filtered_tracks.empty:
        return (condition_name, [0])

    # Add dwell_time and total_mass columns.
    dwell_times = filtered_tracks.groupby("particle")["frame"].count()
    filtered_tracks["dwell_time"] = filtered_tracks["particle"].map(dwell_times)
    filtered_tracks["total_mass"] = filtered_tracks.groupby("particle")["mass"].transform("sum")

    # Compute per-track alpha values and add them to a new "alpha" column.
    track_alpha_dict = {}
    groups = list(filtered_tracks.groupby("particle"))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_particle = {
            executor.submit(compute_track_alpha, group_df, FPS, MPP): particle
            for particle, group_df in groups
        }
        for future in concurrent.futures.as_completed(future_to_particle):
            particle = future_to_particle[future]
            try:
                track_alpha_dict[particle] = future.result()
            except Exception:
                track_alpha_dict[particle] = np.nan
    filtered_tracks["alpha"] = filtered_tracks["particle"].map(track_alpha_dict)
    filtered_tracks.to_csv(tracks_file, index=False)

    # Create overlay image on first frame.
    if len(frames) > 0:
        first_frame = frames[0]
        fig, ax = plt.subplots(figsize=(20, 16))  # Adjusted for a large image frame
        ax.imshow(first_frame, cmap='gray')
        for particle, group in filtered_tracks.groupby("particle"):
            group = group.reset_index(drop=True).sort_values("frame")
            ax.plot(group["x"], group["y"], marker="o", markersize=0.1, linewidth=1.5, alpha=0.9)
        ax.set_title(f"Filtered Tracks Overlay: {movie_name}")
        ax.axis("off")
        overlay_path = os.path.join(movie_output_dir, f"{movie_name}_tracks_overlay.png")
        plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Compute MSDs using the filtered tracks (which now include alpha values).
    imsd = tp.imsd(filtered_tracks, MPP, FPS, max_lagtime=25)
    emsd = tp.emsd(filtered_tracks, MPP, FPS, max_lagtime=25)

    # Compute the ensemble alpha from the EMSD table.
    ensemble_alpha = compute_ensemble_alpha(emsd, FPS)
    print(f"Ensemble alpha for {movie_name}: {ensemble_alpha:.3f}")

    # Add a new column "ensemble_alpha" to the EMSD DataFrame, repeating the ensemble alpha for each row.
    emsd["ensemble_alpha"] = ensemble_alpha

    # Save MSD files.
    imsd.to_csv(imsd_file)
    emsd.to_csv(emsd_file)

    msd_vals = imsd.values.flatten().tolist()
    return (condition_name, msd_vals)

def main():
    output_directory = os.path.join(input_directory, "trackpy-results")
    os.makedirs(output_directory, exist_ok=True)

    if args.notrack:
        print("Trackpy analysis skipped. Generating MSD plots only.")
        return

    file_paths = glob.glob(os.path.join(input_directory, "**", "*.*"), recursive=True)
    results = []
    for fp in file_paths:
        result = process_file(fp)
        results.append(result)

    # Combine MSD values by condition for histogram plotting.
    condition_msd_values = {}
    for condition, msd_vals in results:
        if condition is None:
            continue  # Skip files marked as ignored
        if condition not in condition_msd_values:
            condition_msd_values[condition] = []
        condition_msd_values[condition].extend(msd_vals)

        if condition_msd_values:
            plt.figure(figsize=(10, 6))
            for condition, msd_values in condition_msd_values.items():
                if msd_values:
                    # Convert to NumPy array and remove any non-finite values
                    msd_values = np.array(msd_values)
                    msd_values = msd_values[np.isfinite(msd_values)]
                    # Replace non-positive values with 1 (so log10(1)=0)
                    msd_values[msd_values <= 0] = 1
                    # Compute log10 and filter again (if any inf appears in log10)
                    log_msd = np.log10(msd_values)
                    log_msd = log_msd[np.isfinite(log_msd)]
                    plt.hist(log_msd, bins=100, alpha=0.5,
                             label=condition, histtype='stepfilled')
            plt.xlabel("log10(MSD)")
            plt.ylabel("Frequency")
            plt.title("Histogram of MSD values (log10 scale) by Experimental Condition")
            plt.grid(True)
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(output_directory, "all_msds_histogram.png"), dpi=300)
            plt.close()


    # Create groupnames.csv for all movies, excluding those with "BF_" in the title.
    # The CSV now contains only the folder name and an empty "condition" column.
    folder_list = [d for d in os.listdir(output_directory)
                   if os.path.isdir(os.path.join(output_directory, d)) and "BF_" not in d]
    groupnames_df = pd.DataFrame({
        "folder": folder_list,
        "condition": [''] * len(folder_list)
    })
    groupnames_csv_path = os.path.join(output_directory, "groupnames.csv")
    groupnames_df.to_csv(groupnames_csv_path, index=False)
    print(f"Saved groupnames CSV at {groupnames_csv_path}")

if __name__ == "__main__":
    main()
