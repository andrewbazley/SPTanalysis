#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from glob import glob
import concurrent.futures
from scipy import stats
import argparse

###############################################################################
# Utility functions for mapping folder names to experimental condition
###############################################################################
def simple_clean(s):
    """
    Clean a string by stripping whitespace, removing spaces, and converting to lower-case.
    Used for output filenames.
    """
    return s.strip().replace(" ", "").lower()

def get_mapping_from_groupnames(input_directory):
    """
    Reads groupnames.csv (with two columns: folder, condition) from input_directory.
    Returns a dictionary mapping folder names to condition.
    """
    groupnames_path = os.path.join(input_directory, "groupnames.csv")
    if not os.path.exists(groupnames_path):
        print("Warning: groupnames.csv not found in input directory. All conditions will be 'Unknown'.")
        return {}
    try:
        # If the CSV file has no header, assume two columns.
        group_df = pd.read_csv(groupnames_path, header=None, names=["folder", "condition"])
    except Exception as e:
        print(f"Error reading groupnames.csv: {e}")
        return {}
    mapping = {}
    for idx, row in group_df.iterrows():
        folder = str(row["folder"]).strip()
        condition = str(row["condition"]).strip()
        mapping[folder] = condition
    return mapping

def get_condition_from_path(file_path, folder_to_mapping):
    """
    Returns the condition for a given file path by:
      1) Extracting the folder name from the path.
      2) Looking up an exact (or partial) match in folder_to_mapping.
      3) If no match is found, returns "Unknown".
    """
    folder_name = os.path.basename(os.path.dirname(file_path))
    if folder_name in folder_to_mapping:
        return folder_to_mapping[folder_name]
    for known_folder, condition in folder_to_mapping.items():
        if known_folder in folder_name:
            return condition
    return "Unknown"

###############################################################################
# MSD processing functions (unchanged)
###############################################################################
def process_msd_file(file_path, folder_to_mapping):
    """
    Reads an MSD CSV file and returns:
       (condition, replicate_df)
    replicate_df will have columns: 'Lag Time', 'Trajectory_1', 'Trajectory_2', etc.
    """
    condition = get_condition_from_path(file_path, folder_to_mapping)
    if condition == "Unknown":
        print(f"Skipping {file_path}: Unable to determine condition from folder.")
        return None, None
    try:
        df = pd.read_csv(file_path)
        if "lag time [s]" not in df.columns:
            print(f"Skipping {file_path}: Missing 'lag time [s]' column.")
            return None, None
        lag_times = df["lag time [s]"].values
        traj_cols = df.columns[1:]
        data = np.column_stack([lag_times] + [df[col].values for col in traj_cols])
        columns = ["Lag Time"] + [f"Trajectory_{i+1}" for i in range(len(traj_cols))]
        replicate_df = pd.DataFrame(data, columns=columns)
        return condition, replicate_df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def unify_and_average(dfs):
    """
    Given a list of DataFrames (each with 'Lag Time' and trajectory columns),
    returns (all_lag_times, mean_msd, std_msd) computed over all trajectories across replicates.
    The error bars indicate the standard deviation of MSD values at each lag time.
    """
    if not dfs:
        return np.array([]), np.array([]), np.array([])
    all_lag_times = sorted(set(np.concatenate([df["Lag Time"].values for df in dfs])))
    all_trajectories = []
    for df in dfs:
        replicate_lags = df["Lag Time"].values
        traj_data = df.drop(columns="Lag Time").values  # shape: (n_points, n_trajectories)
        replicate_interp = []
        for col in range(traj_data.shape[1]):
            y = traj_data[:, col]
            interp_vals = np.interp(all_lag_times, replicate_lags, y, left=np.nan, right=np.nan)
            replicate_interp.append(interp_vals)
        replicate_interp = np.column_stack(replicate_interp)
        all_trajectories.append(replicate_interp)
    all_trajectories = np.hstack(all_trajectories)
    mean_msd = np.nanmean(all_trajectories, axis=1)
    std_msd = np.nanstd(all_trajectories, axis=1)
    return np.array(all_lag_times), mean_msd, std_msd

###############################################################################
# Main Script
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Process IMSD data and individual replicate *_tracks.csv files, generating both individual and aggregate-level plots."
    )
    parser.add_argument("input_directory", type=str,
                        help="Path to directory containing groupnames.csv and replicate folders.")
    args = parser.parse_args()
    input_directory = args.input_directory

    # Build folder mapping from groupnames.csv (expects columns: folder, condition)
    folder_to_mapping = get_mapping_from_groupnames(input_directory)

    ###########################################################################
    # 1) Process _IMSD.csv files for MSD vs Lag Time analysis (unchanged)
    ###########################################################################
    msd_data = {}
    imsd_files = glob(os.path.join(input_directory, "**", "*_IMSD.csv"), recursive=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_msd_file, f, folder_to_mapping): f for f in imsd_files}
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            condition, replicate_df = future.result()
            if condition is None or replicate_df is None:
                continue
            if condition not in msd_data:
                msd_data[condition] = []
            msd_data[condition].append(replicate_df)

    ###########################################################################
    # 2) Process individual *_tracks.csv files for per-track calculations
    #    (compute per-track alpha values and mean step sizes)
    #    Aggregate the processed data for later aggregate-level plotting.
    ###########################################################################
    tracks_files = glob(os.path.join(input_directory, "**", "*_tracks.csv"), recursive=True)
    aggregate_tracks_list = []

    def process_track_file(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
        required_cols = {"particle", "frame", "x", "y"}
        if not required_cols.issubset(set(df.columns)):
            print(f"File {file_path} missing required columns {required_cols}. Skipping.")
            return None

        # Compute per-track alpha values using compute_track_alpha (assumed defined)
        track_alpha_dict = {}
        for particle, group in df.groupby("particle"):
            try:
                # compute_track_alpha should return a scalar alpha for a given track DataFrame
                alpha_val = compute_track_alpha(group, FPS, MPP)
            except Exception as ex:
                print(f"Error computing alpha for particle {particle} in {file_path}: {ex}")
                alpha_val = np.nan
            track_alpha_dict[particle] = alpha_val
        df["alpha"] = df["particle"].map(track_alpha_dict)

        # Compute per-track mean step size (Euclidean distance between consecutive (x,y) positions)
        track_mean_step = {}
        for particle, group in df.groupby("particle"):
            group_sorted = group.sort_values("frame")
            dx = group_sorted["x"].diff()
            dy = group_sorted["y"].diff()
            steps = np.sqrt(dx**2 + dy**2).dropna()
            track_mean_step[particle] = np.mean(steps) if len(steps) > 0 else np.nan
        df["mean_step_size"] = df["particle"].map(track_mean_step)

        # Assign experimental condition based on file path
        df["condition"] = get_condition_from_path(file_path, folder_to_mapping)

        # Save updated tracks file (do not overwrite original; here we add a suffix "_tracks_updated.csv")
        updated_file = file_path.replace("_tracks.csv", "_tracks_updated.csv")
        df.to_csv(updated_file, index=False)
        print(f"Updated tracks file saved to: {updated_file}")
        return df

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_track_file, tracks_files))
    for r in results:
        if r is not None:
            aggregate_tracks_list.append(r)
    if aggregate_tracks_list:
        aggregated_tracks = pd.concat(aggregate_tracks_list, ignore_index=True)
    else:
        aggregated_tracks = pd.DataFrame()

    ###########################################################################
    # 3) Combined Aggregate Plots (All Conditions)
    ###########################################################################
    # (a) Combined Histogram (bar chart) for each condition (MSD data)
    for condition, dfs in msd_data.items():
        if not dfs:
            continue
        lts, mean_msd, std_msd = unify_and_average(dfs)
        plt.figure(figsize=(10, 6))
        yerr = [np.zeros_like(std_msd), std_msd]
        x_positions = np.arange(len(lts))
        plt.bar(x_positions, mean_msd, yerr=yerr, alpha=0.7, capsize=3)
        plt.xticks(x_positions, [f"{lt:g}" for lt in lts], rotation=90)
        plt.xlabel("Lag Time (s)")
        plt.ylabel("Mean MSD (µm²)")
        plt.title(f"MSD Histogram for {condition}")
        plt.tight_layout()
        output_file = os.path.join(input_directory, f"MSD_{simple_clean(condition)}_histogram.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved MSD histogram for {condition} to: {output_file}")

    # (b) Combined MSD Line Plot (all replicates, aggregate over all conditions)
    combined_results = []
    def process_file_for_combined(file_path):
        condition = get_condition_from_path(file_path, folder_to_mapping)
        movie_name = os.path.basename(file_path).replace("_IMSD.csv", "")
        try:
            df = pd.read_csv(file_path)
            lag_times = df.iloc[:, 0].values
            msd_values = df.drop(columns=[df.columns[0]]).mean(axis=1).values
            min_length = min(len(lag_times), len(msd_values))
            df_msd = pd.DataFrame({"Lag Time": lag_times[:min_length], "MSD": msd_values[:min_length]})
            return condition, movie_name, df_msd
        except Exception as e:
            print(f"Error processing {file_path} for combined plot: {e}")
            return None, None, None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file_for_combined, imsd_files))
    for condition, movie_name, df_msd in results:
        if condition is None or movie_name is None or df_msd is None:
            continue
        combined_results.append((condition, movie_name, df_msd))
    output_combined_path = os.path.join(input_directory, "combined_msd_plot.png")
    plt.figure(figsize=(8, 6))
    for condition, movie_name, df_msd in combined_results:
        plt.plot(df_msd["Lag Time"], df_msd["MSD"], label=f"{condition} - {movie_name}", linestyle="-", alpha=0.7)
    plt.xlabel("Lag Time (s)")
    plt.ylabel("Mean Squared Displacement (µm²)")
    plt.title("MSD vs Lag Time for All Replicates")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(output_combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined MSD line plot to: {output_combined_path}")

    # (c) Average Combined MSD Plot by Condition
    plt.figure(figsize=(10, 6))
    for condition, dfs in msd_data.items():
        if dfs:
            lts, mean_msd, std_msd = unify_and_average(dfs)
            plt.errorbar(lts, mean_msd, yerr=std_msd, fmt='-o', capsize=3, alpha=0.7, label=condition)
    plt.xlabel("Lag Time (s)")
    plt.ylabel("Mean MSD (µm²)")
    plt.title("Average Combined MSD vs Lag Time by Condition")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_combined_avg = os.path.join(input_directory, "average_combined_msd_by_condition_plot.png")
    plt.savefig(output_combined_avg, dpi=300)
    plt.close()
    print(f"Saved average combined MSD by condition plot to: {output_combined_avg}")

    # (d) Averaged MSD Line Plots (with and without 95% CI) across conditions
    condition_data = {}
    for condition, dfs in msd_data.items():
        if dfs:
            condition_data[condition] = dfs
    def save_msd_plots(output_path, include_ci=False):
        plt.figure(figsize=(8, 6))
        for condition, dfs in condition_data.items():
            all_lag_times = sorted(set(np.concatenate([df["Lag Time"].values for df in dfs])))
            msd_array = np.array([np.interp(
                all_lag_times, 
                df["Lag Time"], 
                df.drop(columns=["Lag Time"]).mean(axis=1),
                left=np.nan, right=np.nan
            ) for df in dfs])
            mean_msd = np.nanmean(msd_array, axis=0)
            plt.plot(all_lag_times, mean_msd, label=f"{condition} (avg)", linewidth=2)
            if include_ci:
                std_err = stats.sem(msd_array, axis=0, nan_policy='omit')
                ci_95 = std_err * stats.t.ppf((1 + 0.95) / 2, msd_array.shape[0] - 1)
                plt.fill_between(all_lag_times, mean_msd - ci_95, mean_msd + ci_95, alpha=0.3)
        plt.xlabel("Lag Time (s)")
        plt.ylabel("Averaged MSD (µm²)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    avg_plot_path = os.path.join(input_directory, "average_msd_plot.png")
    avg_ci_plot_path = os.path.join(input_directory, "average_msd_plot_CI.png")
    save_msd_plots(avg_plot_path, include_ci=False)
    save_msd_plots(avg_ci_plot_path, include_ci=True)
    print(f"Saved averaged MSD plot without CI to: {avg_plot_path}")
    print(f"Saved averaged MSD plot with 95% CI to: {avg_ci_plot_path}")

    # (e) Alpha values density curves per condition (aggregate from processed tracks)
    if not aggregated_tracks.empty:
        plt.figure(figsize=(10, 6))
        conditions = aggregated_tracks["condition"].unique()
        for cond in conditions:
            subset = aggregated_tracks[aggregated_tracks["condition"] == cond]
            alpha_vals = subset["alpha"].dropna().values
            if len(alpha_vals) == 0:
                continue
            kde = stats.gaussian_kde(alpha_vals)
            x_grid = np.linspace(min(alpha_vals), max(alpha_vals), 100)
            plt.plot(x_grid, kde(x_grid), label=f"{cond} (per-track)")
            ens_alpha = np.nanmean(alpha_vals)
            plt.axvline(ens_alpha, linestyle="--", label=f"{cond} (ensemble)")
        plt.xlabel("Alpha Value")
        plt.ylabel("Density")
        plt.title("Alpha Value Density Curves per Experimental Condition")
        plt.legend()
        plt.tight_layout()
        alpha_plot_path = os.path.join(input_directory, "alpha_density_curves.png")
        plt.savefig(alpha_plot_path, dpi=300)
        plt.close()
        print(f"Saved alpha density curves plot to: {alpha_plot_path}")
    else:
        print("No aggregated track data available for alpha density curves.")

    # (f) Histogram and KDE of step size distributions by condition (aggregate)
    aggregate_steps_by_condition = {}
    # Recompute step sizes from the aggregated tracks (grouping by unique track within each condition)
    for (cond, particle), group in aggregated_tracks.groupby(["condition", "particle"]):
        group_sorted = group.sort_values("frame")
        dx = group_sorted["x"].diff()
        dy = group_sorted["y"].diff()
        steps = np.sqrt(dx**2 + dy**2).dropna().tolist()
        if cond not in aggregate_steps_by_condition:
            aggregate_steps_by_condition[cond] = []
        aggregate_steps_by_condition[cond].extend(steps)
    if aggregate_steps_by_condition:
        all_steps = np.concatenate([np.array(steps) for steps in aggregate_steps_by_condition.values() if len(steps) > 0])
        if all_steps.size > 0:
            bin_range = (np.min(all_steps), np.max(all_steps))
            bins = np.linspace(bin_range[0], bin_range[1], 20)
        else:
            bins = 20
        for cond, steps in aggregate_steps_by_condition.items():
            if len(steps) == 0:
                continue
            plt.figure(figsize=(10, 6))
            counts, bin_edges, _ = plt.hist(steps, bins=bins, density=True, alpha=0.6, edgecolor="black", label="Histogram")
            kde = stats.gaussian_kde(steps)
            x_grid = np.linspace(min(steps), max(steps), 100)
            plt.plot(x_grid, kde(x_grid), label="KDE")
            plt.xlabel("Step Size")
            plt.ylabel("Density")
            plt.title(f"Step Size Distribution for {cond}")
            plt.legend()
            plt.tight_layout()
            step_hist_path = os.path.join(input_directory, f"{simple_clean(cond)}_steps_histogram.png")
            plt.savefig(step_hist_path, dpi=300)
            plt.close()
            print(f"Saved step size histogram for {cond} to: {step_hist_path}")
    else:
        print("No step size data available for aggregate histograms.")

    ###########################################################################
    # 4) Extra replicate histograms from *_tracks.csv files (unchanged except for error handling)
    ###########################################################################
    def process_tracks_histograms(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path} for extra histograms: {e}")
            return
        folder = os.path.dirname(file_path)
        base_name = os.path.basename(file_path).replace("_tracks.csv", "")
        # Mass vs Size scatter plot
        if "mass" in df.columns and "size" in df.columns:
            mass = pd.to_numeric(df["mass"], errors="coerce")
            size = pd.to_numeric(df["size"], errors="coerce")
            valid = mass.notna() & size.notna() & np.isfinite(mass) & np.isfinite(size)
            if valid.sum() > 0:
                plt.figure()
                plt.scatter(size[valid], mass[valid], alpha=0.7)
                plt.xlabel("Size")
                plt.ylabel("Mass")
                plt.title(f"Mass vs Size for {base_name}")
                plt.tight_layout()
                output_path = os.path.join(folder, f"{base_name}_mass_vs_size.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"Saved mass vs size plot to: {output_path}")
            else:
                print(f"Skipping Mass vs Size for {base_name}: no valid data.")
        else:
            print(f"Skipping Mass vs Size for {base_name}: 'mass' or 'size' column not found.")
        # Histogram of eccentricity
        if "ecc" in df.columns:
            plt.figure()
            data = pd.to_numeric(df["ecc"], errors="coerce").dropna()
            data = data[np.isfinite(data)]
            if data.size > 0:
                plt.hist(data, bins=20, alpha=0.7, edgecolor="black")
                plt.xlabel("Eccentricity")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of Ecc for {base_name}")
                plt.tight_layout()
                output_path = os.path.join(folder, f"{base_name}_ecc_histogram.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"Saved ecc histogram to: {output_path}")
            else:
                print(f"Skipping Ecc histogram for {base_name}: no valid data.")
        else:
            print(f"Skipping Ecc histogram for {base_name}: 'ecc' column not found.")
        # Histogram of signal
        if "signal" in df.columns:
            plt.figure()
            data = pd.to_numeric(df["signal"], errors="coerce").dropna()
            data = data[np.isfinite(data)]
            if data.size > 0:
                plt.hist(data, bins=20, alpha=0.7, edgecolor="black")
                plt.xlabel("Signal")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of Signal for {base_name}")
                plt.tight_layout()
                output_path = os.path.join(folder, f"{base_name}_signal_histogram.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"Saved signal histogram to: {output_path}")
            else:
                print(f"Skipping Signal histogram for {base_name}: no valid data.")
        else:
            print(f"Skipping Signal histogram for {base_name}: 'signal' column not found.")
        # Histogram of dwell time
        if "dwell time" in df.columns:
            plt.figure()
            data = pd.to_numeric(df["dwell time"], errors="coerce").dropna()
            data = data[np.isfinite(data)]
            if data.size > 0:
                plt.hist(data, bins=20, alpha=0.7, edgecolor="black")
                plt.xlabel("Dwell Time")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of Dwell Time for {base_name}")
                plt.tight_layout()
                output_path = os.path.join(folder, f"{base_name}_dwell_time_histogram.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"Saved dwell time histogram to: {output_path}")
            else:
                print(f"Skipping Dwell Time histogram for {base_name}: no valid data.")
        else:
            print(f"Skipping Dwell Time histogram for {base_name}: 'dwell time' column not found.")
        # Histogram of total mass (log-scaled)
        if "total mass" in df.columns:
            plt.figure()
            data = pd.to_numeric(df["total mass"], errors="coerce").dropna()
            data = data[np.isfinite(data)]
            if data.size > 0:
                plt.hist(data, bins=20, alpha=0.7, edgecolor="black")
                plt.xscale("log")
                plt.xlabel("Total Mass (log scale)")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of Total Mass for {base_name}")
                plt.tight_layout()
                output_path = os.path.join(folder, f"{base_name}_total_mass_histogram.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"Saved total mass histogram to: {output_path}")
            else:
                print(f"Skipping Total Mass histogram for {base_name}: no valid data.")
        else:
            print(f"Skipping Total Mass histogram for {base_name}: 'total mass' column not found.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_tracks_histograms, tracks_files))

    ###########################################################################
    # 5) Aggregate Track Length Histograms by Condition
    ###########################################################################
    track_lengths = {}
    for file_path in tracks_files:
        condition = get_condition_from_path(file_path, folder_to_mapping)
        if condition == "Unknown":
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path} for track lengths: {e}")
            continue
        if "particle" not in df.columns:
            print(f"File {file_path} missing 'particle' column for track lengths. Skipping.")
            continue
        track_length_series = df.groupby("particle").size()
        lengths = track_length_series.values.tolist()
        if condition not in track_lengths:
            track_lengths[condition] = []
        track_lengths[condition].extend(lengths)
    for condition, lengths in track_lengths.items():
        if not lengths:
            continue
        plt.figure()
        plt.hist(lengths, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Track Length (number of frames)")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Track Lengths for {condition}")
        plt.tight_layout()
        output_file = os.path.join(input_directory, f"{simple_clean(condition)}_track_length_histogram.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved track length histogram for {condition} to: {output_file}")

    ###########################################################################
    # 6) Per-track Step Size Matrices for Each Replicate *_tracks.csv File
    #     For each replicate file, compute the step sizes at each time lag per track,
    #     arrange them into a DataFrame (rows = time lag, columns = track), and save.
    ###########################################################################
    def process_step_sizes(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path} for step size matrix: {e}")
            return
        # Check that required columns exist
        required_cols = {"particle", "frame", "x", "y"}
        if not required_cols.issubset(set(df.columns)):
            print(f"File {file_path} missing required columns {required_cols} for step size matrix. Skipping.")
            return
        step_size_dict = {}
        # For each track, compute the step sizes (Euclidean difference) at each time lag
        for particle, group in df.groupby("particle"):
            group = group.sort_values("frame")
            dx = group["x"].diff()
            dy = group["y"].diff()
            # Exclude the initial NaN value
            steps = np.sqrt(dx**2 + dy**2).iloc[1:]
            step_size_dict[particle] = steps.values
        if not step_size_dict:
            print(f"No valid step sizes computed for {file_path}.")
            return
        # Determine the maximum number of steps among all tracks for proper alignment
        max_steps = max(len(arr) for arr in step_size_dict.values())
        step_size_data = {}
        for particle, steps in step_size_dict.items():
            if len(steps) < max_steps:
                padded = np.pad(steps, (0, max_steps - len(steps)), constant_values=np.nan)
            else:
                padded = steps
            step_size_data[particle] = padded
        # Create DataFrame: rows = time lag index, columns = particle (track)
        step_df = pd.DataFrame(step_size_data)
        folder = os.path.dirname(file_path)
        base_name = os.path.basename(file_path).replace("_tracks.csv", "")
        output_path = os.path.join(folder, f"{base_name}_step_sizes.csv")
        step_df.to_csv(output_path, index_label="Time Lag")
        print(f"Saved step size matrix for {base_name} to: {output_path}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_step_sizes, tracks_files))

if __name__ == "__main__":
    main()
