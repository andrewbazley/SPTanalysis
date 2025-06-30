#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde


def simple_clean(s):
    """
    Clean a string by stripping whitespace, removing spaces, and converting to lower-case.
    Used to generate safe filenames.
    """
    return s.strip().replace(" ", "").lower()

def fit_gmm(data, max_components=3):
    """
    Fit a Gaussian Mixture Model to 1D data, trying from 1 to max_components.
    Returns the model with the lowest BIC.
    """
    best_bic = np.inf
    best_model = None
    for n in range(1, max_components+1):
        try:
            model = GaussianMixture(n_components=n, random_state=0)
            model.fit(data.reshape(-1, 1))
            bic = model.bic(data.reshape(-1, 1))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        except Exception:
            continue
    return best_model

def main():
    parser = argparse.ArgumentParser(
        description="Pool replicates from each experimental condition (as defined in groupnames.csv) and plot overlaid kernel density–normalized histograms for each parameter in _tracks.csv (raw and filtered), comparing replicates with and without rNTP. Histograms are organized under a folder named 'no_fits'. On filtered plots, the legend includes the overall mean (vertical dashed line) as well as the mean of each GMM component (displayed as dummy legend items)."
    )
    parser.add_argument("input_folder", help="Path to the folder containing groupnames.csv and replicate folders")
    
    # Default dwell_time cutoff for filtered data
    parser.add_argument("--cutoff", type=float, default=4, 
                        help="Minimum dwell_time (default: 4). Rows below this value are filtered out for filtered data.")
    
    # Optional filters for other columns
    parser.add_argument("--mass", type=float, default=None, help="Minimum mass filter.")
    parser.add_argument("--size", type=float, default=None, help="Minimum size filter.")
    parser.add_argument("--ecc", type=float, default=None, help="Minimum ecc filter.")
    parser.add_argument("--signal", type=float, default=None, help="Minimum signal filter.")
    parser.add_argument("--raw_mass", type=float, default=10000, help="Minimum raw_mass filter.")
    parser.add_argument("--ep", type=float, default=None, help="Minimum ep filter.")
    parser.add_argument("--frame", type=float, default=None, help="Minimum frame filter.")
    parser.add_argument("--particle", type=float, default=None, help="Minimum particle filter.")
    parser.add_argument("--total_mass", type=float, default=None, help="Minimum total_mass filter.")
    args = parser.parse_args()

    # Load groupnames.csv
    group_file = os.path.join(args.input_folder, "groupnames.csv")
    if not os.path.exists(group_file):
        print("groupnames.csv not found in the input folder.")
        return

    try:
        group_df = pd.read_csv(group_file)
    except Exception as e:
        print(f"Error reading groupnames.csv: {e}")
        return

    use_rntp = "rNTP" in group_df.columns

    # Check required columns
    if not all(col in group_df.columns for col in ["folder", "condition"]):
        print("groupnames.csv must contain at least 'folder' and 'condition' columns.")
        return

    if use_rntp:
        print("Detected 'rNTP' column. Grouping by condition and perturbation.")
    else:
        print("No 'rNTP' column detected. Grouping by condition only.")

    # Define parameters
    parameters = ["dwell_time", "mass", "size", "ecc", "signal", "raw_mass", "ep", "frame", "particle", "total_mass"]

    # Prepare dictionaries
    conditions_data_raw = {}
    conditions_data_filtered = {}

    for idx, row in group_df.iterrows():
        folder_name = str(row["folder"]).strip()
        condition = str(row["condition"]).strip()

        if use_rntp:
            rntp_str = str(row["rNTP"]).strip().lower()
            if "yes" in rntp_str:
                has_rntp = True
            elif "no" in rntp_str:
                has_rntp = False
            else:
                print(f"Row {idx}: Unrecognized rNTP value '{row['rNTP']}'. Skipping replicate.")
                continue
        else:
            has_rntp = True

        for d in (conditions_data_raw, conditions_data_filtered):
            if condition not in d:
                d[condition] = {}
            if has_rntp not in d[condition]:
                d[condition][has_rntp] = {}
            for p in parameters:
                if p not in d[condition][has_rntp]:
                    d[condition][has_rntp][p] = []
            for p in parameters:
                if p not in d[condition][has_rntp]:
                    d[condition][has_rntp][p] = []
    # Structure: { condition : { True: { parameter: [values] }, False: { parameter: [values] } } }
    conditions_data_raw = {}
    conditions_data_filtered = {}

    # Process each replicate from groupnames.csv
    for idx, row in group_df.iterrows():
        folder_name = str(row["folder"]).strip()
        condition = str(row["condition"]).strip()

        if use_rntp:
            rntp_str = str(row["rNTP"]).strip().lower()
            if "yes" in rntp_str:
                has_rntp = True
            elif "no" in rntp_str:
                has_rntp = False
            else:
                print(f"Row {idx}: Unrecognized rNTP value '{row['rNTP']}'. Skipping replicate.")
                continue
        else:
            has_rntp = True

        for d in (conditions_data_raw, conditions_data_filtered):
            if condition not in d:
                d[condition] = {}
            if has_rntp not in d[condition]:
                d[condition][has_rntp] = {}
            for p in parameters:
                if p not in d[condition][has_rntp]:
                    d[condition][has_rntp][p] = []
            for p in parameters:
                if p not in d[condition][has_rntp]:
                    d[condition][has_rntp][p] = []

        replicate_folder = os.path.join(args.input_folder, folder_name)
        if not os.path.isdir(replicate_folder):
            print(f"Replicate folder '{replicate_folder}' does not exist. Skipping.")
            continue

        pattern = os.path.join(replicate_folder, "*_tracks.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"No *_tracks.csv file found in '{replicate_folder}'. Skipping.")
            continue

        tracks_file = files[0]
        try:
            df_raw = pd.read_csv(tracks_file)
        except Exception as e:
            print(f"Error reading '{tracks_file}': {e}")
            continue

        
        # Restrict to first 1000 frames
        if "frame" in df_raw.columns:
            df_raw = df_raw[df_raw["frame"] < 1000]

# Pool raw data for each parameter
        for p in parameters:
            if p in df_raw.columns:
                conditions_data_raw[condition][has_rntp][p].extend(df_raw[p].tolist())

        # Create filtered copy
        df_filtered = df_raw.copy()
        df_filtered = df_filtered[df_filtered["dwell_time"] >= args.cutoff]
        optional_filters = {
            "mass": args.mass,
            "size": args.size,
            "ecc": args.ecc,
            "signal": args.signal,
            "raw_mass": args.raw_mass,
            "ep": args.ep,
            "frame": args.frame,
            "particle": args.particle,
            "total_mass": args.total_mass,
        }
        for col_name, min_value in optional_filters.items():
            if min_value is not None and col_name in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col_name] >= min_value]
        if df_filtered.empty:
            print(f"No rows remain after filtering in '{tracks_file}'. Skipping filtered data for this replicate.")
        else:
            for p in parameters:
                if p in df_filtered.columns:
                    conditions_data_filtered[condition][has_rntp][p].extend(df_filtered[p].tolist())

    # For each condition and parameter, generate histograms.
    # Two sets: one for raw data and one for filtered data.
    # Histograms are density normalized.
    # For raw plots: only overall mean is shown.
    # For filtered plots: overall mean is shown plus dummy legend entries for each GMM component's mean.
    for data_label, data_dict in [("raw", conditions_data_raw), ("filtered", conditions_data_filtered)]:
        for cond, groups in data_dict.items():
            cond_folder = os.path.join(args.input_folder, "histograms", simple_clean(cond), "no_fits", data_label)
            os.makedirs(cond_folder, exist_ok=True)

            present_flags = list(groups.keys())
            colors = ["blue", "orange", "green", "purple", "red"]
            flag_labels = {True: "+ rNTP", False: "- rNTP"} if use_rntp else {True: cond}

            for p in parameters:
                flag_data = {flag: np.array(groups[flag].get(p, [])) for flag in present_flags}
                if all(data.size == 0 for data in flag_data.values()):
                    continue

                plt.figure(figsize=(8, 6))
                ax = plt.gca()

                for i, flag in enumerate(present_flags):
                    data = flag_data[flag]
                    if data.size == 0:
                        continue
                    data[data <= 0] = 1  # avoid issues with log transform
                    label_suffix = flag_labels.get(flag, str(flag))
                    ax.hist(data, bins=30, density=True, alpha=0.5, color=colors[i % len(colors)],
                            label=f"{cond} {label_suffix}")
                    overall_mean = data.mean()
                    ax.axvline(overall_mean, color=colors[i % len(colors)], linestyle=":", linewidth=2,
                               label=f"Overall Mean {label_suffix} = {overall_mean:.2f}")

                    # Add GMM means if filtered
                    if data_label == "filtered":
                        gmm = fit_gmm(data)
                        if gmm:
                            for m in gmm.means_.flatten():
                                ax.axvline(m, color=colors[i % len(colors)], linestyle="--", linewidth=1)
                            dummy = [plt.Line2D([0], [0], color=colors[i % len(colors)], linestyle="--",
                                                label=f"Component μ = {m:.2f}") for m in gmm.means_.flatten()]
                            for d in dummy:
                                ax.add_line(d)

                ax.set_title(f"{cond} - {p} ({data_label})")
                ax.set_xlabel(p)
                ax.set_ylabel("Density")
                ax.legend()
                plt.tight_layout()
                plot_path = os.path.join(cond_folder, f"{simple_clean(p)}.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved {data_label} histogram for parameter '{p}' in condition '{cond}' to {plot_path}")

    # Generate side-by-side comparison plots between experimental conditions
        for data_label, data_dict in [("raw", conditions_data_raw), ("filtered", conditions_data_filtered)]:
            comp_folder = os.path.join(args.input_folder, "histograms", "comparisons", "no_fits", data_label)
            os.makedirs(comp_folder, exist_ok=True)

            condition_list = list(data_dict.keys())
            if len(condition_list) < 2:
                continue  # Skip if only one condition is present

            for p in parameters:
                plt.figure(figsize=(10, 7))
                ax = plt.gca()
                for i, cond in enumerate(condition_list):
                    # Collapse all group flags under each condition into one list of values
                    combined = []
                    for group in data_dict[cond].values():
                        combined.extend(group.get(p, []))
                    data = np.array(combined)
                    if data.size == 0:
                        continue

                    data[data <= 0] = 1
                    color = f"C{i}"
                    ax.hist(data, bins=30, density=True, alpha=0.5, color=color, label=cond)
                    overall_mean = data.mean()
                    ax.axvline(overall_mean, color=color, linestyle=":", linewidth=2,
                               label=f"{cond} mean = {overall_mean:.2f}")

                    gmm = fit_gmm(data)
                    if gmm:
                        for m in gmm.means_.flatten():
                            ax.axvline(m, color=color, linestyle="--", linewidth=1)
                        dummy = [plt.Line2D([0], [0], color=color, linestyle="--",
                                            label=f"{cond} comp μ = {m:.2f}") for m in gmm.means_.flatten()]
                        for d in dummy:
                            ax.add_line(d)

                ax.set_title(f"Comparison: {p} ({data_label})")
                ax.set_xlabel(p)
                ax.set_ylabel("Density")
                ax.legend()
                plt.tight_layout()
                plot_path = os.path.join(comp_folder, f"{simple_clean(p)}.png")
                plt.savefig(plot_path)
                plt.close()
# KDE-based logY dwell_time comparison for filtered data only
        data_label = "filtered"
        data_dict = conditions_data_filtered
        comp_folder = os.path.join(args.input_folder, "histograms", "comparisons", "no_fits", data_label)
        os.makedirs(comp_folder, exist_ok=True)

        condition_list = list(data_dict.keys())
        if len(condition_list) >= 2:
            condition_colors = {
                "3-Branched": "tab:blue",
                "4-Branched": "tab:orange"
            }

            p = "dwell_time"
            plt.figure(figsize=(10, 7))
            ax = plt.gca()

            for cond in condition_list:
                combined = []
                for group in data_dict[cond].values():
                    combined.extend(group.get(p, []))
                data = np.array(combined)
                if data.size == 0:
                    continue

                data[data <= 0] = 1  # avoid log(0)
                color = condition_colors.get(cond, "gray")

                # Compute KDE
                kde = gaussian_kde(data)
                x_vals = np.linspace(min(data), max(data), 500)
                y_vals = kde(x_vals)

                ax.plot(x_vals, y_vals, color=color, label=cond)
                ax.fill_between(x_vals, y_vals, alpha=0.3, color=color)

                overall_mean = data.mean()
                ax.axvline(overall_mean, color=color, linestyle=":", linewidth=2,
                           label=f"{cond} mean = {overall_mean:.2f}")

                gmm = fit_gmm(data)
                if gmm and gmm.means_.shape[0] >= 1:
                    first_mean = gmm.means_.flatten()[0]
                    ax.axvline(first_mean, color=color, linestyle="--", linewidth=1)
                    dummy = plt.Line2D([0], [0], color=color, linestyle="--",
                                       label=f"{cond} comp1 mean = {first_mean:.2f}")
                    ax.add_line(dummy)

            ax.set_title("Filtered dwell_time comparison — KDE log Y")
            ax.set_xlabel("dwell_time")
            ax.set_ylabel("KDE (log scale)")
            ax.set_yscale("log")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(comp_folder, "dwell_time_filtered_logY_KDE.png"))
            plt.close()

# Additional: Dwell time histogram (fraction per bin)
        frac_folder = os.path.join(args.input_folder, "histograms", "comparisons", "no_fits", "fraction_histogram")
        os.makedirs(frac_folder, exist_ok=True)
        p = "dwell_time"
        condition_list = list(conditions_data_raw.keys())

        for cond in condition_list:
            raw_vals = []
            for group in conditions_data_raw[cond].values():
                raw_vals.extend(group.get(p, []))
            filt_vals = []
            for group in conditions_data_filtered[cond].values():
                filt_vals.extend(group.get(p, []))
            all_vals = np.array(raw_vals + filt_vals)
            if all_vals.size == 0:
                continue

            all_vals = all_vals[all_vals > 0]  # remove zeros
            total_count = len(all_vals)
            bins = np.linspace(min(all_vals), max(all_vals), 30)
            counts, edges = np.histogram(all_vals, bins=bins)
            fractions = counts / total_count

            centers = 0.5 * (edges[1:] + edges[:-1])
            plt.figure(figsize=(8, 6))
            plt.bar(centers, fractions, width=np.diff(edges), align="center", color="gray", alpha=0.7)
            plt.title(f"Fractional dwell time histogram: {cond}")
            plt.xlabel("dwell_time")
            plt.ylabel("Fraction of total particles")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(frac_folder, f"{simple_clean(cond)}_dwell_time_fraction.png"))
            plt.close()

# Fractional dwell time histogram across conditions — comparisons
        comp_frac_folder = os.path.join(args.input_folder, "histograms", "comparisons", "no_fits", "fraction_comparison")
        os.makedirs(comp_frac_folder, exist_ok=True)
        p = "dwell_time"
        condition_list = list(conditions_data_raw.keys())
        if len(condition_list) >= 2:
            condition_colors = {
                "3-Branched": "tab:blue",
                "4-Branched": "tab:orange"
            }

            all_data = {}
            for cond in condition_list:
                raw_vals = []
                for group in conditions_data_raw[cond].values():
                    raw_vals.extend(group.get(p, []))
                filt_vals = []
                for group in conditions_data_filtered[cond].values():
                    filt_vals.extend(group.get(p, []))
                all_vals = np.array(raw_vals + filt_vals)
                all_vals = all_vals[all_vals > 0]
                if all_vals.size == 0:
                    continue
                all_data[cond] = all_vals

            # Unified bin range
            min_val = min([np.min(v) for v in all_data.values()])
            max_val = max([np.max(v) for v in all_data.values()])
            bins = np.linspace(min_val, max_val, 30)

            plt.figure(figsize=(10, 7))
            ax = plt.gca()

            for cond, vals in all_data.items():
                counts, edges = np.histogram(vals, bins=bins)
                fractions = counts / len(vals)
                centers = 0.5 * (edges[1:] + edges[:-1])
                color = condition_colors.get(cond, "gray")
                ax.plot(centers, fractions, marker="o", label=cond, color=color)

            ax.set_title("Fractional dwell time histogram — comparison")
            ax.set_xlabel("dwell_time")
            ax.set_ylabel("Fraction of total particles")
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(comp_frac_folder, "dwell_time_fraction_comparison.png"))
            plt.close()

# Fractional dwell time histogram across conditions — comparisons (histogram style)
        comp_frac_hist_folder = os.path.join(args.input_folder, "histograms", "comparisons", "no_fits", "fraction_comparison_hist")
        os.makedirs(comp_frac_hist_folder, exist_ok=True)
        p = "dwell_time"
        condition_list = list(conditions_data_raw.keys())
        if len(condition_list) >= 2:
            condition_colors = {
                "3-Branched": "tab:blue",
                "4-Branched": "tab:orange"
            }

            all_data = {}
            for cond in condition_list:
                raw_vals = []
                for group in conditions_data_raw[cond].values():
                    raw_vals.extend(group.get(p, []))
                filt_vals = []
                for group in conditions_data_filtered[cond].values():
                    filt_vals.extend(group.get(p, []))
                all_vals = np.array(raw_vals + filt_vals)
                all_vals = all_vals[all_vals > 0]
                if all_vals.size == 0:
                    continue
                all_data[cond] = all_vals

            # Unified bin range across all conditions
            min_val = min([np.min(v) for v in all_data.values()])
            max_val = max([np.max(v) for v in all_data.values()])
            bins = np.linspace(min_val, max_val, 15)
            bin_width = bins[1] - bins[0]
            centers = 0.5 * (bins[1:] + bins[:-1])

            plt.figure(figsize=(10, 7))
            ax = plt.gca()

            for i, (cond, vals) in enumerate(all_data.items()):
                counts, _ = np.histogram(vals, bins=bins)
                fractions = counts / len(vals)
                offset = bin_width * 0.3 * (i - len(all_data)/2)  # slight horizontal shift for visibility
                color = condition_colors.get(cond, "gray")
                ax.bar(centers + offset, fractions, width=bin_width * 0.3, alpha=0.7,
                       label=cond, color=color, align="center")

            ax.set_title("Fractional dwell time histogram — overlaid histogram")
            ax.set_xlabel("dwell_time")
            ax.set_ylabel("Fraction of total particles")
            ax.set_ylim(0, 1)

            ax.set_title("Fractional dwell time histogram — overlaid histogram", fontsize=18)
            ax.set_xlabel("dwell_time", fontsize=16)
            ax.set_ylabel("Fraction of total particles", fontsize=30)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=14)

            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(comp_frac_hist_folder, "dwell_time_fraction_comparison_hist.png"))
            plt.close()

            
if __name__ == "__main__":
    main()            
