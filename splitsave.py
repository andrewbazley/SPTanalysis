import os
import numpy as np
import nd2reader
import tifffile
from concurrent.futures import ThreadPoolExecutor
import argparse

def extract_first_channel(arr):
    """
    Given a numpy array, if it is 3-dimensional this function attempts to
    select the channel dimension based on its size. Typically, the number of channels
    is small (e.g., less than 10) compared to height/width.
    It returns a 2D array that corresponds to the first channel.
    """
    if arr.ndim == 3:
        # Check if the first dimension is likely the channel dimension.
        if arr.shape[0] < 10:
            return arr[0, :, :]
        # Otherwise, if the last dimension is small, assume that is the channel dimension.
        elif arr.shape[-1] < 10:
            return arr[..., 0]
    return arr

def process_movie(file_path, output_dir):
    """
    Processes an ND2 file that does not have "BF" in its name.
    For each frame, only the first channel is kept.
    The output TIFF file is named by replacing the .nd2 extension with .tif.
    """
    base = os.path.basename(file_path)
    newimage = base.replace(".nd2", ".tif")
    try:
        with nd2reader.ND2Reader(file_path) as img:
            # If the file has multiple channels, try to restrict iteration to channel 0.
            if 'c' in img.sizes and img.sizes['c'] > 1:
                img.default_coords['c'] = 0
            frames = []
            for frame in img:
                arr = np.array(frame)
                arr = extract_first_channel(arr)
                frames.append(arr)
        output_array = np.array(frames)
        output_path = os.path.join(output_dir, newimage)
        tifffile.imwrite(output_path, output_array)
        print(f"Processed movie: {base} -> {newimage}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_image(file_path, output_dir):
    """
    Processes an ND2 file that has "BF" in its name.
    For BF images, we explicitly remove the second (and any additional) channel(s)
    so that only the first channel is kept.

    If the ND2 file contains multiple channels (indicated by the 'c' size),
    we assume that frames are interleaved, i.e. the frames are ordered by time 
    and channel. In that case, the total number of timepoints is determined as:
         timepoints = total_frames // number_of_channels.
    Then only the frame at index t * number_of_channels (i.e. channel 0) is saved for each time point.

    Additionally, the filename is modified by removing an internal '_BF_' substring
    and ensuring it starts with "BF_". For example:
         BF_DMSO-0.1pc_2h_BF_003.nd2  ->  BF_DMSO-0.1pc_2h_003.tif
    """
    base = os.path.basename(file_path)
    base_no_ext, _ = os.path.splitext(base)
    # Remove any internal occurrence of '_BF_'.
    modified_name = base_no_ext.replace("_BF_", "_")
    # Ensure the modified name starts with 'BF_'
    if not modified_name.startswith("BF_"):
        modified_name = "BF_" + modified_name
    newimage = modified_name + ".tif"

    try:
        with nd2reader.ND2Reader(file_path) as img:
            sizes = img.sizes
            frames = []
            if 'c' in sizes and sizes['c'] > 1:
                # Calculate timepoints assuming frames are interleaved by channel.
                num_channels = sizes['c']
                total_frames = len(img)
                timepoints = total_frames // num_channels
                for t in range(timepoints):
                    frame_index = t * num_channels  # select channel 0 for each timepoint
                    frame = np.array(img[frame_index])
                    # In case the returned array still has a channel dimension, extract the first channel.
                    frame = extract_first_channel(frame)
                    frames.append(frame)
            else:
                # If there is only one channel in the metadata,
                # simply process each frame normally.
                for frame in img:
                    arr = np.array(frame)
                    arr = extract_first_channel(arr)
                    frames.append(arr)
        output_array = np.array(frames)
        output_path = os.path.join(output_dir, newimage)
        tifffile.imwrite(output_path, output_array)
        print(f"Processed BF image: {base} -> {newimage}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main(directory):
    # Create the output directory 'tif-files' inside the given directory.
    output_dir = os.path.join(directory, "tif-files")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of ND2 files in the directory.
    files = [f for f in os.listdir(directory) if f.lower().endswith(".nd2")]
    file_paths = [os.path.join(directory, f) for f in files]
    
    # Process each file concurrently using multithreading.
    # The ThreadPoolExecutor is not given an explicit max_workers number,
    # so it will use the default (which is generally based on os.cpu_count()).
    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            base = os.path.basename(file_path)
            if "BF" in base:
                futures.append(executor.submit(process_image, file_path, output_dir))
            else:
                futures.append(executor.submit(process_movie, file_path, output_dir))
        # Wait for all tasks to complete.
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ND2 files into single-channel TIFF files using multithreading. "
                    "For files with 'BF' in the name, only the first channel is saved and the filename is modified."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.getcwd(),
        help="Directory containing ND2 files (default: current working directory)."
    )
    args = parser.parse_args()
    main(args.directory)
