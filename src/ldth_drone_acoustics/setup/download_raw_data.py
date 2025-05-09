from typing import Final

import numpy as np
import requests
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ldth_drone_acoustics import CLASSES, get_clz_dir

URL_BASE: Final[str] = (
    "https://raw.githubusercontent.com/DroneDetectionThesis/Drone-detection-dataset/master/Data/Audio"
)
NUM_PER_CLASS: Final[int] = 30
_, TEST_IDXS = train_test_split(np.arange(NUM_PER_CLASS), test_size=0.2, random_state=42)


for class_name in CLASSES:
    for idx in tqdm(range(NUM_PER_CLASS), desc=f"Downloading data for {class_name}"):
        # Setup output dir.
        output_dir = get_clz_dir("test" if idx in TEST_IDXS else "train", class_name)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Download file from remote.
        file_stem = f"{class_name.upper()}_{idx + 1:03d}"
        local_path = output_dir / f"{file_stem}.wav"
        remote_url = f"{URL_BASE}/{file_stem}.wav"
        response = requests.get(remote_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

        # Read the stereo audio file. 'data' is a NumPy array with shape (n_samples, channels)
        data, samplerate = sf.read(local_path)

        # For a 10-second file, we assume the total number of samples is samplerate * 10.
        # We want to split the file into two 5 second segments.
        segment_samples = int(samplerate * 5)  # Number of samples in 5 seconds

        # Extract left channel segments.
        # Channel indexing: 0 for left, 1 for right.
        left1 = data[:segment_samples, 0]  # Left channel from 0 to 5 seconds.
        left2 = data[segment_samples : 2 * segment_samples, 0]  # Left channel from 5 to 10 seconds.

        # Extract right channel segments.
        right1 = data[:segment_samples, 1]  # Right channel from 0 to 5 seconds.
        right2 = data[segment_samples : 2 * segment_samples, 1]  # Right channel from 5 to 10 seconds.

        # Write the four segments to separate WAV files.
        sf.write(output_dir / f"{file_stem}_L1.wav", left1, samplerate)
        sf.write(output_dir / f"{file_stem}_L2.wav", left2, samplerate)
        sf.write(output_dir / f"{file_stem}_R1.wav", right1, samplerate)
        sf.write(output_dir / f"{file_stem}_R2.wav", right2, samplerate)

        # Delete the original file - we just want to keep the four segments.
        local_path.unlink()
