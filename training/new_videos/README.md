# New recordings (Colab collection notebooks)

One folder per word, filled from the Colab notebooks in `training/collection/notebooks/`.
The notebooks save to Google Drive (`MyDrive/ASL_collection/<word>/`); copy each word folder's
`.mp4` and `.npy` files here, plus `manifest.csv` into this folder.

The old dataset in `training/raw_videos/` is kept untouched for now. Do not mix the two: the new
takes follow the fixed capture spec (640x360, 30 fps, 90 frames) and the old videos do not.
