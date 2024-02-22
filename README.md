#### `main()`

This is the main execution function of the program. It uses argparse to receive the following arguments:

- `--crop`: If this option is enabled, the input video will be cropped using a crop algorithm.
- `--sample`: Sets the frame sampling interval when the crop option is enabled. The default value is 10.
- `--save`: If this option is enabled, the cropped video file will be saved.
- `--mode`: Sets the operation mode. You must select either 'video' or 'eval'.
- `--path`: Sets the video path in 'video' mode.

Below is an example command to illustrate how to run this program. This command enables the `crop` option, sets the sampling interval to 5, and saves the cropped video file. The operation mode is set to `video`, and the video path is set to `./example.mp4`.

```bash
python main.py --crop --sample 5 --save --mode video --path ./example.mp4
```

Based on the input arguments, the function performs the following processes:

- If `video` mode is selected, frames are extracted from the video path specified by the `path` argument. If the `path` argument is empty, an error message is displayed and the program is terminated.

- If the `crop` option is enabled, the `Frames Extracting with Crop` message is displayed, and the `loader.find_crop_area` function is used to find the area to be cropped in the video. The frames are sampled at intervals specified by the `sample` argument. The `loader.get_crop_frames` function is used to crop the video based on the found area, and the cropped frames and frames per second (fps) of the video are returned. If the `save` option is enabled, the cropped video file is saved.

- If the `crop` option is disabled, the `Frames Extracting without Crop` message is displayed, and the `loader.get_frames` function is used to extract frames from the video. The extracted frames and frames per second (fps) of the video are returned.

- You can check `crop` algorithm in [here](https://github.com/jinkyusung/human-centric-video-crop).

From the returned list of frames, hand-clapping, head-banging, and sitting/standing actions are detected:

- The `handclap.model` function is used to detect the action of clapping hands in the input frame list. The detection results are stored in [`handclap_detections`](https://github.com/jinkyusung/handclap-detection). 

- The `headbanging.model` function is used to detect the action of shaking the head in the input frame list. The detection results are stored in [`headbanging_detections`](https://github.com/dongseok1220/detect-headbanging). 

- The `sitstand.model` function is used to detect the action of sitting and standing up in the input frame list. This function takes the frames and frames per second (fps) as arguments. The detection results are stored in [`sitstand_detections`](https://github.com/Ohphara/SitStand_Tracker).

The detected actions are unified into a single dataframe and saved as `./unified.csv`.

