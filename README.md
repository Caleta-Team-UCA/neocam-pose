# NeoCam: newborn pose analysis
By analysing the pose and motion of newborns, we can detect abnormal behaviours.

## Set up

To use this repository, you need to have an
[OAK-D camera](https://store.opencv.ai/products/oak-d) plugged into your computer.

### Create the environment using Conda

  1. Install miniconda
     
     ```
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash
     ```

     Say yes to everything and accept default locations. Refresh bash shell with `bash -l`

  2. Update conda
     
      ```
      conda update -n base -c defaults conda
      ```

  3. Clone this repository and cd into the folder

  4. Create and activate conda environment (removing previously existing env of the same name)
     
       ```
       conda remove --name neocam --all
       conda env create -f environment.yml --force
       conda activate neocam
       ```
 
## Scripts

Remember you have to activate the `neocam` environment before
running these scripts.

To test the motion detection on real time, run:

```
python neocam/process_cam.py
```

To test it on a video, run:

```
python neocam/process_video.py --path-video PATH-TO-VIDEO
```

