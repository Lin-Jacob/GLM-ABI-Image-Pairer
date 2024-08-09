# NOAA_Internship

# Libraries
To replicate my exact setup, please have the following libraries and their versions:
```
Flask==3.0.3
matplotlib==3.7.2
netCDF4==1.6.5
netCDF4==1.7.1.post1
numpy==1.25.0
opencv_contrib_python==4.10.0.84
opencv_python==4.9.0.80
opencv_python_headless==4.9.0.80
Pillow==10.0.0
Pillow==10.4.0
torch==2.3.1+cu118
```
which can be found in the requirements.txt file as well.

You can also do the following:
```
cd to the repository
pip install -r requirements.txt
```

# To use this repository, run the visualize_nc.py folder first with the .nc files in there, then run homography_assessment.py

# Information
Some information about the repository:
- Take the .nc files and place them in the data folder, they should be in the same naming format as the same files in there.
- visualize_nc.py: takes the .nc files and converts them into image files (.png files) and saves them into the images folder. Set host_image to true if you want to see the list of images hosted in a local website
- homography_assessment.py: runs SuperPoint and SuperGlue from MagicLeap and runs a homography assessment on it
- Folders:
  - matched_output folder: folder of matched images side by side
  - predicted_kp: Shows the predicted key points from the homography assessment on the ABI image, the red dots are the actual key points, the blue dots are the predicted key points
  - assessment_histogram: Shows the relative histogram of each pair of matching
  - images: images from the .nc files
  - data: should be where the .nc files be dumped into
  - reports: All of the weekly reports

