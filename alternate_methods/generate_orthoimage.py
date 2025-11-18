#! /usr/bin/env python

"""
Pipeline for constructing an orthoimage from image or video files for the Soo Locks project. 
NOTE: Only the video_folder OR image_folder argument is required. 

Usage: 
----------
python generate_orthoimage.py \
-video_folder <path_to_video_folder> \
-image_folder <path_to_image_folder> \
-target_datetime <YYYYMMDDHHMMSS> \
-inputs_folder <path_to_inputs_folder> \
-output_folder <path_to_output_folder> \
-refine_cameras <0/1> \
-output_res 0.003 \

"""

import argparse
import os
from glob import glob
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
# import the utility functions, assuming the ortho_utils.py file is in the root code directory
import ortho_utils


def getparser():
    parser = argparse.ArgumentParser(description=
                                     'Wrapper script to generate orthoimage from video frames or images for the Soo Locks project. '
                                     'NOTE: Specify video_folder OR image_folder as inputs. If video_folder specified, frames will be pulled at the target_datetime.')
    parser.add_argument('-video_folder', default=None, type=str, help='Path to folder containing video files (only video OR image folder needed)')
    parser.add_argument('-image_folder', default=None, type=str, help='Path to folder containing image files (only video OR image folder needed)')
    parser.add_argument('-target_datetime', default=None, type=str, help='Datetime at which to pull video frames')
    parser.add_argument('-inputs_folder', default=None, type=str, help='Path to folder containing standard input files')
    parser.add_argument('-output_folder', default=None, type=str, help='Path to folder where all outputs will be saved')
    parser.add_argument('-refine_cameras', default=1, type=int, choices=[0,1], help='Whether to re-calibrate cameras. Recommended if cameras have moved slightly since initial lidar scan.')
    return parser


def main():    
    # --- Parse the user arguments ---
    parser = getparser()
    args = parser.parse_args()
    video_folder = args.video_folder
    target_datetime = args.target_datetime
    image_folder = args.image_folder
    inputs_folder = args.inputs_folder
    output_folder = args.output_folder
    refine_cameras = bool(args.refine_cameras)

    # --- Define output folders ---
    out_folder = os.path.join(output_folder, 'soo_locks_photogrammetry_' + target_datetime)
    os.makedirs(out_folder, exist_ok=True)
    refined_cams_folder = os.path.join(out_folder, 'refined_cameras')
    ortho_folder = os.path.join(out_folder, 'orthoimages')

    # --- Set up logging ---
    # Define the timestamped log file name
    dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(out_folder, f'soo_locks_photogrammetry_{dt_now}.log')
    # Configure logging: writes to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Custom class to redirect print() output
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
        def write(self, message):
            message = message.strip()
            if message:
                self.level(message)
        def flush(self):
            pass
    # Redirect standard output and errors
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    # --- Get input files ---
    init_cams_file = os.path.join(inputs_folder, 'original_calibrated_cameras.csv')
    init_images_folder = os.path.join(inputs_folder, 'original_images')
    init_image_files = sorted(glob(os.path.join(init_images_folder, '*.tiff')))
    gcp_file = os.path.join(inputs_folder, 'GCP_merged_stable.gpkg')
    refdem_file = os.path.join(inputs_folder, 'lidar_DSM_filled_cropped.tif')
    closest_cam_map_file = os.path.join(inputs_folder, 'closest_camera_map.tiff')

    # Make sure they exist
    if not os.path.exists(inputs_folder):
        raise FileNotFoundError(f'Inputs folder not found: {inputs_folder}')
    if not os.path.exists(init_cams_file):
        raise FileNotFoundError(f'Original calibrated cameras not found: {init_cams_file}')
    if not os.path.exists(init_images_folder):
        raise FileNotFoundError(f"Original images folder not found: {init_images_folder}")
    if len(init_image_files) < 16:
        raise FileNotFoundError(f"Less than 16 original images found. Cannot continue.")
    if not os.path.exists(refdem_file):
        raise FileNotFoundError(f'Reference DEM file not found: {refdem_file}')
    if not os.path.exists(closest_cam_map_file):
        raise FileNotFoundError(f'Closest camera map file not found: {closest_cam_map_file}')
    
    # Make sure only image or video folder is specified, and that they exist
    if (video_folder is None) and (image_folder is None):
        raise ValueError('Either video_folder or image_folder must be specified.')
    if (video_folder is not None) and (image_folder is not None):
        raise ValueError('Only one of video_folder or image_folder should be specified.')
    if video_folder is not None:
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f'Video folder not found: {video_folder}')
        if not target_datetime:
            raise ValueError("target_datetime must be specified when inputting video files.")
    if image_folder is not None:
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f'Image folder not found: {image_folder}')

    # --- Run the workflow ---
    if video_folder:
        print('\n------------------------------------------')
        print('--- EXTRACTING FRAMES FROM VIDEO FILES ---')
        print('------------------------------------------\n')
        image_folder = os.path.join(out_folder, 'images')
        ortho_utils.process_video_files(
            video_files = sorted(glob(os.path.join(video_folder, '*.avi'))), 
            target_time_string = target_datetime,
            output_folder = image_folder
        )

    image_files = sorted(glob(os.path.join(image_folder, '*.tiff')))
    print(f"Found {len(image_files)} images to process.")

    if refine_cameras:
        print('\n-----------------------------')
        print('--- REFINING CAMERA POSES ---')
        print('-----------------------------\n')

        new_cams_file = ortho_utils.refine_camera_poses(
            image_files, init_image_files, init_cams_file, gcp_file, refined_cams_folder
        )

        refined_cams_file = new_cams_file
    else:
        refined_cams_file = init_cams_file

    print('\n------------------------------')
    print('--- ORTHORECTIFYING IMAGES ---')
    print('------------------------------\n')
    os.makedirs(ortho_folder, exist_ok=True)

    # Open the refined cameras file
    cams = pd.read_csv(refined_cams_file)
    for k in ['K', 'D', 'K_full', 'rvec', 'tvec']:
        cams[k] = cams[k].apply(literal_eval)

    # Iterate over image files
    for image_file in tqdm(image_files):
        # get channel from file name
        ch = 'ch' + os.path.basename(image_file).split('ch')[1][0:2]

        # check if output file already exists
        ortho_file = os.path.join(ortho_folder, f"{ch}_orthoimage.tiff")
        if os.path.exists(ortho_file):
            print('Orthoimage already exists, skipping.')
            continue

        # get the camera parameters
        ch = 'ch' + os.path.basename(image_file).split('ch')[1][0:2]
        cam = cams.loc[cams['channel']==ch]
        K = np.array(cam['K'].values[0])
        D = np.array(cam['D'].values[0])
        K_full = np.array(cam['K_full'].values[0])
        rvec = np.array(cam['rvec'].values[0])
        tvec = np.array(cam['tvec'].values[0])

        # orthorectify
        ortho_utils.orthorectify(image_file, refdem_file, K, D, K_full, rvec, tvec, ortho_file)
    
    print('\n------------------------------')
    print('--- MOSAICKING ORTHOIMAGES ---')
    print('------------------------------\n')
    ortho_utils.mosaic_orthoimages(
        image_files = sorted(glob(os.path.join(ortho_folder, '*.tiff'))), 
        closest_cam_map_file = closest_cam_map_file, 
        output_folder = out_folder,
        )
    
    print('\nDone! :)')
    

if __name__ == '__main__':
    main()