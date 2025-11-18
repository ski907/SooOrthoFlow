#! /usr/bin/env python

"""
Utility functions for generating an orthoimage from video files for the Soo Locks project.
"""

import os
from glob import glob
import numpy as np
import cv2
import rioxarray as rxr
import xarray as xr
import rasterio as rio
import datetime
import dask.array as da
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import json
# Ignore warnings (rasterio throws a warning whenever an image is not georeferenced. Annoying in this case.)
import warnings
warnings.filterwarnings('ignore')


def string_to_datetime(
        datetime_string: str = None
        ):
    """
    Convert a datetime string in the format 'YYYYMMDD_HHMMSS' to a datetime object.
    
    Parameters
    ----------
    datetime_string: str
        datetime string in the format 'YYYYMMDD_HHMMSS'

    Returns
    ----------
    datetime.datetime
        corresponding datetime object
    """
    return datetime.datetime(
        int(datetime_string[0:4]), 
        int(datetime_string[4:6]),
        int(datetime_string[6:8]),
        int(datetime_string[8:10]),
        int(datetime_string[10:12]),
        int(datetime_string[12:14])
        )


def extract_frame_at_clock_time(
        video_file: str = None, 
        target_time_string: str = None, 
        output_folder: str = None 
        ) -> bool:
    """
    Extract a single frame from a video file at the specified clock time.

    Parameters
    ----------
    video_file: str
        path to the video file
    target_time_string: str
        target clock time in the format 'YYYYMMDD_HHMMSS'
    output_folder: str
        folder where the extracted frame will be saved

    Returns
    ----------
    success: bool
        True if frame was successfully extracted and saved, False otherwise
    """
    # parse start and end times from video file name
    start_time_string = os.path.basename(video_file).split('_')[3]
    end_time_string = os.path.splitext(os.path.basename(video_file))[0].split('_')[4].split('(')[0]

    # convert datetime strings to datetime objects
    target_time = string_to_datetime(target_time_string)
    start_time = string_to_datetime(start_time_string)
    end_time = string_to_datetime(end_time_string)

    print(f"\nProcessing {video_file}")
    print(f'Detected video time range: {start_time} to {end_time}')

    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return False

    # Determine the video time duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Check if the target time is beyond the video coverage
    time_offset = (target_time - start_time).total_seconds()
    if time_offset < 0 or (time_offset > duration):
        print(f"Error: Target time {time_offset:.2f}s is outside video duration ({duration:.2f}s)")
        cap.release()
        return False

    # Otherwise, get the appropriate frame
    frame_number = int(time_offset * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not extract frame at {time_offset:.2f}s")
        cap.release()
        return False
    
    # Determine the camera number
    ch = os.path.basename(video_file).split('ch')[1][0:2]
    if (frame.shape[1] > 4000) & (ch=='1_'):
        ch = '09'
    elif (frame.shape[1] > 4000):
        ch = str(int(ch[0]) + 8)
    else:
        ch = '0' + ch[0]

    # Save to file
    output_image_file = os.path.join(
        output_folder, 
        f"ch{ch}_{target_time_string}.tiff"
        )
    
    # make sure frame has int datatype
    frame = frame.astype(np.int8)

    if cv2.imwrite(output_image_file, frame):
        print(f"Extracted frame -> {output_image_file}")
        cap.release()
        return True
    else:
        print(f"Failed to save frame")
        cap.release()
        return False
    

def process_video_files(
        video_files: list[str] = None, 
        target_time_string: str = None, 
        output_folder: str = None, 
        ) -> None:
    """
    Extract frames from a list of video files at the specified clock time.

    Parameters
    ----------
    video_files: list[str]
        list of paths to video files
    target_time_string: str
        target clock time in the format 'YYYYMMDD_HHMMSS'
    output_folder: str
        folder where the extracted frames will be saved

    Returns
    ----------
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    print('Target time:', string_to_datetime(target_time_string))

    # Iterate over video files
    for video_file in video_files:
        extract_frame_at_clock_time(video_file, target_time_string, output_folder)
    return


class GCPSelector:
    def __init__(self, img1_file, img2_file, gcp1, zoom=100):
        # Load grayscale images
        self.img1_file = img1_file
        self.img1 = plt.imread(img1_file)
        if self.img1.ndim == 3:
            self.img1 = np.mean(self.img1, axis=2)
        self.img2_file = img2_file
        self.img2 = plt.imread(img2_file)
        if self.img2.ndim == 3:
            self.img2 = np.mean(self.img2, axis=2)

        self.gcp1 = gcp1
        self.gcp2 = gcp1.copy()
        self.zoom = zoom
        self.clicked_points = [None] * len(self.gcp2)
        self.current_index = 0

        # Setup figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)

        print(
            "Instructions:\n"
            " - Click on the right image to record a new GCP location.\n"
            " - Press 'k' to skip if feature not visible.\n"
            " - Press 'b' to go back and redo previous GCP.\n"
        )

        self.update_display()
        plt.show()

    def update_display(self):
        """Show zoomed-in patches around current GCP"""
        if self.current_index >= len(self.gcp1):
            print("All GCPs processed.")
            plt.close(self.fig)
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        u, v = self.gcp1.iloc[self.current_index][['col_sample','row_sample']]
        u, v = int(u), int(v)
        half = self.zoom

        # Crop images to around GCP
        x1, x2 = max(0, u - half), min(self.img1.shape[1], u + half)
        y1, y2 = max(0, v - half), min(self.img1.shape[0], v + half)
        crop1 = self.img1[y1:y2, x1:x2]
        crop2 = self.img2[y1:y2, x1:x2]

        # Show zoomed-in crops
        self.ax1.imshow(crop1, cmap='gray', extent=[x1, x2, y2, y1])
        self.ax1.set_title(f"Original GCP #{self.current_index}")
        self.ax1.plot(u, v, 'om', markersize=6)

        self.ax2.imshow(crop2, cmap='gray', extent=[x1, x2, y2, y1])
        self.ax2.set_title("Click New Location")

        self.fig.suptitle(self.img2_file)

        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(u - half, u + half)
            ax.set_ylim(v + half, v - half)

        self.fig.canvas.draw()

    def onclick(self, event):
        if event.inaxes != self.ax2:
            return
        u, v = event.xdata, event.ydata
        self.clicked_points[self.current_index] = (u, v)
        self.gcp2.loc[self.current_index, ['col_sample', 'row_sample']] = u, v
        self.current_index += 1
        self.update_display()

    def onkey(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'k':  # skip
            self.clicked_points[self.current_index] = None
            self.current_index += 1
            self.update_display()
        elif event.key == 'b':  # back
            if self.current_index > 0:
                self.current_index -= 1
                self.update_display()
        elif event.key == 'q':
            print("User quit early.")
            plt.close(self.fig)

    def get_results(self, out_file=None):
        """Return/save GCP2 with updated pixel coordinates, with skipped GCPs as NaN"""
        gcp2 = self.gcp2
        gcp2 = gcp2.dropna().reset_index(drop=True)
        # print(f"Compiled {len(gcp2)} updated GCPs")

        if out_file:
            gcp2.to_file(out_file)
            print(f'Updated GCP saved to:\n{out_file}')

        return gcp2


def solve_new_pose(
        gcp_df: gpd.GeoDataFrame = None, 
        K: np.array = None, 
        D: np.array = None, 
        rvec1: np.array = None, 
        tvec1: np.array = None
        ) -> tuple[np.array, np.array]:
    """
    Solve for a new camera pose given updated GCP locations.

    Parameters
    ----------
    gcp_df : gpd.GeoDataFrame
        GeoDataFrame containing GCPs with columns 'X', 'Y', 'Z', 'col_sample', 'row_sample'
    K : np.array
        Camera intrinsic matrix
    D : np.array
        Camera distortion coefficients
    rvec1 : np.array
        Initial rotation vector
    tvec1 : np.array
        Initial translation vector

    Returns
    ----------
    rvec2 : np.array
        Refined rotation vector
    tvec2 : np.array
        Refined translation vector
    """
    # Prepare GCPs and image points
    obj_pts = gcp_df[['X', 'Y', 'Z']].values.astype(np.float32)
    img_pts = gcp_df[['col_sample', 'row_sample']].values.astype(np.float32)

    # Undistort the image points
    img_pts_undistorted = cv2.fisheye.undistortPoints(
        img_pts.reshape(-1, 1, 2), K, D, None, K
    ).reshape(-1, 2)

    # Convert rvec1, tvec1 to rotation matrix
    R1, _ = cv2.Rodrigues(rvec1)
    tvec1 = tvec1.reshape(3, 1)

    # Project GCPs into the camera frame using initial pose
    # world -> camera
    obj_cam = (R1 @ obj_pts.T + tvec1).T

    # Triangulate new camera-frame coordinates based on observed pixels
    # Backproject undistorted pixels into 3D rays, then scale by depth of old model
    pts_cam_dir = np.concatenate([img_pts_undistorted, np.ones((len(img_pts_undistorted), 1))], axis=1)
    pts_cam_dir = np.linalg.inv(K) @ pts_cam_dir.T
    pts_cam_dir = pts_cam_dir.T

    # Use existing depths from old pose to approximate new camera-frame coordinates
    depths = obj_cam[:, 2:3]
    obj_cam_new = pts_cam_dir * depths

    # Compute best-fit rigid transform (dR, dt) between obj_cam_new and obj_cam
    mu_old = obj_cam.mean(axis=0)
    mu_new = obj_cam_new.mean(axis=0)
    X0 = obj_cam - mu_old
    X1 = obj_cam_new - mu_new

    U, _, Vt = np.linalg.svd(X0.T @ X1)
    R_delta = Vt.T @ U.T
    if np.linalg.det(R_delta) < 0:  # ensure a right-handed rotation
        Vt[-1, :] *= -1
        R_delta = Vt.T @ U.T
    t_delta = mu_new - R_delta @ mu_old

    # Construct new pose
    R2 = R_delta @ R1
    tvec2 = R_delta @ tvec1 + t_delta.reshape(3, 1)
    rvec2, _ = cv2.Rodrigues(R2)

    return rvec2, tvec2


def refine_camera_poses(
        image_files: list[str] = None, 
        init_image_files: list[str] = None, 
        init_cams_file: str = None, 
        init_gcp_file: str = None, 
        out_folder: str = None
        ) -> str:
    """
    Refine camera poses for a set of images using user-selected GCPs.

    Parameters
    ----------
    image_files : list[str]
        List of paths to images to refine
    init_image_files : list[str]
        List of paths to initial images corresponding to image_files
    init_cams_file : str
        Path to CSV file containing initial camera parameters
    init_gcp_file : str
        Path to GPKG file containing initial GCPs
    out_folder : str
        Folder to save refined camera parameters

    Returns
    ----------
    new_cams_file : str
        Path to CSV file containing refined camera parameters
    """

    os.makedirs(out_folder, exist_ok=True)

    # Load initial GCPs
    gcp = gpd.read_file(init_gcp_file, layer='gcp_merged_stable')
    gcp = gcp.dropna().reset_index(drop=True)
    gcp['channel'] = [f"ch0{ch}" if ch < 10 else f"ch{ch}" for ch in gcp['channel']]

    # Load initial camera specs
    init_cams = pd.read_csv(init_cams_file)
    for k in ['K', 'D', 'K_full', 'rvec', 'tvec']:
        init_cams[k] = init_cams[k].apply(literal_eval)
        
    # Iterate over images
    for i, image_file in enumerate(image_files):
        # Check if output file already exists
        ch = "ch" + os.path.basename(image_file).split('ch')[1][0:2]
        new_cam_file = os.path.join(out_folder, f"{ch}_refined_camera.csv")
        if os.path.exists(new_cam_file):
            print('Refined camera pose already exists in file, skipping.')
            continue

        # Get the original image
        init_image_file = [x for x in init_image_files if ch in os.path.basename(x)][0]

        # Subset the initial GCP
        init_gcp = gcp.loc[gcp['channel']==ch]

        # Get camera intrinsics
        init_cam = init_cams.loc[init_cams['channel']==ch]
        K = np.array(init_cam['K'].values[0])
        D = np.array(init_cam['D'].values[0])
        K_full = np.array(init_cam['K_full'].values[0])
        rvec1 = np.array(init_cam['rvec'].values[0])
        tvec1 = np.array(init_cam['tvec'].values[0])

        # Check if updated GCP already exist
        new_gcp_file = os.path.join(out_folder, f"{ch}_updated_GCP.gpkg")
        if not os.path.exists(new_gcp_file):

            # Run interactive GCP selection
            selector = GCPSelector(
                init_image_file, image_file, init_gcp, zoom=100
            )

            # Get the updated GCP
            new_gcp = selector.get_results(out_file=new_gcp_file)
        else:
            new_gcp = gpd.read_file(new_gcp_file)

        # Refine camera pose
        rvec2, tvec2 = solve_new_pose(
            new_gcp, K, D, rvec1, tvec1
        )

        # Save new camera specs
        new_cam = pd.DataFrame({
            'channel': [ch],
            'K': [json.dumps(K.tolist())],
            'D': [json.dumps(D.tolist())],
            'K_full': [json.dumps(K_full.tolist())],
            'rvec': [json.dumps(rvec2.tolist())],
            'tvec': [json.dumps(tvec2.tolist())]
        }, index=[i])
        new_cam.to_csv(new_cam_file, index=False)
        print(f'Refined camera saved to:\n{new_cam_file}')
        
    # Merge refined cameras into one file
    new_cam_files = glob(os.path.join(out_folder, '*refined_camera.csv'))
    new_cams_list = []
    for new_cam_file in new_cam_files:
        new_cams_list += [pd.read_csv(new_cam_file)]
    new_cams = pd.concat(new_cams_list).reset_index(drop=True)
    new_cams_file = os.path.join(out_folder, "refined_cameras_merged.csv")
    new_cams.to_csv(new_cams_file, index=False)
    print(f'Merged refined cameras and saved to:\n{new_cams_file}')

    # Remove intermediary files
    # for new_cam_file in new_cam_files:
    #     os.remove(new_cam_file)

    return new_cams_file


def undistort_image(
        img: np.array = None, 
        K: np.array = None, 
        D: np.array = None, 
        K_full: np.array = None, 
        mask_nodata: bool = True
        ) -> np.array:
    """
    Undistort an image using fisheye model with full FOV intrinsics.

    Parameters
    ----------
    img : np.array
        Input distorted image
    K : np.array
        Camera intrinsic matrix
    D : np.array
        Camera distortion coefficients
    K_full : np.array
        Full FOV camera intrinsic matrix
    mask_nodata : bool
        Whether to mask invalid pixels as NaN

    Returns
    ----------
    img_undistorted : np.array
        Undistorted image
    """
    # Undistort with full FOV
    h,w = img.shape
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_full, (w,h), cv2.CV_32FC1)
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Mask invalid pixels
    if mask_nodata:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask_undistorted = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
        valid_mask = mask_undistorted > 0
        img_undistorted = img_undistorted.astype(np.float32)
        img_undistorted[~valid_mask] = np.nan

    return img_undistorted


def orthorectify(
        image_file: str = None, 
        dem_file: str = None, 
        K: np.array = None, 
        D: np.array = None, 
        K_full: np.array = None, 
        rvec: np.array = None, 
        tvec: np.array = None, 
        out_file: str = None
        ) -> xr.DataArray:
    """
    Orthorectify an image using the provided DEM and camera parameters.

    Parameters
    ----------
    image_file : str
        Path to the input image file
    dem_file : str
        Path to the DEM file
    K : np.array
        Camera intrinsic matrix
    D : np.array
        Camera distortion coefficients
    K_full : np.array
        Full FOV camera intrinsic matrix
    rvec : np.array
        Rotation vector
    tvec : np.array
        Translation vector
    out_file : str
        Path to save the orthorectified image

    Returns
    ----------
    ortho_xr : xr.DataArray
        Orthorectified image as an xarray DataArray
    """
    # Undistort the image
    image = rxr.open_rasterio(image_file).isel(band=0).data
    image_undistorted = undistort_image(image, K, D, K_full)

    # Build coordinate grid from DEM
    dem = rxr.open_rasterio(dem_file).squeeze()
    crs = dem.rio.crs
    dem = xr.where(dem==-9999, np.nan, dem)
    dem_z = dem.data.astype(np.float32)
    X, Y = np.meshgrid(dem.x.data, dem.y.data)
    h_img, w_img = image_undistorted.shape[:2]
    ortho = np.full_like(dem_z, np.nan, dtype=np.float32)

    # Calculate rotation matrix 
    R, _ = cv2.Rodrigues(rvec)

    # Flatten world points
    world_pts = np.stack([X.ravel(), Y.ravel(), dem_z.ravel()], axis=1)

    # Transform world→camera
    cam_pts = (R @ world_pts.T + tvec).T

    # Prevent divide-by-zero (points behind camera)
    valid = cam_pts[:, 2] > 0

    # Project onto image plane
    uv = (K_full @ (cam_pts[valid].T / cam_pts[valid, 2])).T  # (N_valid,3)
    u = uv[:, 0]
    v = uv[:, 1]

    # Fill map arrays
    map_x = np.full_like(dem_z, np.nan, dtype=np.float32)
    map_y = np.full_like(dem_z, np.nan, dtype=np.float32)
    map_x.ravel()[valid] = u
    map_y.ravel()[valid] = v

    # Sample from the undistorted image
    safe_map_x = map_x.copy()
    safe_map_y = map_y.copy()
    safe_map_x[np.isnan(safe_map_x)] = -1
    safe_map_y[np.isnan(safe_map_y)] = -1

    sampled = cv2.remap(
        image_undistorted.astype(np.float32),
        safe_map_x, safe_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan
    )

    # Mask invalid or out-of-bounds regions
    in_bounds = (
        (map_x >= 0) & (map_x < w_img) &
        (map_y >= 0) & (map_y < h_img)
    )
    ortho[in_bounds] = sampled[in_bounds]

    # Return as xarray
    ortho_xr = xr.DataArray(
        ortho,
        dims=('y', 'x'),
        coords={'x': dem.x.data, 'y': dem.y.data}
    )

    # Remove totally empty rows and columns
    ortho_xr = ortho_xr.dropna(dim='x', how='all').dropna(dim='y', how='all')

    # Save to file
    if out_file:
        # convert datatype to int
        nodata_val = 9999
        ortho_xr = xr.where(np.isnan(ortho_xr), nodata_val, ortho_xr)
        ortho_xr = ortho_xr.astype(np.uint16)
        # assign CRS and nodata
        ortho_xr = ortho_xr.rio.write_nodata(nodata_val)
        ortho_xr = ortho_xr.rio.write_crs(crs)
        # ensure orientation matches DEM (north-up)
        if dem.rio.resolution()[1] < 0:
            ortho_xr = ortho_xr.sortby('y', ascending=False)
        # save with integer compression
        ortho_xr.rio.to_raster(
            out_file,
            dtype='uint16'
        )
        print(f"Orthorectified image saved to:\n{out_file}")

    return ortho_xr


def mosaic_orthoimages(
    image_files: list[str] = None, 
    closest_cam_map_file: str = None, 
    output_folder: str = None,
    chunk_size: int | str = 2048
) -> None:
    """
    Mosaic orthorectified images by sampling the closest_cam_map_file. 

    Parameters
    ----------
    image_files : list[str]
        List of paths to orthorectified images
    closest_cam_map_file : str
        Path to the closest camera map
    output_folder : str
        Folder to save the mosaic
    chunk_size : int | str
        Chunk size for Dask arrays, passed to rioxarray.open_rasterio
    """
    os.makedirs(output_folder, exist_ok=True)

    print("Reading closest camera map")
    closest_cam_map = rxr.open_rasterio(closest_cam_map_file, chunks=chunk_size)
    crs = closest_cam_map.rio.crs

    print("Reading orthoimages with Dask")
    datasets = [rxr.open_rasterio(f, masked=True, chunks=chunk_size) for f in image_files]

    # Verify consistent CRS
    for ds in datasets:
        if ds.rio.crs != crs:
            ds = ds.rio.reproject(crs)

    num_bands = datasets[0].rio.count
    print(f"Detected {num_bands} band(s) per image")

    # Determine target resolution
    res_x = np.mean([abs(ds.rio.resolution()[0]) for ds in datasets])
    res_y = np.mean([abs(ds.rio.resolution()[1]) for ds in datasets])
    print(f"Target resolution: {res_x:.3f}, {res_y:.3f}")

    # Output bounds & transform
    bounds = closest_cam_map.rio.bounds()
    width = int((bounds[2] - bounds[0]) / res_x)
    height = int((bounds[3] - bounds[1]) / res_y)
    transform = rio.transform.from_bounds(*bounds, width=width, height=height)

    # Dummy grid
    dummy_grid = xr.DataArray(
        da.full((height, width), 9999, dtype=np.uint16, chunks=(chunk_size, chunk_size)),
        dims=("y", "x"),
        coords={
            "y": np.linspace(bounds[3], bounds[1], height),
            "x": np.linspace(bounds[0], bounds[2], width),
        },
    ).rio.write_crs(crs).rio.write_transform(transform)

    # Reproject images lazily with dask
    print("Reprojecting images to target grid...")
    reprojected = [
        ds.rio.reproject_match(dummy_grid, resampling=rio.enums.Resampling.nearest)
        for ds in datasets
    ]

    stack = xr.concat(reprojected, dim="camera")

    # Reproject closest_cam_map lazily
    closest_cam_map = closest_cam_map.rio.reproject_match(dummy_grid, resampling=rio.enums.Resampling.nearest)

    # Initialize mosaic with dask array
    print("Creating mosaic")
    mosaic_shape = (num_bands, height, width)
    mosaic = xr.DataArray(
        da.full(mosaic_shape, 9999, dtype=np.uint16, chunks=(1, chunk_size, chunk_size)),
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, num_bands + 1), "y": dummy_grid.y, "x": dummy_grid.x},
    ).rio.write_crs(crs).rio.write_transform(transform)

    # Fill mosaic lazily using dask.where
    for i in range(len(stack.camera)):
        mask = (closest_cam_map.squeeze() == i)
        if num_bands == 1:
            mosaic = xr.where(mask, stack.isel(camera=i)[0], mosaic)
        else:
            for b in range(num_bands):
                mosaic[b] = xr.where(mask, stack.isel(camera=i, band=b), mosaic[b])
    
    # Make sure dimensions are in correct order
    mosaic = mosaic.transpose('band', 'y', 'x')

    # Make sure no data value, data type, and CRS are properly set
    mosaic = mosaic.astype(np.uint16)
    mosaic = mosaic.rio.write_nodata(9999)
    mosaic = mosaic.rio.write_crs(crs)

    # Save mosaic (compute in chunks)
    mosaic_file = os.path.join(output_folder, "orthomosaic.tiff")
    print("Saving mosaic...")
    mosaic.rio.to_raster(mosaic_file, compute=True)
    print("Saved orthomosaic:", mosaic_file)
    
    return
