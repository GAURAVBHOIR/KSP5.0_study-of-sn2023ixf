#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import warnings
import os
import subprocess

# Suppress warnings
warnings.filterwarnings("ignore")

# Get folder locations from the user
corr_folder_path = input("Enter the path to the folder containing the corr files: ")
wcs_folder_path = input("Enter the path to the folder containing the wcs files: ")
astrom_folder_path = input("Enter the path to the folder containing the astrometry files: ")

# Get the list of files in each folder
corr_files = [os.path.join(corr_folder_path, f) for f in os.listdir(corr_folder_path) if f.endswith('.fits')]
wcs_files = [os.path.join(wcs_folder_path, f) for f in os.listdir(wcs_folder_path) if f.endswith('.fits')]
astrom_files = [os.path.join(astrom_folder_path, f) for f in os.listdir(astrom_folder_path) if f.endswith('.fits')]

# Sort the files for consistent processing
corr_files.sort()
wcs_files.sort()
astrom_files.sort()

# Define file paths for SWarp processing
swarpConfigFile = input("Enter the path to the SWarp configuration file: ")
procFolder = input("Enter the path to the processing folder: ")
weightFiles = [os.path.join(procFolder, f'weight_file_{i+1}.wcs.proc.fits') for i in range(len(astrom_files))]
resampledFiles = [os.path.join(procFolder, f'resampled_file_{i+1}.wcs.proc.fits') for i in range(len(astrom_files))]
swarpCommand = 'swarp'  # Adjust if SWarp command is different

# Loop through each file to process
for i, (corr_file, wcs_file, astrom_file, weight_file, resampled_file) in enumerate(zip(corr_files, wcs_files, astrom_files, weightFiles, resampledFiles)):
    # Open and read the FITS file
    corr2 = fits.open(corr_file)
    data = corr2[1].data
    table = Table(data)
    print(table)
    
    # Read the astrometry image file
    with fits.open(astrom_file) as HDUList:
        img = HDUList[0].data

    # Load the WCS transformation
    wcs = WCS(wcs_file)

    # Get the RA and DEC from the data table
    ra = data['field_ra']
    dec = data['field_dec']

    # Convert world coordinates (RA, DEC) to pixel coordinates
    coordsx, coordsy = wcs.all_world2pix(ra, dec, 0)

    # Create a list of circle patches for plotting
    patch_list = [Circle((x, y), radius=20, fill=False, ec='C1') for x, y in zip(coordsx, coordsy)]

    # Plot the image with WCS projection and overlay the circles
    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot(projection=wcs)
    mean, median, std = sigma_clipped_stats(img)
    plt.imshow(img, vmin=median - 3 * std, vmax=median + 10 * std, origin="lower")

    for p in patch_list:
        ax.add_patch(p)

    plt.show()

    # Run the SWarp command
    try:
        command = f'{swarpCommand} {astrom_file} -c {swarpConfigFile} -IMAGEOUT_NAME {resampled_file} -WEIGHTOUT_NAME {weight_file} -RESAMPLE_DIR {procFolder}'
        print(f"Executing command: {command}")
        print("Processing...")
        rval = subprocess.run(command.split(), check=True, capture_output=True)
        print("Process completed.")
        print(rval.stdout.decode())

    except subprocess.CalledProcessError as err:
        print('Could not run SWarp. Can you run it from the terminal?')

# Display the resampled images
for resampled_file in resampledFiles:
    HDUList = fits.open(resampled_file)
    wcs = WCS(HDUList[0].header)
    resampledData = HDUList[0].data
    HDUList.close()

    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot(projection=wcs)
    mean, median, std = sigma_clipped_stats(resampledData)
    plt.imshow(resampledData, vmin=mean - 3 * std, vmax=mean + 100 * std, origin="lower")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Images are stil remaining")
    else:
        print("Done.")