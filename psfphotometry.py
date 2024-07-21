import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from astroquery.vizier import Vizier
import subprocess
import os
from glob import glob

def test_dependency(dep, alternate_name=None):
    """
    Test external dependency by trying to run it as a subprocess
    """
    try:
        subprocess.check_output(dep, stderr=subprocess.PIPE, shell=True)
        print("%s is installed properly as %s. OK" % (dep, dep))
        return 1
    except subprocess.CalledProcessError:
        try:
            subprocess.check_output(alternate_name, stderr=subprocess.PIPE, shell=True)
            print("%s is installed properly as %s. OK" % (dep, alternate_name))
            return 1
        except subprocess.CalledProcessError:
            print("===%s/%s IS NOT YET INSTALLED PROPERLY===" % (dep, alternate_name))
            return 0
    
def run_command(command):
    """
    Run a system command and handle errors
    """
    try:
        print(f'Executing command: {command}')
        subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as err:
        print(f'Could not run command with exit error {err}')

def run_source_extractor(imageName, configFile, catalogName, paramName):
    """
    Run Source Extractor
    """
    command = f'sex -c {configFile} {imageName} -CATALOG_NAME {catalogName} -PARAMETERS_NAME {paramName}'
    run_command(command)

def run_psfex(catalogName, psfConfigFile):
    """
    Run PSFEx
    """
    command = f'psfex -c {psfConfigFile} {catalogName}'
    run_command(command)

def run_psf_source_extractor(imageName, configFile, psfName, psfcatalogName, psfparamName):
    """
    Run Source Extractor with PSF model
    """
    command = f'sex -c {configFile} {imageName} -CATALOG_NAME {psfcatalogName} -PSF_NAME {psfName} -PARAMETERS_NAME {psfparamName}'
    run_command(command)

def process_fits_files(directory, ra, dec):
    os.chdir(directory)
    fits_files = glob('*.wcs.proc.fits')
    psfmags = []
    errpsfmags = []
    errfiles = []

    for imageName in fits_files:
        try:
            with fits.open(imageName) as HDUList:
                header = HDUList[0].header
                image = HDUList[0].data

            zscale = ZScaleInterval().get_limits(image)
            w = WCS(header)
            (raImage, decImage) = w.all_pix2world(image.shape[0] / 2, image.shape[1] / 2, 1)
            boxsize = 30  # arcminutes
            maxmag = 18
            catNum = 'II/349'

            try:
                v = Vizier(columns=['*'], column_filters={"gmag": f"<{maxmag}", "Nd": ">6", "e_gmag": f"<{1.086/3}"}, row_limit=-1)
                Q = v.query_region(SkyCoord(ra=raImage, dec=decImage, unit=(u.deg, u.deg)), radius=str(boxsize) + 'm', catalog=catNum, cache=False)
            except Exception as e:
                print(f'Could not reach the Vizier database. Is the internet working? {e}')
                continue

            Q[0].meta['desc'] = Q[0].meta.pop('description')
            Q[0].write(f'{directory}/{imageName}ps1Catalog.fits', format='fits', overwrite=True)

            ps1_imCoords = w.all_world2pix(Q[0]['RAJ2000'], Q[0]['DEJ2000'], 1)
            good_cat_stars = Q[0][np.where((ps1_imCoords[0] > 500) & (ps1_imCoords[0] < 3500) & (ps1_imCoords[1] > 500) & (ps1_imCoords[1] < 3500))]
            ps1_imCoords = w.all_world2pix(good_cat_stars['RAJ2000'], good_cat_stars['DEJ2000'], 1)

            configFile = 'photomCat.sex'
            catalogName = imageName + '.cat'
            paramName = 'photomCat.param'
            run_source_extractor(imageName, configFile, catalogName, paramName)

            with fits.open(catalogName) as HDU:
                sourceTable = Table(HDU[2].data)

            cleanSources = sourceTable[(sourceTable['FLAGS'] == 0) & (sourceTable['FWHM_WORLD'] < 2) & (sourceTable['XWIN_IMAGE'] < 3500) & (sourceTable['XWIN_IMAGE'] > 500) & (sourceTable['YWIN_IMAGE'] < 3500) & (sourceTable['YWIN_IMAGE'] > 500)]

            psfConfigFile = 'psfex_conf.psfex'
            run_psfex(catalogName, psfConfigFile)

            psfModelHDU = fits.open('moffat_' + imageName + '.fits')[0]
            psfModelData = psfModelHDU.data
            mean, median, std = sigma_clipped_stats(psfModelData)

            psfName = imageName + '.psf'
            psfcatalogName = imageName + '.psf.cat'
            psfparamName = 'photomPSF.param'
            run_psf_source_extractor(imageName, configFile, psfName, psfcatalogName, psfparamName)

            with fits.open(psfcatalogName) as HDU:
                psfsourceTable = Table(HDU[2].data)

            cleanPSFSources = psfsourceTable[(psfsourceTable['FLAGS'] == 0) & (psfsourceTable['FLAGS_MODEL'] == 0) & (psfsourceTable['FWHM_WORLD'] < 2) & (psfsourceTable['XMODEL_IMAGE'] < 3500) & (psfsourceTable['XMODEL_IMAGE'] > 500) & (psfsourceTable['YMODEL_IMAGE'] < 3500) & (psfsourceTable['YMODEL_IMAGE'] > 500)]

            psfsourceCatCoords = SkyCoord(ra=cleanPSFSources['ALPHAWIN_J2000'], dec=cleanPSFSources['DELTAWIN_J2000'], frame='icrs', unit='degree')
            ps1CatCoords = SkyCoord(ra=good_cat_stars['RAJ2000'], dec=good_cat_stars['DEJ2000'], frame='icrs', unit='degree')
            photoDistThresh = 0.6
            idx_psfimage, idx_psfps1, d2d, d3d = ps1CatCoords.search_around_sky(psfsourceCatCoords, photoDistThresh*u.arcsec)

            print(f'Found {len(idx_psfimage)} good cross-matches')

            psfoffsets = ma.array(good_cat_stars['gmag'][idx_psfps1] - cleanPSFSources['MAG_POINTSOURCE'][idx_psfimage])
            zero_psfmean, zero_psfmed, zero_psfstd = sigma_clipped_stats(psfoffsets)

            sn2023ixf_coords = SkyCoord(ra=[ra], dec=[dec], frame='icrs', unit='degree')
            idx_sn2023ixf, idx_cleanpsf_sn2023ixf, d2d, d3d = psfsourceCatCoords.search_around_sky(sn2023ixf_coords, photoDistThresh*u.arcsec)
            print(f'Found the source at index {idx_cleanpsf_sn2023ixf[0]}')

            sn2023ixf_psfinstmag = cleanPSFSources[idx_cleanpsf_sn2023ixf]['MAG_POINTSOURCE'][0]
            sn2023ixf_psfinstmagerr = cleanPSFSources[idx_cleanpsf_sn2023ixf]['MAGERR_POINTSOURCE'][0]

            sn2023ixf_psfmag = zero_psfmed + sn2023ixf_psfinstmag
            psfmags.append(sn2023ixf_psfmag)
            sn2023ixf_psfmagerr = np.sqrt(sn2023ixf_psfinstmagerr**2 + zero_psfstd**2)
            errpsfmags.append(sn2023ixf_psfmagerr)
            print(f'PSF-fit magnitude of SN2023ixf is {sn2023ixf_psfmag} +/- {sn2023ixf_psfmagerr}')

        except Exception as e:
            print(f'Error processing {imageName}: {e}')
            errfiles.append(imageName)
            continue
    return psfmags, errpsfmags, errfiles

def main():
    # Prompt user for directory and source location
    directory = input("Enter the directory containing the .wcs.proc.fits files: ")
    ra = float(input("Enter the RA of the source (in degrees): ")) # (For SN2023xif)210.910674637
    dec = float(input("Enter the Dec of the source (in degrees): "))# 54.3116510708
    band = input("Enter the filter band: ")
    # Test dependencies
    dependencies = [('source-extractor', 'sex'), ('psfex', 'PSFEx')]
    i = 0
    for dep_name1, dep_name2 in dependencies:
        i += test_dependency(dep_name1, dep_name2)
    print("%i out of %i external dependencies installed properly.\n" % (i, len(dependencies)))
    if i != len(dependencies):
        print("Please correctly install these programs before continuing by following the instructions in README.md.")
        return

    # Process FITS files
    psfmags, errpsfmags = process_fits_files(directory, ra, dec)

    # Output results
    print("PSF-fit magnitudes:", psfmags)
    print("Errors in PSF-fit magnitudes:", errpsfmags)
    print(f'Could not process {errfiles} files')

    # Plotting
    if psfmags and errpsfmags:  # Ensure there is data to plot
        plt.figure(figsize=(15, 15))
        plt.errorbar(range(len(psfmags)), psfmags, yerr=errpsfmags, fmt='o', color='blue', ecolor='red', elinewidth=2, capsize=5, capthick=2, label='PSF Magnitudes with Error Bars')
        plt.xlabel('Observation Index', fontsize=14)
        plt.ylabel('PSF Magnitude', fontsize=14)
        plt.title(f'PSF Magnitudes for {band}band', fontsize=16)
        plt.legend()
        plt.grid(True)
        plot_filename = 'psf_magnitudes_plot.png'
        plt.savefig(plot_filename, format='png')
        plt.show()
    else:
        print("No data available for plotting.")

if __name__ == "__main__":
    main()
