### Importing Packages

import numpy as np
import os
import glob
from astropy.io import fits
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from astropy.nddata import Cutout2D
from astroquery.vizier import Vizier
import subprocess
import sys
import warnings
from image_registration import chi2_shift
from image_registration.fft_tools import shift2d
from photutils import Background2D, MedianBackground

warnings.filterwarnings("ignore")

def test_dependency(dep,alternate_name=None):

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


# To be used later
# dependencies = [('source-extractor', 'sex'), ('psfex', 'PSFEx')]
# i = 0
# for dep_name1, dep_name2 in dependencies:
#     i += test_dependency(dep_name1, dep_name2)
# print("%i out of %i external dependencies installed properly.\n" % (i, len(dependencies)))
# if i != len(dependencies):
#     print("Please correctly install these programs before continuing by following the instructions in README.md.")
# else:
#     print("You are ready to continue.")

def get_image_data(image_path, extension=0):

    """
    Get image data from a fits file
    """
    with fits.open(image_path) as hdul:
        header = hdul[extension].header
        data = hdul[extension].data
    return data , header

## Plotting the image

def plot_image(data, figsize=(10,10), origin='lower', save=False, save_path='.',cmap='gray',patch = False):

    """
    Plot ds9 type image
    """

    if patch:
        #fig = plt.figure(figsize=figsize)
        zscale = ZScaleInterval().get_limits(data)
        #ax = fig.gca()
        #plt.imshow(data, origin = origin,vmax=zscale[1],vmin=zscale[0],cmap=cmap)
        #circles = [plt.Circle((ps1_imCoords[0][i], ps1_imCoords[1][i]), radius = 10, edgecolor='C1', facecolor='None') for i in range(len(ps1_imCoords[0]))]
        #for c in circles:
            #ax.add_artist(c)
        if save:
            plt.savefig(save_path+'patch.png')

        #plt.show()

    else:
            #plt.figure(figsize=figsize)
            #zscale = ZScaleInterval().get_limits(data)
            #plt.imshow(data, origin = origin,vmax=zscale[1],vmin=zscale[0],cmap=cmap)
            #plt.colorbar()
            if save:
                plt.savefig(save_path+'image.png')
            #plt.show()

### Vizier Query

def vizier_query(data ,header, catalog = 'II/349',boxsize = 30,mag = 18,file_name = 'ps1Catalog.fits'):

    """
    Query Vizier for a catalog
    """

    w = WCS(header)
    (raImage, decImage) = w.all_pix2world(data.shape[0]/2, data.shape[1]/2, 1)

    boxsize = boxsize
    catNum = catalog
    print(f'\nQuerying Vizier {catNum} around RA {raImage:.4f}, Dec {decImage:.4f} with a radius of {boxsize} arcmin')

    maxmag = mag

    try:
        # You can set the filters for the individual columns (magnitude range, number of detections) inside the Vizier query
        v = Vizier(columns=['*'], column_filters={"gmag":f"<{maxmag}", "Nd":">6", "e_gmag":f"<{1.086/3}"}, row_limit=-1)
        Q = v.query_region(SkyCoord(ra=raImage, dec=decImage, unit=(u.deg, u.deg)), radius=str(boxsize)+'m', catalog=catNum, cache=False)
        print(Q[0])
    except:
        print('I cannnot reach the Vizier database. Is the internet working?')

    Q[0].meta['desc'] = Q[0].meta.pop('description')
    Q[0].write(file_name, format='fits', overwrite=True)
    return Q[0]


def run_sextractor(image_path,config = 'photomCat.sex', image_name = 'AbC'):

    """
    Run SExtractor on an image
    """
    if image_name != None :
        image_name = image_name
        configFile = config
        catalogName = image_name + '.cat'
        paramName = 'photomCat.param'
        try:
               command = f'sex -c {configFile} {image_path} -CATALOG_NAME {catalogName} -PARAMETERS_NAME {paramName}'
               print(f'Executing command: {command}')
               rval = subprocess.run(command.split(), check=True)

               with fits.open(catalogName) as HDU:
                print(HDU.info())
                sourceTable = Table(HDU[2].data)

                print(sourceTable.colnames)
                print(sourceTable)

                cleanSources = sourceTable[(sourceTable['FLAGS']==0) & (sourceTable['FWHM_WORLD'] < 2) & (sourceTable['XWIN_IMAGE']<3500) & (sourceTable['XWIN_IMAGE']>500) & (sourceTable['YWIN_IMAGE']<3500) & (sourceTable['YWIN_IMAGE']>500)]
        except subprocess.CalledProcessError as err:
               print(f'Could not run sextractor with exit error {err}')
        return cleanSources , catalogName
    else:
        print('Please provide the image name')
        return None


def run_psfex(image_name,config = 'psfex_conf.psfex'):

    psfConfigFile = config

    try:
        command = f'psfex -c {psfConfigFile} {image_name}' + '.cat'
        print(f'Executing command: {command}')
        rval = subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as err:
        print(f'Could not run psfex with exit error {err}')

    psfModelHDU = fits.open('moffat_' + image_name + '.fits')[0]
    psfModelData = psfModelHDU.data

    return psfModelData



def plot_psf(image_name,save=False,save_path='.'):

    psfModelHDU = fits.open('moffat_' + image_name + '.fits')[0]
    psfModelData = psfModelHDU.data
    mean, median, std = sigma_clipped_stats(psfModelData)

    plt.figure(figsize=(6,6))
    plt.imshow(psfModelData, vmin=0, vmax=median+20*std, origin='lower')
    #plt.show()
    if save:
        plt.savefig(save_path+'psf.png')

    return psfModelData

def sextractor_photometry(image_name='abc',psf_param = 'photomPSF.param',config = 'photomCat.sex'):

    psfName = image_name + '.psf'
    psfcatalogName = image_name+'.psf.cat'
    psfparamName = psf_param # This is a new set of parameters to be obtained from SExtractor, including PSF-fit magnitudes
    try:
        # We are supplying SExtactor with the PSF model with the PSF_NAME option
        command = f'sex -c {config} {image_name} -CATALOG_NAME {psfcatalogName} -PSF_NAME {psfName} -PARAMETERS_NAME {psfparamName}'
        print(f"Executing command: {command}")
        rval = subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as err:
        print(f'Could not run sextractor with exit error {err}')

    with fits.open(psfcatalogName) as HDU:
        psfsourceTable = Table(HDU[2].data)

    print(psfsourceTable.colnames)
    print(psfsourceTable)

    cleanPSFSources = psfsourceTable[(psfsourceTable['FLAGS']==0) & (psfsourceTable['FLAGS_MODEL']==0)  & (psfsourceTable['FWHM_WORLD'] < 2) & (psfsourceTable['XMODEL_IMAGE']<3500) & (psfsourceTable['XMODEL_IMAGE']>500) &(psfsourceTable['YMODEL_IMAGE']<3500) &(psfsourceTable['YMODEL_IMAGE']>500)]

    return cleanPSFSources , psfcatalogName

# psfsourceCatCoords = SkyCoord(ra=cleanPSFSources['ALPHAWIN_J2000'], dec=cleanPSFSources['DELTAWIN_J2000'], frame='icrs', unit='degree')
# ps1CatCoords = SkyCoord(ra=good_cat_stars['RAJ2000'], dec=good_cat_stars['DEJ2000'], frame='icrs', unit='degree')
# photoDistThresh = 0.6
# idx_psfimage, idx_psfps1, d2d, d3d = ps1CatCoords.search_around_sky(psfsourceCatCoords, photoDistThresh*u.arcsec)

def psf_fit_plot(a,b,save = False,save_path = None):

    plt.figure(figsize=(8,8))
    plt.scatter(a , b, color='C0', s=10)
    plt.xlabel('PS1 magnitude', fontsize=15)
    plt.ylabel('Instrumental PSF-fit magnitude', fontsize=15)
    if save:
        plt.savefig(save_path+'psf_fit_plot.png')
   # plt.show()

def psf_offset(a,b):

    psfoffsets = np.array(a - b)
    zero_psfmean, zero_psfmed, zero_psfstd = sigma_clipped_stats(psfoffsets,sigma=2.0)
    print('PSF Mean ZP: %.2f\nPSF Median ZP: %.2f\nPSF STD ZP: %.2f'%(zero_psfmean, zero_psfmed, zero_psfstd))

    return zero_psfmean, zero_psfmed, zero_psfstd

def psf_fit(backsub_cutout, psf_data, cam):
  backsub_cutout = backsub_cutout[10:50, 10:50]
  psf_data = psf_data[10:50, 10:50]
  # backsub_cutout = backsub_cutout[20:40, 20:40]
  # psf_data = psf_data[20:40, 20:40]
  xoff, yoff, exoff, eyoff = chi2_shift(
    psf_data, backsub_cutout, 10, boundary="wrap", nfitted=2, upsample_factor="auto"
  )
  print(
    f"peak of source detected at offset {xoff, yoff} (in pixel) from target position"
  )
  if cam == "andor":
    if abs(xoff) > 3.0 or abs(yoff) > 3.0:
      xoff = 0.0
      yoff = 0.0
      phot_type = "Forced_PSF"
    else:
      phot_type = "PSF"
  if cam == "sbig":
    if abs(xoff) > 17.0 or abs(yoff) > 17.0:
      xoff = 0.0
      yoff = 0.0
      phot_type = "Forced_PSF"
    else:
      phot_type = "PSF"
  resize_backsub_cutout = shift2d(
    backsub_cutout, -xoff, -yoff
  )
  flux = np.sum(resize_backsub_cutout * psf_data) / np.sum(psf_data * psf_data)
  # convolved = convolve(resize_backsub_cutout, psf_data, mode='constant')
  # flux = np.sum(convolved)
  return flux, phot_type, xoff, yoff

def cal_inst_mag(psf_data, image_path, ra, dec, cam):
  image_data, image_header = get_image_data(image_path)
  wcs = WCS(image_header)
  [x_img, y_img] = wcs.all_world2pix(ra, dec, 1)
  x = int(x_img)
  y = int(y_img)
  cutout = Cutout2D(image_data, position=(x, y), size=np.shape(psf_data), wcs=wcs)
  Is = cutout.data
  #is_saturated = np.ravel(Is > saturLevel).any()
#   sigma = sigma_clip(sigma=3.0)
  try:
    bkg_estimator = MedianBackground()
    bkg = Background2D(
      Is,
      np.shape(psf_data),
      filter_size=(3, 3),
      sigma_clip=sigma_clip(sigma=3.0),
      bkg_estimator=bkg_estimator,
    )
    background = bkg.background
  except:
    _, background, _ = sigma_clipped_stats(Is)
  sigma_flux = np.sqrt(np.sum(Is * psf_data**2)) / np.sum(psf_data * psf_data)
  Is = Is - background
  flux, PSF_type, xoff, yoff = psf_fit(Is, psf_data, cam)
  source_x = x + xoff
  source_y = y + yoff
  [source_ra, source_dec] = wcs.all_pix2world(source_x, source_y, 1)
  print(f"psf_fit_coord : {source_ra} {source_dec} ")
  int_mag = -2.5 * np.log10(flux)
  int_mag_err = 2.5 * np.log10(1 + sigma_flux / flux)
  return (
    flux,
    sigma_flux,
    int_mag,
    int_mag_err,
    PSF_type,
    source_ra,
    source_dec,
  )

def lim_mag(image,ra,dec,psf_zp,fwhm,pix_scale):
    fwhm = fwhm/pix_scale
    image_data,image_header = get_image_data(image)
    wcs = WCS(image_header)
    x,y = wcs.world_to_pixel(SkyCoord(ra = ra*u.deg,dec = dec*u.deg))
    position = [x,y]
    cutout = Cutout2D(image_data,position,size = 300,wcs=wcs)
    _,_,std = sigma_clipped_stats(cutout.data)
    flux_5 = 5*std*np.sqrt(np.pi*float(fwhm)**2)
    mag_5 = -2.5*np.log10(flux_5)+psf_zp

    return mag_5


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory_with_fits_files> <output_directory>")
        sys.exit(1)

    fits_directory = sys.argv[1]
    output_directory = sys.argv[2]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fits_files = glob.glob(os.path.join(fits_directory, '*.wcs.proc.fits'))
    if not fits_files:
        print("No .wcs.proc.fits files found in the specified directory.")
        sys.exit(1)

    for fits_file in fits_files:
        image_path = fits_file
        image_name = os.path.basename(fits_file)
        save = os.path.join(output_directory, 'plots')
        path = output_directory

        if not os.path.exists(save):
            os.makedirs(save)

        test_dependency(dep=['sex', 'psfex'])
        data, header = get_image_data(image_path)
        plot_image(data, save=save)
        table = vizier_query(data, header)
        ps1_imCoords = WCS(header).all_world2pix(table['RAJ2000'], table['DEJ2000'], 1)
        good_cat_stars = table[np.where((ps1_imCoords[0] > 500) & (ps1_imCoords[0] < 3500) & (ps1_imCoords[1] > 500) & (ps1_imCoords[1] < 3500))]
        ps1_imCoords = WCS(header).all_world2pix(good_cat_stars['RAJ2000'], good_cat_stars['DEJ2000'], 1)

        plot_image(data, patch=True, save=save, save_path=path)

        cleanSources, catalogName = run_sextractor(image_path, image_name=image_name)
        psf_data = run_psfex(image_name=image_name)
        plot_psf(image_name=image_name, save=save, save_path=path)
        cleanPSFSources, psfcatalogName = sextractor_photometry(image_name=image_name)

        psfsourceCatCoords = SkyCoord(ra=cleanPSFSources['ALPHAWIN_J2000'], dec=cleanPSFSources['DELTAWIN_J2000'], frame='icrs', unit='degree')
        ps1CatCoords = SkyCoord(ra=good_cat_stars['RAJ2000'], dec=good_cat_stars['DEJ2000'], frame='icrs', unit='degree')
        photoDistThresh = 0.6

        idx_psfimage, idx_psfps1, d2d, d3d = ps1CatCoords.search_around_sky(psfsourceCatCoords, photoDistThresh*u.arcsec)
        print(f'Found {len(idx_psfimage)} good cross-matches')
        psf_fit_plot(good_cat_stars['gmag'][idx_psfps1], cleanPSFSources['MAG_POINTSOURCE'][idx_psfimage], save=save, save_path=path)
        zero_psfmean, zero_psfmed, zero_psfstd = psf_offset(good_cat_stars['gmag'][idx_psfps1], cleanPSFSources['MAG_POINTSOURCE'][idx_psfimage])

        ps1CatCoords.search_around_sky(psfsourceCatCoords, photoDistThresh*u.arcsec)

        ra = 210.910674637
        dec = 54.3116510708

        # Mean FWHM and limiting magnitude calculation
        fwhm = np.median(cleanPSFSources['FWHM_WORLD'][idx_psfimage]) * 3600
        lim_mag = lim_mag(image_name, ra, dec, zero_psfmed, fwhm, 0.7)
        print(f'Limiting magnitude is {lim_mag}')

        sn2023ixf_coords = SkyCoord(ra=[ra], dec=[dec], frame='icrs', unit='degree')
        idx_sn2023ixf, idx_cleanpsf_sn2023ixf, d2d, d3d = psfsourceCatCoords.search_around_sky(sn2023ixf_coords, photoDistThresh*u.arcsec)
        print(f'Found the source at index {idx_cleanpsf_sn2023ixf[0]}')

        flux, sigma_flux, sn2023ixf_psfinstmag, sn2023ixf_psfinstmagerr, PSF_type, source_ra, source_dec = cal_inst_mag(psf_data=psf_data, image_path=image_path, ra=ra, dec=dec, cam='andor')

        sn2023ixf_psfmag = zero_psfmed + sn2023ixf_psfinstmag
        sn2023ixf_psfmagerr = np.sqrt(sn2023ixf_psfinstmagerr**2 + zero_psfstd**2)

        print(f'PSF-fit magnitude of SN2023ixf is {sn2023ixf_psfmag} +/- {sn2023ixf_psfmagerr}')

        # Writing the output to a file
        with open(os.path.join(output_directory, 'output-mag-r.txt'), 'a') as f:
            f.write(f'{image_name[0:11]}\t \t \t {sn2023ixf_psfmag} \t \t \t {sn2023ixf_psfmagerr} \t \t \t {lim_mag} \n')

        # Clean up the extra files
        for temp_file in glob.glob(f'{image_name}*.tmp'):
            os.remove(temp_file)
        for temp_file in glob.glob(f'{image_name}*.fits'):
            os.remove(temp_file)

if __name__ == '__main__':
    main()
