# Polarimetric analysis (polana)
[developer mail](mailto:jinbeniyama@gmail.com)

The icon of the repository is DIPOL-2 from Piirola et al. (2014).
This code is used in [Beniyama et al. (2023b)](https://iopscience.iop.org/article/10.3847/1538-4357/ace88f).

## Overview
Analyze polarimetric data obtained with 

- Multi-Spectral Imager (MSI, no paper has been published about polarimetry)
- Wide Field Grism Spectrograph 2 (WFGS2, [Kawakami et al. 2021](https://doi.org/10.32231/starsandgalaxies.4.0_5))
- Hiroshima Optical and Near-InfraRed Camera (HONIR, [Akitaya et al. 2014](https://ui.adsabs.harvard.edu/abs/2014SPIE.9147E..4OA/abstract)) 
- Double Image High Precision Polarimeter (DIPOL-2, [Piirola et al. 2014](https://ui.adsabs.harvard.edu/abs/2014SPIE.9147E..8IP/abstract)).

## Installing
Please install by pip (sorry in prep.), otherwise open paths to src and `polana` directories by yourself.

## Usage
Before polarimetry, we should check the source location in pixel coordinates
by eye using fits viewer such as ds9 and save it as `input.txt`.
Polarimetric parameters, q, u, P, theta, and their uncertainties, are saved in the output file.

### 1. MSI 
The 1 set consists of 4 images obtained with angles of half-wave plates at 0, 45, 22.5, and 67.5.
The format of `input.txt` is as follows:
(Do not think! Feel the meaning of each column.)
```
xo yo xe ye fits
275 187 240 70 msi221221_805294.fits
275 184 240 66 msi221221_805295.fits
277 187 242 68 msi221221_805296.fits
276 186 242 65 msi221221_805297.fits
```

```
[usage]
pol_MSI.py (object name) (input file)  --radius (circular aperture radius in pixex) --width (width of baricentric search)

[example]
pol_MSI.py "HD19820" input.txt  --radius 20 --width 60
```


### 2. WFGS2
Essentially the same with MSI.
But the 1 set consists of 8 images since ordinary and extraordinary images are
separated into different fits files.
The format of `input.txt` is as follows:
(Do not think! Feel the meaning of each column.)
```
x y fits
477 1475 wfgs2_221220_0013.HD19820.Rc.e.cr.fits
480 1485 wfgs2_221220_0013.HD19820.Rc.o.cr.fits
484 1477 wfgs2_221220_0014.HD19820.Rc.e.cr.fits
484 1486 wfgs2_221220_0014.HD19820.Rc.o.cr.fits
484 1477 wfgs2_221220_0015.HD19820.Rc.e.cr.fits
487 1487 wfgs2_221220_0015.HD19820.Rc.o.cr.fits
481 1478 wfgs2_221220_0016.HD19820.Rc.e.cr.fits
485 1486 wfgs2_221220_0016.HD19820.Rc.o.cr.fits
```

### 3. HONIR
Before polarimetry, we should check the source location in pixel coordinates
by eye using fits viewer such as ds9 and save it as `input.txt`.
Polarimetric parameters, q, u, P, theta, and their uncertainties, are saved in the output file.

The 1 set consists of 4 images obtained with angles of half-wave plates at 0, 45, 22.5, and 67.5.
The format of `input.txt` is as follows:
(Do not think! Feel the meaning of each column.)
```
xo yo xe ye fits
88 267 248 271 HN0322604opt00_bt_bs_fl_clip.fits
86 269 248 271 HN0322605opt00_bt_bs_fl_clip.fits
89 269 249 273 HN0322606opt00_bt_bs_fl_clip.fits
88 271 248 273 HN0322607opt00_bt_bs_fl_clip.fits
```

```
[usage]
pol_HONIR.py (object name) (input file)  --radius (circular aperture radius in pixex) --width (width of baricentric search)

[example]
pol_HONIR.py "HD19820" input.txt  --radius 20 --width 60
```


### 4. DIPOL-2
In prep.

## Acknowledgments
I would like to express the gratitude to the people involved in 
[OISTER](https://oister.kwasan.kyoto-u.ac.jp/) and T60/DIPOL-2.

## Dependencies
This library is depending on `NumPy`, `SciPy`, `SEP`, and `Astropy`.
Scripts are developed on `Python 3.7.10`, `NumPy 1.19.2`, `SciPy 1.6.1`, 
`SEP 1.0.3`, and `Astropy 4.2`.
