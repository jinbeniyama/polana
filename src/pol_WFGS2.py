#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with WFGS2.
The format of input file is as follows:
x y fits
468 1462 wfgs2_221220_0001.2010XC15.Rc.e.cr.fits
.

Ordainary and Extra-ordinary can be distinguished from fits file.
"""
import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
from scipy.spatial import KDTree
from datetime import datetime
import sep
import astropy.io.fits as fits

from dipolana.util import *
from myplot import mycolor, myls



if __name__ == "__main__":
    parser = ap(
        description="Do photometry for images obtained with WFGS2.")
    parser.add_argument(
        "obj", type=str, 
        help="Object name")
    parser.add_argument(
        "loc", type=str, 
        help="Observation location (MPC code)")
    parser.add_argument(
        "inp", type=str, nargs="*",
        help="Input file with certain format")
    parser.add_argument(
        "--fitsdir", type=str, default=".",
        help="Fits directory")
    parser.add_argument(
        "--radius", type=float, default=40, 
        help="aperture radius in pixel")
    parser.add_argument(
        "--ann_gap", type=float, default=2, 
        help="gap between annulus and circle ")
    parser.add_argument(
        "--ann_width", type=float, default=3, 
        help="width of annulus")
    parser.add_argument(
        "--band", type=str, default="R", 
        help="Filter (to set gain)")
    parser.add_argument(
        "--outdir", type=str, default=".",
        help="Output directory")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output filename")
    parser.add_argument(
        "--wobgsub", action='store_false', default=True,
        help='Do not subtract background')
    parser.add_argument(
        "-p", "--photmap", action='store_true',
        help='create photometry region map (a bit slow)')
    parser.add_argument(
        "--ann", action='store_true', default=False,
        help='Do photometry with annulus')
    args = parser.parse_args()
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ## For photometry region png in output directory
    if args.photmap:
        photmapdir = os.path.join(outdir, "photregion")
        os.makedirs(photmapdir, exist_ok=True)

    fitsdir = args.fitsdir
    radius = args.radius
    band = args.band
    print(f"  Aperture radius {radius} pix")
    print(f"  filter {band}-band")

    key_texp = "EXPTIME"
    # Only date
    key_date = "DATE-OBS"
    # Central UTC
    key_ut = "UT-CEN"
    # 0, 45, 22.5, 67.5
    key_ang = "HWP-AGL"
    # Inverse gain e/ADU
    key_gain = "GAIN"
    
    
    alpha_list, P_list, Perr_list = [], [], []
    for x in args.inp:
        # Read input files
        df_in = pd.read_csv(x, sep=" ")
        N_fits = len(df_in)
        N_fits_per_set = 8
        N_set = int(N_fits/N_fits_per_set)
         
        for idx_set in range(N_set):
            print("")
            print(f"Start analysis of {idx_set+1}/{N_set}-th set")
            # List to save results 
            df_res_list = []
            for idx_fi in range(N_fits_per_set):
                fi = df_in.at[idx_set*N_fits_per_set+idx_fi, "fits"]
                fi_path = os.path.join(fitsdir, fi)
                src = fits.open(fi_path)[0]
                hdr = src.header 

                # Judge Ordinary or Extra-ordinary from fitsname
                # At least applicable 2010 XC15 data on December 2022
                # like 'wfgs2_221220_0001.2010XC15.Rc.e.cr.fits'
                o_or_e = fi.split(".")[-3]

                print(f"    Start analysis of {idx_fi+1}/{N_fits_per_set}-th fits")
                print(f"      {fi} {o_or_e}")

                # Set inverse gain
                # Read GAIN in fits header
                # ex) GAIN    =                 2.28 / CCD gain in e/ADU, ref: MINT wiki
                gain = hdr[key_gain]
                
                # Read 2-d image
                img = src.data
                ny, nx = img.shape[0], img.shape[1]
                print(f"    Data dimension nx, ny = {nx}, {ny}")
                # Exposure time
                t_exp = src.header[key_texp]
                print(f"    Exposure time {t_exp} s")

                # Save photometry info.
                # Assume e -> o -> e -> o ...
                if o_or_e == "e":
                    info = dict()
                else:
                    pass


               
                # Background subtraction ==============================================
                # Convert to float to suppress error
                img = img.astype(np.float32)
                # args.wobgsub useless?
                if (not args.ann) and (args.wobgsub):
                    img, bg_info = remove_background2d_Dipol2(img)
                    bgerr = np.round(bg_info["rms"], 2)
                    #print("")
                    #print(f"Subtracted Mean: {np.mean(img)}")
                    #print(f"Subtracted Median: {np.median(img)}")
                    #print(f"Estimated sky error per pixel is {bgerr} [ADU]")
                    #print("")
                else:
                    print("    !! Do not subtract background !!")
                    # Temporally
                    bgerr = 0
                info["gloabalrms"] = bgerr
                info["level_mean"] = np.mean(img)
                info["level_median"] = np.median(img)
                # Background subtraction ==============================================


                # Source detection ====================================================
                # 5-sigma detection
                #dth     = 5
                #minarea = 15
                #objects = extract(img, dth, minarea, bgerr, mask=None)
                #N_obj   = len(objects)
                #print(f"{N_obj} objects detected on {idx_fi+1}-th fits {fi}.")
                
                # Original coordinates
                x1 = df_in.at[idx_set*N_fits_per_set+idx_fi, "x"]
                y1 = df_in.at[idx_set*N_fits_per_set+idx_fi, "y"]

                # # Search the most suitable objects
                # x_base, y_base = objects["x"], objects["y"]
                # tree_base = KDTree(list(zip(x_base, y_base)), leafsize=10)
                # # Ordinary
                # res = tree_base.query_ball_point((x0, y0), radius)

                # # New barycenters
                # x1, y1 = res
                # Source detection ====================================================


                # Do photometry =======================================================
                if args.ann:
                    ann_gap = args.ann_gap
                    ann_width = args.ann_width
                    bkgann  = (radius + ann_gap, radius + ann_gap + ann_width)
                else:
                    bkgann = None
                # In ADU, 
                # fluxerr**2 = bgerr_per_pix**2*N_pix + Poission**2
                #            = bgerr_per_pix**2*N_pix + (flux*gain)/gain**2
                flux, fluxerr, eflag = sep.sum_circle(
                    img, [x1], [y1], r=radius, err=bgerr, gain=gain,
                    bkgann=bkgann)
                flux, fluxerr = float(flux), float(fluxerr)
                # Do photometry =====================================================


                # Plot photometry region ======================================
                if args.photmap:
                    out = os.path.join(
                        photmapdir, f"{fi}_photmap.png")

                    label = f"{args.obj} (x, y)=({x1}, {y1})"
                    color = mycolor[0]
                    ls = myls[0]

                    # Plot src image after 5-sigma clipping 
                    sigma = 5
                    _, vmin, vmax = sigmaclip(img, sigma, sigma)
                    ny, nx = img.shape

                    fig = plt.figure(figsize=(12,int(12*ny/nx)))
                    ax = fig.add_subplot(111)
                    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                    ax.scatter(
                        x1, y1, color=color, s=radius, lw=1, 
                        facecolor="None", alpha=1, label=label)
                    ax.add_collection(PatchCollection(
                        [Circle((x1,y1), radius)],
                        color=color, ls=ls, lw=1, facecolor="None", label=None)
                        )
                    ax.set_xlim([0, nx])
                    ax.set_ylim([0, ny])
                    ax.legend().get_frame().set_alpha(1.0)
                    ax.invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(out, dpi=200)
                    plt.close()
                # Plot photometry region ======================================


                # Noise calculation ===================================================
                # if idx_fi == 0:
                #     ## Background noise (sum of background + readout)
                #     med, std = np.median(img), np.std(img)
                #     #print(f"median, std = {med:.3f}, {std:.3f}")
                #     #print(f"N_original = {nx*ny}")
                #     # Clip
                #     sigma = 3
                #     for i in range(5):
                #         img = img[(img < sigma*std) & (img > -sigma*std)]
                #         std = np.std(img)
                #         #print(f"N_clipped = {len(img)}")
                #         #print(f"median, std = {med:.3f}, {std:.3f} (after {i}-th {sigma}-sigma clipping)")
                #     print(f"Background noise = {std:.3f} ADU/pix")
                #     print(f", which should be close to bgrms {bgerr:.3f} ADU/pix (returns of sep)\n")
                #     ## Poission noise 
                #     # N^2_e = flux_e = flux_ADU*gain(e/ADU)
                #     # N_ADU = N_e/gain
                #     # -> N_ADU = sqrt(flux_ADU/gain)
                #     fluxerr_o_ADU = np.sqrt(flux_o/gain)
                #     fluxerr_e_ADU = np.sqrt(flux_e/gain)
                #     print(f"Poisson noise (o, e) = {fluxerr_o_ADU:.2f} {fluxerr_e_ADU:.2f} ADU")
                #     ## Total noise
                #     # Circular aperture area in pix
                #     N_app = np.pi*radius**2
                #     # Sum of background noise
                #     std_sum = np.sqrt(N_app)*std
                #     sumerr_o = adderr(std_sum, fluxerr_o_ADU)
                #     sumerr_e = adderr(std_sum, fluxerr_e_ADU)
                #     print(f"Sum of noise (o, e) = {sumerr_o:.2f} {sumerr_e:.2f} ADU")
                #     print(f", which should be close to fluxerr {fluxerr_o:.3f} and {fluxerr_e:.3f} ADU (returns of sep)\n")
                # # Noise calculation ===================================================

                info[f"flux_{o_or_e}"]    = flux
                info[f"fluxerr_{o_or_e}"] = fluxerr

                # Assume e (rot X)-> o (rot X)-> e (rot Y)-> o (rot Y) ...
                if o_or_e == "e":
                    # Rotation angle 0, 22.5 deg (22.5 deg -> 2250)
                    ang = hdr[key_ang]
                    info["angle"] = f"{int(ang*10):04d}"
                    date = hdr[key_date]
                    ut = hdr[key_ut]
                    info["utc"] = f"{date}T{ut}"
                
                if o_or_e == "o":
                    df_res = pd.DataFrame(info.values(), index=info.keys()).T
                    df_res_list.append(df_res)
                    print(f"Ratio e/o = {info[f'flux_e']/info['flux_o']}")
                    assert False, df_res
            # 1 set results (Length = 4)
            df_res = pd.concat(df_res_list, axis=0)
            df_res = df_res.reset_index()
        
            # Calculate linear polarization degree P
            P, Perr = calc_P_4angle(df_res)
            print(f"  polarization degree P = {P:.3f}+-{Perr:.3f}")
            P_list.append(P)
            Perr_list.append(Perr)

            # Obtain phase angle with object name
            # Use the first time
            ut = df_res.at[0, "utc"]
            alpha = utc2alpha(args.obj, ut, args.loc)
            alpha_list.append(alpha)

        
        # Round parameters
        # Save results
        if args.out:
            out = args.out
        else:
            out = "polres_WFGS2.txt"
        out = os.path.join(outdir, out)
        df_all = pd.DataFrame(dict(
            alpha=alpha_list, P=P_list, Perr=Perr_list
            ))
        df_all.to_csv(out, sep=" ", index=False)
