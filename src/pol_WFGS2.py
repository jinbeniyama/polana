#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with Nayuta/WFGS2.
Pixel scale is 0.198 arcsec/pix. (Kawakami+2021, Stars and Galaxies)

Ordainary and Extra-ordinary sources can be distinguished from the file name.
The format of input file is as follows:
x y fits
477 1475 wfgs2_221220_0013.HD19820.Rc.e.cr.fits
480 1485 wfgs2_221220_0013.HD19820.Rc.o.cr.fits
484 1477 wfgs2_221220_0014.HD19820.Rc.e.cr.fits
484 1486 wfgs2_221220_0014.HD19820.Rc.o.cr.fits
.

The position angle of HWP is saved as 'HWP-AGL'.
HWP-AGL =                 22.5 / Half-wave plate rotation angle (deg)
The position angle of instumental rotator is saved as INSROT.
INSROT  =              135.583 / Typical inst rot. Angle ar exp.(degree)


Note:
1. The output time is mid-exposure time (already saved in header as UT-CEN).
2. theta and theta error are in radians.
3. With --mp option, phase angle and position angle of the scattering plane
can be obtained. Pr and Ptheta are calculated.
But those calculations need to be done using the same aspect data in the 
table of the paper.
4. Typical postion angle of instrument saved as INST-PA (fixed value) is 
necessary only when determination of coefficients for pa offset correction?
INST-PA =                  0.0 / Approx PA of instrument (deg)
Anyway, INSTPA can be ignored in this script because INSTPA=0 for WFGS2 data,
at least obtained in December 2022.
"""
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
import sep
import astropy.io.fits as fits

from polana.util import utc2alphaphi, remove_bg_2d, loc_Nayuta
from polana.util_pol import (
    polana_4angle, cor_poleff, cor_instpol, cor_paoffset,
    calc_Ptheta, projectP2scaplane)
from polana.visualization import mycolor


if __name__ == "__main__":
    parser = ap(
        description="Do photometry for images obtained with WFGS2.")
    parser.add_argument(
        "obj", type=str, 
        help="Object name")
    parser.add_argument(
        "inp", type=str, nargs="*",
        help="Input file with appropreate format")
    parser.add_argument(
        "--mp", action='store_true',
        help='Save phase angle in the output for minor planet')
    parser.add_argument(
        "--pp", action='store_true',
        help='Do preprocess')
    parser.add_argument(
        "--fitsdir", type=str, default=".",
        help="Directory in which fits exists")
    parser.add_argument(
        "--radius", type=float, default=40, 
        help="aperture radius in pixel (p_scale = 0.198 arcsec/s)")
    parser.add_argument(
        "--ann", action='store_true', default=False,
        help='Do photometry with annulus')
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
        "-p", "--photmap", action='store_true',
        help='create photometry region map (a bit slow)')
    args = parser.parse_args()
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ## Output photometry region png in the directory
    if args.photmap:
        photmapdir = os.path.join(outdir, "photregion")
        os.makedirs(photmapdir, exist_ok=True)

    fitsdir = args.fitsdir
    radius = args.radius
    band = args.band
    inst = "WFGS2"
    # See SCALE in the fits header
    p_scale = 0.198
    radius_arcsec = radius * p_scale
    print(f"  Aperture radius {radius} pix = {radius_arcsec} arcsec")
    print(f"  filter {band}-band")

    key_texp = "EXPTIME"
    # Only date
    key_date = "DATE-OBS"
    # Central UTC
    key_ut = "UT-CEN"
    # Angle of retarder 0, 45, 22.5, 67.5
    key_ang = "HWP-AGL"
    # Inverse gain e/ADU
    key_gain = "GAIN"
    # The position angle of instument due to rotator is saved as INSROT.
    key_insrot = "INSROT"
    # Typical postion angle of instrument is saved as INST-PA (fixed value).
    key_instpa = "INST-PA"
    
    u_list, uerr_list, q_list, qerr_list = [], [], [], []
    alpha_list, phi_list = [], []
    texp_list = []
    insrot1_list, insrot2_list    = [], []
    utc000_list, utc450_list      = [], []
    utc225_list, utc675_list      = [], []
    fi000_list, fi450_list        = [], []
    fi225_list, fi675_list        = [], []

    for x in args.inp:
        # Read input files
        df_in = pd.read_csv(x, sep=" ")
        N_fits = len(df_in)
        # 1 set = 4 angels x 2 o/e
        N_fits_per_set = 8
        # The number of sets
        N_set = int(N_fits/N_fits_per_set)
         
        for idx_set in range(N_set):
            print("")
            print(f"Start analysis of {idx_set+1}/{N_set}-th set")
            # List to save results of a set
            df_res_list = []
            for idx_fi in range(N_fits_per_set):
                fi = df_in.at[idx_set*N_fits_per_set+idx_fi, "fits"]
                fi_path = os.path.join(fitsdir, fi)
                src = fits.open(fi_path)[0]
                hdr = src.header 

                # Judge Ordinary or Extra-ordinary from fitsname
                # At least applicable 2010 XC15 data in December 2022
                # like 'wfgs2_221220_0001.2010XC15.Rc.e.cr.fits'
                o_or_e = fi.split(".")[-3]

                print(f"    Start analysis of {idx_fi+1}/{N_fits_per_set}-th fits")
                print(f"      {fi} {o_or_e}")

                # Set inverse gain
                # Read GAIN in fits header
                # ex) GAIN    =                 2.28 / CCD gain in e/ADU, ref: MINT wiki
                gain = hdr[key_gain]

                # Obtain position angle of insrumental rotator (INSROT) 
                insrot = hdr[key_insrot]
                
                # Read 2-d image
                img = src.data
                ny, nx = img.shape[0], img.shape[1]
                print(f"    Data dimension nx, ny = {nx}, {ny}")
                # Exposure time
                texp = src.header[key_texp]
                print(f"    Exposure time {texp} s")

                # Save photometry info.
                # assuming e -> o -> e -> o ... .
                if o_or_e == "e":
                    info = dict()
                else:
                    pass
               
                # Background subtraction ======================================
                # Convert to float to suppress unfavorable error
                img = img.astype(np.float32)
                if args.ann:
                    print("    !! Do not subtract background !!")
                    # Temporally
                    bgerr = 0
                else:
                    img, bg_info = remove_bg_2d(img)
                    bgerr = np.round(bg_info["rms"], 2)
                    #print("")
                    #print(f"Subtracted Mean: {np.mean(img)}")
                    #print(f"Subtracted Median: {np.median(img)}")
                    #print(f"Estimated sky error per pixel is {bgerr} [ADU]")
                    #print("")

                # Save info.
                info["gloabalrms"] = bgerr
                info["level_mean"] = np.mean(img)
                info["level_median"] = np.median(img)
                # Background subtraction ======================================


                # Source detection ============================================
                # 5-sigma detection
                #dth     = 5
                #minarea = 15
                #objects = extract(img, dth, minarea, bgerr, mask=None)
                #N_obj   = len(objects)
                #print(f"{N_obj} objects detected on {idx_fi+1}-th fits {fi}.")
                
                # Original coordinates of the target
                x0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "x"]
                y0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "y"]
                print(f"  {fi}, {x0}, {y0}")

                # # Search the most suitable objects
                # x_base, y_base = objects["x"], objects["y"]
                # tree_base = KDTree(list(zip(x_base, y_base)), leafsize=10)
                # # Ordinary
                # res = tree_base.query_ball_point((x0, y0), radius)

                # # New barycenters
                # x1, y1 = res
                # Source detection ============================================


                # Do photometry ===============================================
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
                    img, [x0], [y0], r=radius, err=bgerr, gain=gain,
                    bkgann=bkgann)
                flux, fluxerr = float(flux), float(fluxerr)
                # Do photometry ===============================================


                # Plot photometry region ======================================
                if args.photmap:
                    import matplotlib.pyplot as plt
                    from matplotlib.collections import PatchCollection
                    from matplotlib.patches import Circle
                    from scipy.stats import sigmaclip

                    out = os.path.join(photmapdir, f"{fi}_photmap.png")
                    label = f"{args.obj} (x, y)=({x0}, {y0})"
                    color = mycolor[0]
                    ls = "solid"

                    # Plot src image after 5-sigma clipping 
                    sigma = 5
                    _, vmin, vmax = sigmaclip(img, sigma, sigma)

                    fig = plt.figure(figsize=(12,int(12*ny/nx)))
                    ax = fig.add_subplot(111)
                    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                    ax.scatter(
                        x0, y0, color=color, s=radius, lw=1, 
                        facecolor="None", alpha=1, label=label)
                    ax.add_collection(PatchCollection(
                        [Circle((x0, y0), radius)],
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


                # Noise calculation ===========================================
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
                # # Noise calculation =========================================

                info[f"flux_{o_or_e}"]    = flux
                info[f"fluxerr_{o_or_e}"] = fluxerr
                print(
                    f"    flux, fluxerr, SNR = "
                    f"{flux:.2f}, {fluxerr:.2f}, {flux/fluxerr:.1f}"
                    )

                # Assume order of input fits as 
                # e (rot X)-> o (rot X)-> e (rot Y)-> o (rot Y) ...
                if o_or_e == "e":
                    # Rotation angle 0, 22.5 deg (22.5 deg -> 2250), ...
                    ang = hdr[key_ang]
                    info["angle"] = f"{int(ang*10):04d}"
                    # pa of instument
                    info["insrot"] = insrot
                    date = hdr[key_date]
                    # Already mid-time of exposure
                    ut = hdr[key_ut]
                    info["utc"] = f"{date}T{ut}"
                    # Redister fitsname of ordinary
                    info["fits"] = fi
                
                if o_or_e == "o":
                    df_res = pd.DataFrame(info.values(), index=info.keys()).T
                    df_res_list.append(df_res)
                    print(f"Ratio e/o = {info[f'flux_e']/info['flux_o']}")
            # Concatenate results of the set (Length = 4)
            df_res = pd.concat(df_res_list, axis=0)
            df_res = df_res.reset_index()
        
            # Calculate normalized Stokes parameters 
            u, uerr, q, qerr  = polana_4angle(df_res, inst)
            u_list.append(u)
            uerr_list.append(uerr)
            q_list.append(q)
            qerr_list.append(qerr)

            # Position angle of instumental rotator 
            # 1. theta1 is average PA of instrument rotator at 0 and 45 
            insrot000 = df_res[df_res["angle"]=="0000"].insrot.values.tolist()[0]
            insrot450 = df_res[df_res["angle"]=="0450"].insrot.values.tolist()[0]
            # Average
            insrot1   = (insrot000 + insrot450)*0.5
            insrot1_list.append(insrot1)
            # 2. theta2 is average PA of instrument rotator at 225 and 675 
            insrot225 = df_res[df_res["angle"]=="0225"].insrot.values.tolist()[0]
            insrot675 = df_res[df_res["angle"]=="0675"].insrot.values.tolist()[0]
            # Average
            insrot2   = (insrot225 + insrot675)*0.5
            insrot2_list.append(insrot2)

            texp_list.append(texp)

            # UTC
            utc000 = df_res[df_res["angle"]=="0000"].utc.values.tolist()[0]
            utc450 = df_res[df_res["angle"]=="0450"].utc.values.tolist()[0]
            utc225 = df_res[df_res["angle"]=="0225"].utc.values.tolist()[0]
            utc675 = df_res[df_res["angle"]=="0675"].utc.values.tolist()[0]
            utc000_list.append(utc000)
            utc450_list.append(utc450)
            utc225_list.append(utc225)
            utc675_list.append(utc675)

            # Fits
            fi000 = df_res[df_res["angle"]=="0000"].fits.values.tolist()[0]
            fi450 = df_res[df_res["angle"]=="0450"].fits.values.tolist()[0]
            fi225 = df_res[df_res["angle"]=="0225"].fits.values.tolist()[0]
            fi675 = df_res[df_res["angle"]=="0675"].fits.values.tolist()[0]
            fi000_list.append(fi000)
            fi450_list.append(fi450)
            fi225_list.append(fi225)
            fi675_list.append(fi675)

            if args.mp:
                # Obtain phase angle of a minor planet (mp)
                # Use the first time
                ut = df_res.at[0, "utc"]
                alpha, phi = utc2alphaphi(args.obj, ut, loc_Nayuta)
                alpha_list.append(alpha)
                phi_list.append(phi)

        # Save results
        if args.out:
            out = args.out
        else:
            out = "polres_WFGS2.txt"
        out = os.path.join(outdir, out)

        N = len(u_list)
        if args.mp:
            df = pd.DataFrame(dict(
                obj=[args.obj]*N, inst=[inst]*N, band=[band]*N,
                alpha=alpha_list, phi=phi_list,
                texp=texp_list,
                utc000 = utc000_list, 
                utc450 = utc450_list, 
                utc225 = utc225_list, 
                utc675 = utc675_list, 
                fi000 = fi000_list, 
                fi450 = fi450_list, 
                fi225 = fi225_list, 
                fi675 = fi675_list, 
                u=u_list, uerr=uerr_list,
                q=q_list, qerr=qerr_list,
                insrot1=insrot1_list, 
                insrot2=insrot2_list, 
                ))
        else:
            df = pd.DataFrame(dict(
                obj=[args.obj]*N, inst=["WFGS2"]*N, band=[band]*N,
                texp=texp_list,
                utc000 = utc000_list, 
                utc450 = utc450_list, 
                utc225 = utc225_list, 
                utc675 = utc675_list, 
                fi000 = fi000_list, 
                fi450 = fi450_list, 
                fi225 = fi225_list, 
                fi675 = fi675_list, 
                u=u_list, uerr=uerr_list,
                q=q_list, qerr=qerr_list,
                insrot1=insrot1_list, 
                insrot2=insrot2_list, 
            ))

        # Add only for post processes
        df["instpa"] = 0.
        

        # Do post processes here
        if args.pp:
            # Post processes (pp)
            df = cor_poleff(
                df, inst, band, 
                "q", "u", "qerr", "uerr", 
                "q_cor0", "u_cor0", "qerr_ran_cor0", "uerr_ran_cor0",
                "qerr_sys_cor0", "uerr_sys_cor0",
                "qerr_cor0", "uerr_cor0"
                )
            df = cor_instpol(
                df, inst, band, 
                "q_cor0", "u_cor0", 
                "qerr_ran_cor0", "uerr_ran_cor0",
                "qerr_sys_cor0", "uerr_sys_cor0",
                "q_cor1", "u_cor1",
                "qerr_ran_cor1", "uerr_ran_cor1",
                "qerr_sys_cor1", "uerr_sys_cor1",
                "qerr_cor1", "uerr_cor1",
                "insrot1", "insrot2")
            df = cor_paoffset(
                df, inst, band, 
                "q_cor1", "u_cor1", 
                "qerr_ran_cor1", "uerr_ran_cor1",
                "qerr_sys_cor1", "uerr_sys_cor1",
                "q_cor2", "u_cor2",
                "qerr_ran_cor2", "uerr_ran_cor2",
                "qerr_sys_cor2", "uerr_sys_cor2",
                "qerr_cor2", "uerr_cor2",
                "instpa")
            df = calc_Ptheta(
                df, "P_cor2", "theta_cor2", "Perr_cor2", "thetaerr_cor2",
                "q_cor2", "u_cor2", "qerr_cor2", "uerr_cor2")
            if args.mp:
                df = projectP2scaplane(
                    df, "Pr", "Prerr", "thetar", "thetarerr", 
                    "P_cor2", "Perr_cor2", "theta_cor2", "thetaerr_cor2", "phi")
        df.to_csv(out, sep=" ", index=False)
