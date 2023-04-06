#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with Kanata/HONIR.
Pixel scale is 0.294 arcsec/pix. (Akitaya+2014, Proc. of SPIE)

Both ordainary and Extra-ordinary sources exist in a single fits.
The format of input file is as follows:
xo yo xe ye fits
88 267 248 271 HN0322604opt00_bt_bs_fl_clip.fits
86 269 248 271 HN0322605opt00_bt_bs_fl_clip.fits
.

The position angle of HWP is saved as 'HWPANGLE'.
HWPANGLE=                   0. / Position angle of half-wave plate (wh18)
The position angle of instumental rotator (not instument) is saved as 
'CROT-STR' and 'CROT-END'.
CROT-STR=              56.7131 / PA of Cs rotator [deg] at exp start (Tel Log)
CROT-END=              57.3702 / PA of Cs rotator [deg] at exp start (Tel Log)


Note:
1. The output time is mid-exposure time.
2. theta and theta error are in radians.
3. With --mp option, phase angle and position angle of the scattering plane
can be obtained. Pr and Ptheta are calculated.
But those calculations need to be done using the same aspect data in the 
table of the paper.
4. Typical postion angle of instrument (INSTPA) is 
necessary only when determination of coefficients for pa offset correction?
Anyway, INSTPA can be ignored in this script because INSTPA=0 for HONIR data,
at least obtained in December 2022.
"""
import os
import datetime
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
import sep
import astropy.io.fits as fits

from polana.util import utc2alphaphi, remove_bg_2d, loc_Kanata
from polana.util_pol import (
    polana_4angle, cor_poleff, cor_instpol, cor_paoffset,
    calc_Ptheta, projectP2scaplane)
from polana.visualization import mycolor
from movphot.photfunc import obtain_winpos


if __name__ == "__main__":
    parser = ap(
        description="Do photometry for images obtained with HONIR.")
    parser.add_argument(
        "obj", type=str, 
        help="Object name")
    parser.add_argument(
        "inp", type=str, nargs="*",
        help="Input file with certain format")
    parser.add_argument(
        "--mp", action='store_true',
        help='Save phase angle in the output for minor planet')
    parser.add_argument(
        "--pp", action='store_true',
        help='Do preprocess')
    parser.add_argument(
        "--fitsdir", type=str, default=".",
        help="Fits directory")
    parser.add_argument(
        "--radius", type=float, default=10, 
        help="aperture radius in pixel")
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
    parser.add_argument(
        "--width", type=int, default=50,
        help="x and y width in pixel")
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
    # Use around width from (xo0, yo0) and (xe0, ye0)
    wi = args.width/2.0
    assert wi >= radius*1.1, "Width should be larger than 1.1 radius."

    inst = "HONIR"
    print(f"  Aperture radius {radius} pix")
    print(f"  filter {band}-band")

    
    # in sec
    key_texp = "EXPTIME"
    key_date = "DATE-OBS"
    # UTC at exposure start (hh:mm:ss)
    key_ut = "UT-STR"
    # 0, 45, 22.5, 67.5
    key_ang = "HWPANGLE"
    # Inverse gain e/ADU
    key_gain = "GAIN"
    # TODO: check
    
    # Position angle of Instumental rotator
    key_insrot0 = "CROT-STR"
    key_insrot1 = "CROT-END"
    # There is no keyword for instpa.
    key_instpa = None
    
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
        N_fits_per_set = 4
        N_set = int(N_fits/N_fits_per_set)
         
        for idx_set in range(N_set):
            print("")
            print(f"Start analysis of {idx_set+1}/{N_set}-th set")
            # List to save results 
            df_res_list = []
            for idx_fi in range(N_fits_per_set):
                fi = df_in.at[idx_set*N_fits_per_set+idx_fi, "fits"]
                print("")
                print(f"    Start analysis of {idx_fi+1}-th fits")
                fi_path = os.path.join(fitsdir, fi)
                src = fits.open(fi_path)[0]
                hdr = src.header 

                # Set inverse gain
                # Read GAIN in fits header
                # ex) GAIN    =                2.690 / Detecctor gain [electron/ADU]
                gain = hdr[key_gain]

                # Obtain position angle of instument
                insrot0 = hdr[key_insrot0]
                insrot1 = hdr[key_insrot1]
                # Average
                insrot = (insrot0 + insrot1)*0.5

                # Read 2-d image
                img = src.data
                ny, nx = img.shape[0], img.shape[1]
                print(f"    Data dimension nx, ny = {nx}, {ny}")
                # Exposure time
                texp = src.header[key_texp]
                print(f"    Exposure time {texp} s")


                # Original coordinates
                xo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xo"]
                yo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "yo"]
                xe0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xe"]
                ye0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "ye"]


                xmin_e, xmax_e = xe0 - wi - 1, xe0 + wi
                ymin_e, ymax_e = ye0 - wi - 1, ye0 + wi
                xmin_e, xmax_e = int(xmin_e), int(xmax_e)
                ymin_e, ymax_e = int(ymin_e), int(ymax_e)
                img_e = img[ymin_e:ymax_e, xmin_e:xmax_e]

                xmin_o, xmax_o = xo0 - wi - 1, xo0 + wi
                ymin_o, ymax_o = yo0 - wi - 1, yo0 + wi
                xmin_o, xmax_o = int(xmin_o), int(xmax_o)
                ymin_o, ymax_o = int(ymin_o), int(ymax_o)
                img_o = img[ymin_o:ymax_o, xmin_o:xmax_o]

                # Save photometry info.
                info = dict()
                

                # Background subtraction ======================================
                # Convert to float to suppress error
                img_e = img_e.astype(np.float32)
                img_o = img_o.astype(np.float32)
                #print(f"Original Mean: {np.mean(img)}")
                #print(f"Original Median: {np.median(img)}")
                if args.ann:
                    print("    !! Do not subtract background with sep!!")
                    print(f"    !! Simply subtract median {np.median(img):.1f} ADU!!")
                    # For error calculation in sep.sum_circle
                    bgerr_e = 0
                    bgerr_o = 0

                else:
                    img_e, bg_info_e = remove_bg_2d(img_e)
                    img_o, bg_info_o = remove_bg_2d(img_o)
                    bgerr_e = np.round(bg_info_e["rms"], 2)
                    bgerr_o = np.round(bg_info_o["rms"], 2)


                #print("")
                #print(f"Subtracted Mean: {np.mean(img)}")
                #print(f"Subtracted Median: {np.median(img)}")
                #print(f"Estimated sky error per pixel is {bgerr} [ADU]")
                #print("")
                info["gloabalrms_e"] = bgerr_e
                info["gloabalrms_o"] = bgerr_o
                info["level_mean_e"] = np.mean(img_e)
                info["level_mean_o"] = np.mean(img_o)
                info["level_median_e"] = np.median(img_e)
                info["level_median_o"] = np.median(img_o)
                # Background subtraction ======================================

                # Source detection for baricentric search =====================
                # Assuming background subtracted
                # 5-sigma detection
                dth     = 3
                minarea = 10
                objects_e = sep.extract(
                    img_e, dth, err=bgerr_e, minarea=minarea, mask=None)
                objects_o = sep.extract(
                    img_o, dth, err=bgerr_o, minarea=minarea, mask=None)
                N_obj_e   = len(objects_e)
                N_obj_o   = len(objects_o)
                print("x, y of objects in e")
                print(objects_e["x"])
                print(objects_e["y"])
                print("x, y of objects in o")
                print(objects_o["x"])
                print(objects_o["y"])
                # Soooooo important
                assert N_obj_e == 1, "Check the image. There might be a cosmic ray?"
                assert N_obj_o == 1, "Check the image. There might be a cosmic ray?"

                # Search the baricenters after cut and bgsub
                print(f"  Aperture location after baricenter search")
                # Ordinary
                xo1, yo1 = objects_o["x"][0], objects_o["y"][0]
                xo1_full = xo1 + xmin_o
                yo1_full = yo1 + ymin_o
                print(f"    xo0, yo0 = {xo0:.2f}, {yo0:.2f}")
                print(f"    xo1, yo1 = {xo1:.2f}, {yo1:.2f}")

                # Extra-ordinary
                xe1, ye1 = objects_e["x"][0], objects_e["y"][0]
                # In original image
                xe1_full = xe1 + xmin_e
                ye1_full = ye1 + ymin_e
                print(f"  Aperture location after baricenter search")
                print(f"    xe0, ye0 = {xe0:.2f}, {ye0:.2f}")
                print(f"    xe1, ye1 = {xe1:.2f}, {ye1:.2f}")


                # winpos
                # initial guesses are returns of sep.extract
                xwino, ywino, flag = obtain_winpos(img_o, [xo1], [yo1], radius, nx, ny)
                xwine, ywine, flag = obtain_winpos(img_e, [xe1], [ye1], radius, nx, ny)
                xwino_full = xwino[0] + xmin_o
                ywino_full = ywino[0] + ymin_o
                xwine_full = xwine[0] + xmin_e
                ywine_full = ywine[0] + ymin_e
                
                xe1, ye1, xo1, yo1 = xwine, ywine, xwino, ywino
                xe1_full = xwine_full
                xo1_full = xwino_full
                ye1_full = ywine_full
                yo1_full = ywino_full
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
                flux_o, fluxerr_o, eflag_o = sep.sum_circle(
                    img_o, [xo1], [yo1], r=radius, err=bgerr_o, gain=gain,
                    bkgann=bkgann)
                flux_o, fluxerr_o = float(flux_o), float(fluxerr_o)
                flux_e, fluxerr_e, eflag_e = sep.sum_circle(
                    img_e, [xe1], [ye1], r=radius, err=bgerr_e, gain=gain,
                    bkgann=bkgann)
                flux_e, fluxerr_e = float(flux_e), float(fluxerr_e)
                print(f"  flux_o, fluxerr_o, SNR_o = {flux_o:.2f}, {fluxerr_o:.2f}, {flux_o/fluxerr_o:.1f}")
                print(f"  flux_e, fluxerr_e, SNR_e = {flux_e:.2f}, {fluxerr_e:.2f}, {flux_e/fluxerr_e:.1f}")
                print(f"Ratio e/o = {flux_e/flux_o}")
                # Do photometry ===============================================


                # Plot photometry region ======================================
                if args.photmap:
                    import matplotlib.pyplot as plt
                    from matplotlib.collections import PatchCollection
                    from matplotlib.patches import Circle
                    from scipy.stats import sigmaclip

                    out = os.path.join(photmapdir, f"{fi}_photmap.png")
                    label_o = f"{args.obj} (xo, yo)=({xo1_full:.1f}, {yo1_full:.1f})"
                    label_e = f"{args.obj} (xe, ye)=({xe1_full:.1f}, {ye1_full:.1f})"

                    color_o, color_e = mycolor[0], mycolor[1]
                    ls = "solid"

                    # Plot src image after 5-sigma clipping 
                    sigma = 5

                    fig = plt.figure(figsize=(12,int(12*ny/nx)))
                    ax = fig.add_subplot(111)
                    _, vmin, vmax = sigmaclip(img, sigma, sigma)
                    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)

                    # Ordainary
                    ax.scatter(
                        xo1_full, yo1_full, color=color_o, s=radius, lw=1, 
                        facecolor="None", alpha=1, label=label_o)
                    ax.add_collection(PatchCollection(
                        [Circle((xo1_full, yo1_full), radius)], color=color_o, ls=ls, 
                        lw=1, facecolor="None", label=None)
                        )

                    # Extra-ordainary
                    ax.scatter(
                        xe1_full, ye1_full, color=color_e, s=radius, lw=1, 
                        facecolor="None", alpha=1, label=label_e)
                    ax.add_collection(PatchCollection(
                        [Circle((xe1_full, ye1_full), radius)], color=color_e, ls=ls, 
                        lw=1, facecolor="None", label=None)
                        )

                    ax.set_xlim([0, nx])
                    ax.set_ylim([0, ny])
                    ax.legend().get_frame().set_alpha(1.0)
                    ax.invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(out, dpi=200)
                    plt.close()
                # Plot photometry region ======================================


                # Rotatie angle 0, 22.5 deg (22.5 deg -> 2250)
                ang = hdr[key_ang]
                info[f"flux_o"]    = flux_o
                info[f"fluxerr_o"] = fluxerr_o
                info[f"flux_e"]    = flux_e
                info[f"fluxerr_e"] = fluxerr_e
                info["angle"] = f"{int(ang*10):04d}"
                # Average (start and end) pa of instument
                info["insrot"] = insrot
                date = hdr[key_date]
                # Starting time of exposure
                utc0 = hdr[key_ut]
                utc0 = f"{date}T{utc0}"
                # Convert to mid-time of exposure
                utc0_dt = datetime.datetime.strptime(utc0, "%Y-%m-%dT%H:%M:%S")
                utcmid_dt = utc0_dt + datetime.timedelta(seconds=texp)
                utcmid = datetime.datetime.strftime(utcmid_dt, "%Y-%m-%dT%H:%M:%S")
                info["utc"] = utcmid
                info["fits"] = fi
                df_res = pd.DataFrame(info.values(), index=info.keys()).T
                df_res_list.append(df_res)
            # 1 set results (Length = 4)
            df_res = pd.concat(df_res_list, axis=0)
            df_res = df_res.reset_index()
        
            # Calculate polarization parameters
            u, uerr, q, qerr = polana_4angle(df_res, inst)
            u_list.append(u)
            uerr_list.append(uerr)
            q_list.append(q)
            qerr_list.append(qerr)
            
            # Position angle of instumental rotator 
            # 1. theta1 is average PA of instrument rotator at 0 and 45 
            insrot000 = df_res[df_res["angle"]=="0000"].insrot.values.tolist()[0]
            insrot450 = df_res[df_res["angle"]=="0450"].insrot.values.tolist()[0]
            insrot1   = (insrot000 + insrot450)*0.5
            insrot1_list.append(insrot1)
            # 2. theta2 is average PA of instrument rotator at 225 and 675 
            insrot225 = df_res[df_res["angle"]=="0225"].insrot.values.tolist()[0]
            insrot675 = df_res[df_res["angle"]=="0675"].insrot.values.tolist()[0]
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
            
            # Delete in the future.
            if args.mp:
                # Obtain phase angle with object name
                # Use the first time
                ut = df_res.at[0, "utc"]
                alpha, phi = utc2alphaphi(args.obj, ut, loc_Kanata)
                alpha_list.append(alpha)
                phi_list.append(phi)

        
        # Round parameters
        # Save results
        if args.out:
            out = args.out
        else:
            out = "polres_HONIR.txt"
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
                obj=[args.obj]*N, inst=[inst]*N, band=[band]*N,
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

        if args.pp:
            # Post processes
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

        df.to_csv(out, sep=" ", index=False)
