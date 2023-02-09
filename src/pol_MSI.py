#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with MSI.
The format of input file is as follows:
xo yo xe ye fits
273 185 239 67 msi221221_805278.fits
273 183 239 65 msi221221_805279.fits
.

The position angle of HWP is saved as RET-ANG2.

RET-ANG2=             22.50000 / [deg] Position angle of retarder plate 2
The position angle of instument due to rotator is saved as INSROT.
INSROT  =             73.25892 / [deg] Typical instrument rotator angle
Typical postion angle of instrument is saved as INST-PA (fixed value).
INST-PA =               -0.520 / [deg] Typical position angle of instrument
"""
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
from scipy.spatial import KDTree
import sep
import astropy.io.fits as fits

from polana.util import *
from polana.visualization import mycolor, myls


if __name__ == "__main__":
    parser = ap(
        description="Do photometry for images obtained with MSI.")
    parser.add_argument(
        "obj", type=str, 
        help="Object name")
    parser.add_argument(
        "inp", type=str, nargs="*",
        help="Input file with certain format")
    parser.add_argument(
        "--loc", type=str, default="Q33",
        help="Observation location (MPC code)")
    parser.add_argument(
        "--mp", action='store_true',
        help='Save phase angle in the output for minor planet')
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
    print(f"  Aperture radius {radius} pix")
    print(f"  filter {band}-band")

    key_texp = "EXPTIME"
    key_date = "DATE-OBS"
    key_ut = "UT-STR"
    # 0, 45, 22.5, 67.5
    key_ang = "RET-ANG2"
    # Inverse gain e/ADU
    key_gain = "GAIN"
    # The position angle of instument due to rotator is saved as INSROT.
    key_insrot = "INSROT"
    # Typical postion angle of instrument is saved as INST-PA (fixed value).
    key_instpa = "INST-PA"
    
    u_list, uerr_list, q_list, qerr_list = [], [], [], []
    alpha_list, P_list, Perr_list = [], [], []
    theta_list, thetaerr_list     = [], []
    insrot_list, instpa_list      = [], []
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
                # ex) GAIN    =                 1.65 / [electrons/DN] Effective AD conversion factor
                gain = hdr[key_gain]

                # Obtain rotator angle (INSROT) 
                # and position angle of the instrument (INSTPA)
                insrot = hdr[key_insrot]
                instpa = hdr[key_instpa]
                
                # Read 2-d image
                img = src.data
                ny, nx = img.shape[0], img.shape[1]
                print(f"    Data dimension nx, ny = {nx}, {ny}")
                # Exposure time
                t_exp = src.header[key_texp]
                print(f"    Exposure time {t_exp} s")

                # Original coordinates
                xo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xo"]
                yo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "yo"]
                xe0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xe"]
                ye0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "ye"]

                # Use around width from (xo0, yo0) and (xe0, ye0)
                wi = args.width/2.0

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
                
                # Do not work well for MSI data !
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
                    bgerr = 0

                else:
                    img_e, bg_info_e = remove_background2d_pol(img_e)
                    img_o, bg_info_o = remove_background2d_pol(img_o)
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
                dth     = 5
                minarea = 10
                objects_e = sep.extract(img_e, dth, err=bgerr_e, minarea=minarea, mask=None)
                objects_o = sep.extract(img_o, dth, err=bgerr_o, minarea=minarea, mask=None)
                N_obj_e   = len(objects_e)
                N_obj_o   = len(objects_o)
                assert N_obj_e == 1, "Check the coordinates"
                assert N_obj_o == 1, "Check the coordinates"

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
                print(f"    xe1, ye1 = {xe1_full:.2f}, {ye1_full:.2f}")

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
                print(f" xo0, yo0 = {xo0}, {yo0}")
                print(f"  flux_o, fluxerr_o, SNR_o = {flux_o:.2f}, {fluxerr_o:.2f}, {flux_o/fluxerr_o:.1f}")
                flux_e, fluxerr_e, eflag_e = sep.sum_circle(
                    img_e, [xe1], [ye1], r=radius, err=bgerr_e, gain=gain,
                    bkgann=bkgann)
                flux_e, fluxerr_e = float(flux_e), float(fluxerr_e)

                print(f"Ratio e/o = {flux_e/flux_o}")
                # Do photometry ===============================================


                # Plot photometry region ======================================
                if args.photmap:
                    out = os.path.join(photmapdir, f"{fi}_photmap.png")
                    label_o = f"{args.obj} (xo, yo)=({xo1_full:.1f}, {yo1_full:.1f})"
                    label_e = f"{args.obj} (xe, ye)=({xe1_full:.1f}, {ye1_full:.1f})"

                    color_o, color_e = mycolor[0], mycolor[1]
                    ls = myls[0]

                    # Plot src image after 5-sigma clipping 
                    sigma = 5
                    _, vmin, vmax = sigmaclip(img, sigma, sigma)

                    fig = plt.figure(figsize=(12,int(12*ny/nx)))
                    ax = fig.add_subplot(111)
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
                # # Noise calculation ===================================================

                # Rotatie angle 0, 22.5 deg (22.5 deg -> 2250)
                ang = hdr[key_ang]
                info[f"flux_o"]    = flux_o
                info[f"fluxerr_o"] = fluxerr_o
                info[f"flux_e"]    = flux_e
                info[f"fluxerr_e"] = fluxerr_e
                info["angle"] = f"{int(ang*10):04d}"
                date = hdr[key_date]
                ut = hdr[key_ut]
                info["utc"] = f"{date}T{ut}"
                df_res = pd.DataFrame(info.values(), index=info.keys()).T
                df_res_list.append(df_res)
            # 1 set results (Length = 4)
            df_res = pd.concat(df_res_list, axis=0)
            df_res = df_res.reset_index()
        
            # Calculate linear polarization degree P
            u, uerr, q, qerr, P, Perr, theta, thetaerr = polana_4angle(df_res)
            print(f"  Polarization degree P = {P:.3f}+-{Perr:.3f}")
            print(f"  Position              = {theta:.3f}+-{thetaerr:.3f}")
            u_list.append(u)
            uerr_list.append(uerr)
            q_list.append(q)
            qerr_list.append(qerr)
            P_list.append(P)
            Perr_list.append(Perr)
            theta_list.append(theta)
            thetaerr_list.append(thetaerr)
            insrot_list.append(insrot)
            instpa_list.append(instpa)

            if args.mp:
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
            out = "polres_MSI.txt"
        out = os.path.join(outdir, out)

        N = len(P_list)
        if args.mp:
            df_all = pd.DataFrame(dict(
                obj=[args.obj]*N, inst=["MSI"]*N, band=[band]*N
                alpha=alpha_list, 
                u=u_list, uerr=uerr_list,
                q=q_list, qerr=qerr_list,
                P=P_list, Perr=Perr_list,
                theta=theta_list, thetaerr=thetaerr_list,
                insrot=insrot_list, instpa=instpa_list,
                ))
        else:
            df_all = pd.DataFrame(dict(
                obj=[args.obj]*N, inst=["MSI"]*N, band=[band]*N
                u=u_list, uerr=uerr_list,
                q=q_list, qerr=qerr_list,
                P=P_list, Perr=Perr_list, 
                theta=theta_list, thetaerr=thetaerr_list,
                insrot=insrot_list, instpa=instpa_list,
            ))
        df_all.to_csv(out, sep=" ", index=False)
