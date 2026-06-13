#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with Subaru/FOCAS.
Pixel scale is 0.1038 arcsec/pix. (https://subarutelescope.org/Instruments/FOCAS/parameters.html)

Both ordainary and Extra-ordinary sources exist in a single fits.

The position angle of HWP and position angle of instrumental rotator are not saved in the fits header?

Note:
1. The output time is mid-exposure time.
2. theta and theta error are in radians.
3. With --mp option, phase angle and position angle of the scattering plane
can be obtained. Pr and Ptheta are calculated.
But those calculations need to be done using the same aspect data in the 
table of the paper.
4. Typical postion angle of instrument saved as INST-PA (fixed value) is 
necessary only when determination of coefficients for pa offset correction?
INST-PA =               -0.520 / [deg] Typical position angle of instrument
"""
import os
import datetime
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
import sep
import astropy.io.fits as fits
from matplotlib.patches import Ellipse 


from polana.util import utc2alphaphi, remove_bg_2d, loc_Subaru, obtain_winpos
from polana.util_pol import (
    polana_4angle, cor_poleff, cor_instpol, cor_paoffset, 
    calc_Ptheta, projectP2scaplane)
from polana.visualization import mycolor


def get_args():
    parser = ap(
        description="Do photometry for images obtained with FOCAS.")
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
        "--ann0", type=float, default=None,
        help='Inner annulus in pix')
    parser.add_argument(
        "--ann1", type=float, default=None,
        help='Outer annulus in pix')
    parser.add_argument(
        "--ann_gap", type=float, default=2, 
        help="gap between annulus and circle ")
    parser.add_argument(
        "--ann_width", type=float, default=3, 
        help="width of annulus")
    parser.add_argument(
        "--ell", action='store_true', default=False,
        help='Do photometry with elliptical aperture')
    parser.add_argument(
        "--elong_ratio", dest='elong_ratio', type=float, default=3.0,
        help='Constant to calculate major axis')
    parser.add_argument(
        "--theta", dest='theta', type=float, default=None,
        help='Position angle from +x to major axis in [-pi/2, pi/2]')
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
        "--width", type=int, default=100,
        help="x and y width in pixel")
    parser.add_argument(
      "--bw", type=int, default=16, 
      help="box width and height for bgsubtraction")
    parser.add_argument(
      "--fw", type=int, default=3, 
      help="median filter width and height for bgsubtraction")
    return parser.parse_args()
    

def main(args=None):
    if args == None:
        args = get_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ## Output photometry region png in the directory
    if args.photmap:
        photmapdir = os.path.join(outdir, "photregion")
        os.makedirs(photmapdir, exist_ok=True)

    fitsdir = args.fitsdir
    radius = args.radius
    band = args.band
    inst = "FOCAS"
    
    is_ell = getattr(args, 'ell', False)
    elong_ratio = getattr(args, 'elong_ratio', 1.0)
    theta = getattr(args, 'theta', 0.0)

    if is_ell:
        print(f"  Elliptical Photometry: semi-minor={radius} pix, semi-major={radius*elong_ratio} pix, theta={theta} rad")
    else:
        print(f"  Aperture radius {radius} pix")
    print(f"  filter {band}-band")

    key_texp = "EXPTIME"
    key_date = "DATE-OBS"
    key_ut = "UT-STR"
    key_gain = "GAIN"
    key_insrot = "INSROT"
    key_instpa = "INST-PA"
    
    u_list, uerr_list, q_list, qerr_list = [], [], [], []
    alpha_list, phi_list = [], []
    texp_list = []
    instpa_list = []
    insrot1_list, insrot2_list    = [], []
    utc000_list, utc450_list      = [], []
    utc225_list, utc675_list      = [], []
    fi000_list, fi450_list        = [], []
    fi225_list, fi675_list        = [], []

    flux_000_o_list, flux_000_e_list = [], []
    fluxerr_000_o_list, fluxerr_000_e_list = [], []
    flux_450_o_list, flux_450_e_list = [], []
    fluxerr_450_o_list, fluxerr_450_e_list = [], []
    flux_225_o_list, flux_225_e_list = [], []
    fluxerr_225_o_list, fluxerr_225_e_list = [], []
    flux_675_o_list, flux_675_e_list = [], []
    fluxerr_675_o_list, fluxerr_675_e_list = [], []

    for x in args.inp:
        df_in = pd.read_csv(x, sep=" ")
        N_fits = len(df_in)
        N_fits_per_set = 4
        N_set = int(N_fits/N_fits_per_set)
         
        for idx_set in range(N_set):
            print("")
            print(f"Start analysis of {idx_set+1}/{N_set}-th set")
            df_res_list = []
            for idx_fi in range(N_fits_per_set):
                fi = df_in.at[idx_set*N_fits_per_set+idx_fi, "fits"]
                print("")
                print(f"    Start analysis of {idx_fi+1}-th fits")
                fi_path = os.path.join(fitsdir, fi)
                src = fits.open(fi_path)[0]
                hdr = src.header 

                gain = hdr[key_gain]
                insrot = hdr[key_insrot]
                instpa = hdr[key_instpa]
                
                img = src.data
                ny, nx = img.shape[0], img.shape[1]
                print(f"    Data dimension nx, ny = {nx}, {ny}")
                texp = src.header[key_texp]
                print(f"    Exposure time {texp} s")

                xo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xo"]
                yo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "yo"]
                xe0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xe"]
                ye0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "ye"]

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

                info = dict()
                
                img_e = img_e.astype(np.float32)
                img_o = img_o.astype(np.float32)

                if args.ann:
                    print("    !! Do not subtract background with sep!!")
                    bgerr_e = 0
                    bgerr_o = 0
                else:
                    img_e, bg_info_e = remove_bg_2d(img_e, None, args.bw, args.fw)
                    img_o, bg_info_o = remove_bg_2d(img_o, None, args.bw, args.fw)
                    bgerr_e = np.round(bg_info_e["rms"], 2)
                    bgerr_o = np.round(bg_info_o["rms"], 2)

                info["gloabalrms_e"] = bgerr_e
                info["gloabalrms_o"] = bgerr_o
                info["level_mean_e"] = np.mean(img_e)
                info["level_mean_o"] = np.mean(img_o)
                info["level_median_e"] = np.median(img_e)
                info["level_median_o"] = np.median(img_o)

                # Source detection for baricentric search
                if args.ann:
                    tmp_img_e = img_e - np.median(img_e)
                    tmp_img_o = img_o - np.median(img_o)
                    tmp_err_e = 1.4826 * np.median(np.abs(tmp_img_e - np.median(tmp_img_e)))
                    tmp_err_o = 1.4826 * np.median(np.abs(tmp_img_o - np.median(tmp_img_o)))
                    print(tmp_err_e)
                    print(tmp_err_o)
                else:
                    tmp_img_e = img_e
                    tmp_img_o = img_o
                    tmp_err_e = bgerr_e
                    tmp_err_o = bgerr_o

                dth     = 3
                minarea = 10
                objects_e = sep.extract(tmp_img_e, dth, err=tmp_err_e, minarea=minarea, mask=None)
                objects_o = sep.extract(tmp_img_o, dth, err=tmp_err_o, minarea=minarea, mask=None)

                N_obj_e   = len(objects_e)
                N_obj_o   = len(objects_o)

                local_xo0 = xo0 - xmin_o
                local_yo0 = yo0 - ymin_o
                local_xe0 = xe0 - xmin_e
                local_ye0 = ye0 - ymin_e

                if N_obj_o == 0:
                    print(f"    [Notice] No objects detected for o-ray. Keep initial coordinates.")
                    xo1, yo1 = local_xo0, local_yo0
                elif N_obj_o == 1:
                    xo1 = objects_o["x"][0]
                    yo1 = objects_o["y"][0]
                elif N_obj_o == 2:
                    dist0 = np.hypot(objects_o["x"][0] - local_xo0, objects_o["y"][0] - local_yo0)
                    dist1 = np.hypot(objects_o["x"][1] - local_xo0, objects_o["y"][1] - local_yo0)
                    idx = 0 if dist0 < dist1 else 1
                    print(f"    [Notice] 2 objects detected for o-ray. Selected the closer one (index {idx}).")
                    xo1 = objects_o["x"][idx]
                    yo1 = objects_o["y"][idx]
                else:
                    assert False, f"Check the coordinates, No={N_obj_o} (3 or more objects detected)"

                xo1_full = xo1 + xmin_o
                yo1_full = yo1 + ymin_o

                if N_obj_e == 0:
                    print(f"    [Notice] No objects detected for e-ray. Keep initial coordinates.")
                    xe1, ye1 = local_xe0, local_ye0
                elif N_obj_e == 1:
                    xe1 = objects_e["x"][0]
                    ye1 = objects_e["y"][0]
                elif N_obj_e == 2:
                    dist0 = np.hypot(objects_e["x"][0] - local_xe0, objects_e["y"][0] - local_ye0)
                    dist1 = np.hypot(objects_e["x"][1] - local_xe0, objects_e["y"][1] - local_ye0)
                    idx = 0 if dist0 < dist1 else 1
                    print(f"    [Notice] 2 objects detected for e-ray. Selected the closer one (index {idx}).")
                    xe1 = objects_e["x"][idx]
                    ye1 = objects_e["y"][idx]
                else:
                    assert False, f"Check the coordinates, Ne={N_obj_e} (3 or more objects detected)"

                xe1_full = xe1 + xmin_e
                ye1_full = ye1 + ymin_e

                print(f"  Aperture location after baricenter search")
                print(f"    xo0, yo0 = {xo0:.2f}, {yo0:.2f} -> xo1, yo1 = {xo1_full:.2f}, {yo1_full:.2f}")
                print(f"    xe0, ye0 = {xe0:.2f}, {ye0:.2f} -> xe1, ye1 = {xe1_full:.2f}, {ye1_full:.2f}")

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

                # Do photometry ===============================================
                if args.ann:
                    if args.ann0 is not None:
                        ann0 = args.ann0
                        ann1 = args.ann1
                    else:
                        ann_gap = args.ann_gap
                        ann_width = args.ann_width
                        ann0 = radius + ann_gap
                        ann1 = radius + ann_gap + ann_width

                    if args.ell:
                        # 楕円の場合: 短軸基準の指定を elong_ratio 倍して「長軸」に変換
                        bkgann = (ann0 * args.elong_ratio, ann1 * args.elong_ratio)
                    else:
                        # 真円の場合
                        bkgann = (ann0, ann1)
                else:
                    bkgann = None

                # 測光の実行
                if args.ell:
                    # --- 楕円測光 (sep.sum_ellipse) ---
                    # radius を短軸とし、elong_ratio倍して長軸 a を作る
                    b_ell = radius
                    a_ell = b_ell * args.elong_ratio
                    theta_val = args.theta if args.theta is not None else 0.0

                    flux_o, fluxerr_o, eflag_o = sep.sum_ellipse(
                        img_o, [xo1], [yo1], a=a_ell, b=b_ell, theta=theta_val, 
                        err=bgerr_o, gain=gain, bkgann=bkgann)
                    
                    flux_e, fluxerr_e, eflag_e = sep.sum_ellipse(
                        img_e, [xe1], [ye1], a=a_ell, b=b_ell, theta=theta_val, 
                        err=bgerr_e, gain=gain, bkgann=bkgann)
                else:
                    # --- 真円測光 (sep.sum_circle) ---
                    flux_o, fluxerr_o, eflag_o = sep.sum_circle(
                        img_o, [xo1], [yo1], r=radius, 
                        err=bgerr_o, gain=gain, bkgann=bkgann)
                    
                    flux_e, fluxerr_e, eflag_e = sep.sum_circle(
                        img_e, [xe1], [ye1], r=radius, 
                        err=bgerr_e, gain=gain, bkgann=bkgann)

                # 後処理・出力
                flux_o, fluxerr_o = float(flux_o), float(fluxerr_o)
                SNR_o = flux_o / fluxerr_o
                print(f"  xo0, yo0 = {xo0}, {yo0}")
                print(f"  flux_o, fluxerr_o, SNR_o = {flux_o:.2f}, {fluxerr_o:.2f}, {SNR_o:.1f}")

                flux_e, fluxerr_e = float(flux_e), float(fluxerr_e)
                SNR_e = flux_e / fluxerr_e
                print(f"  flux_e, fluxerr_e, SNR_e = {flux_e:.2f}, {fluxerr_e:.2f}, {SNR_e:.1f}")

                print(f"  -> Ratio e/o = {flux_e / flux_o}")
                # Do photometry ===============================================

                if idx_fi%4 == 0:
                    ang = 0
                elif idx_fi%4 == 1:
                    ang = 45
                elif idx_fi%4 == 2:
                    ang = 22.5
                elif idx_fi%4 == 3:
                    ang = 67.5

                info[f"flux_o"]    = flux_o
                info[f"fluxerr_o"] = fluxerr_o
                info[f"flux_e"]    = flux_e
                info[f"fluxerr_e"] = fluxerr_e
                info["angle"] = f"{int(ang*10):04d}"
                info["insrot"] = insrot
                date = hdr[key_date]
                utc0 = hdr[key_ut]
                utc0 = f"{date}T{utc0}"
                utc0_dt = datetime.datetime.strptime(utc0, "%Y-%m-%dT%H:%M:%S.%f")
                utcmid_dt = utc0_dt + datetime.timedelta(seconds=texp)
                utcmid = datetime.datetime.strftime(utcmid_dt, "%Y-%m-%dT%H:%M:%S.%f")
                info["utc"] = utcmid
                info["fits"] = fi
                df_res = pd.DataFrame(info.values(), index=info.keys()).T
                df_res_list.append(df_res)

                # Plot photometry region ======================================
                if args.photmap:
                    import matplotlib.pyplot as plt
                    from matplotlib.collections import PatchCollection
                    from matplotlib.patches import Circle, Ellipse  
                    from scipy.stats import sigmaclip

                    out = os.path.join(photmapdir, f"{fi}_photmap_rad{radius}.png")
                    label_o = (
                        f"{args.obj} o-ray (xo, yo)=({xo1_full:.1f}, {yo1_full:.1f})\n"
                        f"flux={flux_o:.1f}+-{fluxerr_o:.1f} (S/N={SNR_o:.1f})")
                    label_e = (
                        f"{args.obj} e-ray (xe, ye)=({xe1_full:.1f}, {ye1_full:.1f})\n"
                        f"flux={flux_e:.1f}+-{fluxerr_e:.1f} (S/N={SNR_e:.1f})")

                    color_o, color_e = mycolor[0], mycolor[1]
                    ls = "solid"
                    sigma = 3
                    lw_aperture = 2.0

                    if args.ell:
                        r_min = radius
                        r_maj = radius * args.elong_ratio

                        ann_in_min = args.ann0 if args.ann0 else radius + args.ann_gap
                        ann_in_maj = ann_in_min * args.elong_ratio

                        ann_out_min = args.ann1 if args.ann1 else ann_in_min + args.ann_width
                        ann_out_maj = ann_out_min * args.elong_ratio

                        theta_deg = np.degrees(theta) if args.theta is not None else 0.0

                        cos_t, sin_t = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
                        dx_r = np.sqrt((r_maj * cos_t)**2 + (r_min * sin_t)**2)
                        dy_r = np.sqrt((r_maj * sin_t)**2 + (r_min * cos_t)**2)
                        dx_in = np.sqrt((ann_in_maj * cos_t)**2 + (ann_in_min * sin_t)**2)
                        dy_in = np.sqrt((ann_in_maj * sin_t)**2 + (ann_in_min * cos_t)**2)
                        dx_out = np.sqrt((ann_out_maj * cos_t)**2 + (ann_out_min * sin_t)**2)
                        dy_out = np.sqrt((ann_out_maj * sin_t)**2 + (ann_out_min * cos_t)**2)
                    else:
                        ann_in = args.ann0 if args.ann0 else radius + args.ann_gap
                        ann_out = args.ann1 if args.ann1 else radius + args.ann_gap + args.ann_width
                        dx_r = dy_r = radius
                        dx_in = dy_in = ann_in
                        dx_out = dy_out = ann_out

                    fig = plt.figure(figsize=(16, 8))

                    # =========================================================
                    # 1. Ordinary (o-ray)
                    # =========================================================
                    ax_img_o   = fig.add_axes([0.06, 0.12, 0.34, 0.65])
                    ax_top_o   = fig.add_axes([0.06, 0.78, 0.34, 0.10], sharex=ax_img_o)
                    ax_right_o = fig.add_axes([0.41, 0.12, 0.05, 0.65], sharey=ax_img_o)

                    # --- Main Image (o-ray) ---
                    _, vmin_o, vmax_o = sigmaclip(img_o, sigma, sigma)
                    ax_img_o.imshow(img_o, cmap='gray_r', vmin=vmin_o, vmax=vmax_o)
                    ax_img_o.scatter(xo1, yo1, color=color_o, s=150, lw=lw_aperture, marker="x", alpha=1, label=label_o)

                    # パッチ表示の切り替え
                    if args.ell:
                        # 1. 内側のアパーチャ (実線)
                        ax_img_o.add_patch(Ellipse((xo1[0], yo1[0]), 2*r_maj, 2*r_min, angle=theta_deg, ec=color_o, ls="solid", lw=lw_aperture, fc="None"))

                        if args.ann:
                            # 2. 内側のアニュラス境界 (破線)
                            ax_img_o.add_patch(Ellipse((xo1[0], yo1[0]), 2*ann_in_maj, 2*ann_in_min, angle=theta_deg, ec=color_o, ls="dashed", lw=lw_aperture, fc="None"))
                            # 3. 外側のアニュラス境界 (破線)
                            ax_img_o.add_patch(Ellipse((xo1[0], yo1[0]), 2*ann_out_maj, 2*ann_out_min, angle=theta_deg, ec=color_o, ls="dashed", lw=lw_aperture, fc="None"))
                    else:
                        ax_img_o.add_patch(Circle((xo1[0], yo1[0]), radius, ec=color_o, ls="solid", lw=lw_aperture, fc="None"))
                        if args.ann:
                            ax_img_o.add_patch(Circle((xo1[0], yo1[0]), ann_in, ec=color_o, ls="dashed", lw=lw_aperture, fc="None"))
                            ax_img_o.add_patch(Circle((xo1[0], yo1[0]), ann_out, ec=color_o, ls="dashed", lw=lw_aperture, fc="None"))


                    ax_img_o.set_xlabel("x [pix]")
                    ax_img_o.set_ylabel("y [pix]")
                    ax_img_o.set_xlim([0, img_o.shape[1]])
                    ax_img_o.set_ylim([0, img_o.shape[0]])
                    ax_img_o.invert_yaxis()
                    ax_img_o.legend(loc="upper left").get_frame().set_alpha(0.8)
                    ax_top_o.set_title(f"Ordinary (o-ray)")

                    # --- Top Profile (o-ray) ---
                    xo1_val, yo1_val = xo1[0], yo1[0]
                    row_idx_o = np.arange(int(yo1_val)-1, int(yo1_val)+2)
                    row_idx_o = np.clip(row_idx_o, 0, img_o.shape[0]-1)
                    row_prof_o = img_o[row_idx_o, :].mean(axis=0)
                    ax_top_o.plot(np.arange(img_o.shape[1]), row_prof_o, color='gray', lw=1.5)
                    ax_top_o.set_ylabel("Counts")
                    ax_top_o.tick_params(labelbottom=False)

                    ax_top_o.vlines([xo1_val - dx_r, xo1_val + dx_r], ymin=row_prof_o.min(), ymax=row_prof_o.max(), color=color_o, linestyle=ls, lw=lw_aperture)
                    if args.ann:
                        ax_top_o.vlines([xo1_val - dx_in, xo1_val + dx_in, xo1_val - dx_out, xo1_val + dx_out], ymin=row_prof_o.min(), ymax=row_prof_o.max(), color=color_o, linestyle='dashed', lw=lw_aperture)

                    # --- Right Profile (o-ray) ---
                    col_idx_o = np.arange(int(xo1_val)-1, int(xo1_val)+2)
                    col_idx_o = np.clip(col_idx_o, 0, img_o.shape[1]-1)
                    col_prof_o = img_o[:, col_idx_o].mean(axis=1)
                    ax_right_o.plot(col_prof_o, np.arange(img_o.shape[0]), color='gray', lw=1.5)
                    ax_right_o.set_xlabel("Counts")
                    ax_right_o.tick_params(labelleft=False)
                    ax_right_o.invert_yaxis()

                    ax_right_o.hlines([yo1_val - dy_r, yo1_val + dy_r], xmin=col_prof_o.min(), xmax=col_prof_o.max(), color=color_o, linestyle=ls, lw=lw_aperture)
                    if args.ann:
                        ax_right_o.hlines([yo1_val - dy_in, yo1_val + dy_in, yo1_val - dy_out, yo1_val + dy_out], xmin=col_prof_o.min(), xmax=col_prof_o.max(), color=color_o, linestyle='dashed', lw=lw_aperture)

                    # =========================================================
                    # 2. Extra-ordinary (e-ray)
                    # =========================================================
                    ax_img_e   = fig.add_axes([0.54, 0.12, 0.34, 0.65])
                    ax_top_e   = fig.add_axes([0.54, 0.78, 0.34, 0.10], sharex=ax_img_e)
                    ax_right_e = fig.add_axes([0.89, 0.12, 0.05, 0.65], sharey=ax_img_e)

                    # --- Main Image (e-ray) ---
                    _, vmin_e, vmax_e = sigmaclip(img_e, sigma, sigma)
                    ax_img_e.imshow(img_e, cmap='gray_r', vmin=vmin_e, vmax=vmax_e)
                    ax_img_e.scatter(xe1, ye1, color=color_e, s=150, lw=lw_aperture, marker="x", alpha=1, label=label_e)


                    # パッチ表示の切り替え
                    if args.ell:
                        # 1. 内側のアパーチャ (実線)
                        ax_img_e.add_patch(Ellipse((xe1[0], ye1[0]), 2*r_maj, 2*r_min, angle=theta_deg, ec=color_e, ls="solid", lw=lw_aperture, fc="None"))
                        
                        if args.ann:
                            # 2. 内側のアニュラス境界 (破線)
                            ax_img_e.add_patch(Ellipse((xe1[0], ye1[0]), 2*ann_in_maj, 2*ann_in_min, angle=theta_deg, ec=color_e, ls="dashed", lw=lw_aperture, fc="None"))
                            # 3. 外側のアニュラス境界 (破線)
                            ax_img_e.add_patch(Ellipse((xe1[0], ye1[0]), 2*ann_out_maj, 2*ann_out_min, angle=theta_deg, ec=color_e, ls="dashed", lw=lw_aperture, fc="None"))
                    else:
                        ax_img_e.add_patch(Circle((xe1[0], ye1[0]), radius, ec=color_e, ls="solid", lw=lw_aperture, fc="None"))
                        if args.ann:
                            ax_img_e.add_patch(Circle((xe1[0], ye1[0]), ann_in, ec=color_e, ls="dashed", lw=lw_aperture, fc="None"))
                            ax_img_e.add_patch(Circle((xe1[0], ye1[0]), ann_out, ec=color_e, ls="dashed", lw=lw_aperture, fc="None"))

                    ax_img_e.set_xlabel("x [pix]")
                    ax_img_e.set_xlim([0, img_e.shape[1]])
                    ax_img_e.set_ylim([0, img_e.shape[0]])
                    ax_img_e.invert_yaxis()
                    ax_img_e.legend(loc="upper left").get_frame().set_alpha(0.8)
                    ax_top_e.set_title(f"Extra-ordinary (e-ray)")

                    # --- Top Profile (e-ray) ---
                    xe1_val, ye1_val = xe1[0], ye1[0]
                    row_idx_e = np.arange(int(ye1_val)-1, int(ye1_val)+2)
                    row_idx_e = np.clip(row_idx_e, 0, img_e.shape[0]-1)
                    row_prof_e = img_e[row_idx_e, :].mean(axis=0)
                    ax_top_e.plot(np.arange(img_e.shape[1]), row_prof_e, color='gray', lw=1.5)
                    ax_top_e.set_ylabel("Counts")
                    ax_top_e.tick_params(labelbottom=False)

                    ax_top_e.vlines([xe1_val - dx_r, xe1_val + dx_r], ymin=row_prof_e.min(), ymax=row_prof_e.max(), color=color_e, linestyle=ls, lw=lw_aperture)
                    if args.ann:
                        ax_top_e.vlines([xe1_val - dx_in, xe1_val + dx_in, xe1_val - dx_out, xe1_val + dx_out], ymin=row_prof_e.min(), ymax=row_prof_e.max(), color=color_e, linestyle='dashed', lw=lw_aperture)

                    # --- Right Profile (e-ray) ---
                    col_idx_e = np.arange(int(xe1_val)-1, int(xe1_val)+2)
                    col_idx_e = np.clip(col_idx_e, 0, img_e.shape[1]-1)
                    col_prof_e = img_e[:, col_idx_e].mean(axis=1)
                    ax_right_e.plot(col_prof_e, np.arange(img_e.shape[0]), color='gray', lw=1.5)
                    ax_right_e.set_xlabel("Counts")
                    ax_right_e.tick_params(labelleft=False)
                    ax_right_e.invert_yaxis()

                    ax_right_e.hlines([ye1_val - dy_r, ye1_val + dy_r], xmin=col_prof_e.min(), xmax=col_prof_e.max(), color=color_e, linestyle=ls, lw=lw_aperture)
                    if args.ann:
                        ax_right_e.hlines([ye1_val - dy_in, ye1_val + dy_in, ye1_val - dy_out, ye1_val + dy_out], xmin=col_prof_e.min(), xmax=col_prof_e.max(), color=color_e, linestyle='dashed', lw=lw_aperture)

                    main_title = f"{args.obj} ({band}-band), {fi}, {utcmid}"
                    plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.96)

                    plt.savefig(out, dpi=200)
                    plt.close()

            df_res = pd.concat(df_res_list, axis=0)
            df_res = df_res.reset_index()

            # Save flux
            f_000_o    = df_res[df_res["angle"]=="0000"].flux_o.values.tolist()[0]
            ferr_000_o = df_res[df_res["angle"]=="0000"].fluxerr_o.values.tolist()[0]
            f_000_e    = df_res[df_res["angle"]=="0000"].flux_e.values.tolist()[0]
            ferr_000_e = df_res[df_res["angle"]=="0000"].fluxerr_e.values.tolist()[0]
            f_450_o    = df_res[df_res["angle"]=="0450"].flux_o.values.tolist()[0]
            ferr_450_o = df_res[df_res["angle"]=="0450"].fluxerr_o.values.tolist()[0]
            f_450_e    = df_res[df_res["angle"]=="0450"].flux_e.values.tolist()[0]
            ferr_450_e = df_res[df_res["angle"]=="0450"].fluxerr_e.values.tolist()[0]
            f_225_o    = df_res[df_res["angle"]=="0225"].flux_o.values.tolist()[0]
            ferr_225_o = df_res[df_res["angle"]=="0225"].fluxerr_o.values.tolist()[0]
            f_225_e    = df_res[df_res["angle"]=="0225"].flux_e.values.tolist()[0]
            ferr_225_e = df_res[df_res["angle"]=="0225"].fluxerr_e.values.tolist()[0]
            f_675_o    = df_res[df_res["angle"]=="0675"].flux_o.values.tolist()[0]
            ferr_675_o = df_res[df_res["angle"]=="0675"].fluxerr_o.values.tolist()[0]
            f_675_e    = df_res[df_res["angle"]=="0675"].flux_e.values.tolist()[0]
            ferr_675_e = df_res[df_res["angle"]=="0675"].fluxerr_e.values.tolist()[0]

            flux_000_o_list.append(f_000_o)
            fluxerr_000_o_list.append(ferr_000_o)
            flux_000_e_list.append(f_000_e)
            fluxerr_000_e_list.append(ferr_000_e)
            flux_450_o_list.append(f_450_o)
            fluxerr_450_o_list.append(ferr_450_o)
            flux_450_e_list.append(f_450_e)
            fluxerr_450_e_list.append(ferr_450_e)
            flux_225_o_list.append(f_225_o)
            fluxerr_225_o_list.append(ferr_225_o)
            flux_225_e_list.append(f_225_e)
            fluxerr_225_e_list.append(ferr_225_e)
            flux_675_o_list.append(f_675_o)
            fluxerr_675_o_list.append(ferr_675_o)
            flux_675_e_list.append(f_675_e)
            fluxerr_675_e_list.append(ferr_675_e)

            u, uerr, q, qerr  = polana_4angle(df_res, inst)

            u_list.append(u)
            uerr_list.append(uerr)
            q_list.append(q)
            qerr_list.append(qerr)

            insrot000 = df_res[df_res["angle"]=="0000"].insrot.values.tolist()[0]
            insrot450 = df_res[df_res["angle"]=="0450"].insrot.values.tolist()[0]
            insrot1   = (insrot000 + insrot450)*0.5
            insrot1_list.append(insrot1)
            insrot225 = df_res[df_res["angle"]=="0225"].insrot.values.tolist()[0]
            insrot675 = df_res[df_res["angle"]=="0675"].insrot.values.tolist()[0]
            insrot2   = (insrot225 + insrot675)*0.5
            insrot2_list.append(insrot2)

            texp_list.append(texp)
            instpa_list.append(instpa)

            utc000 = df_res[df_res["angle"]=="0000"].utc.values.tolist()[0]
            utc450 = df_res[df_res["angle"]=="0450"].utc.values.tolist()[0]
            utc225 = df_res[df_res["angle"]=="0225"].utc.values.tolist()[0]
            utc675 = df_res[df_res["angle"]=="0675"].utc.values.tolist()[0]
            utc000_list.append(utc000)
            utc450_list.append(utc450)
            utc225_list.append(utc225)
            utc675_list.append(utc675)

            fi000 = df_res[df_res["angle"]=="0000"].fits.values.tolist()[0]
            fi450 = df_res[df_res["angle"]=="0450"].fits.values.tolist()[0]
            fi225 = df_res[df_res["angle"]=="0225"].fits.values.tolist()[0]
            fi675 = df_res[df_res["angle"]=="0675"].fits.values.tolist()[0]
            fi000_list.append(fi000)
            fi450_list.append(fi450)
            fi225_list.append(fi225)
            fi675_list.append(fi675)

            if args.mp:
                ut = df_res.at[0, "utc"]
                alpha, phi = utc2alphaphi(args.obj, ut, loc_Subaru)
                alpha_list.append(alpha)
                phi_list.append(phi)

        if args.out:
            out = args.out
        else:
            out = "polres_FOCAS.txt"
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
                flux_000_o = flux_000_o_list,
                fluxerr_000_o = fluxerr_000_o_list,
                flux_000_e = flux_000_e_list,
                fluxerr_000_e = fluxerr_000_e_list,
                flux_450_o = flux_450_o_list,
                fluxerr_450_o = fluxerr_450_o_list,
                flux_450_e = flux_450_e_list,
                fluxerr_450_e = fluxerr_450_e_list,
                flux_225_o = flux_450_o_list,
                fluxerr_225_o = fluxerr_225_o_list,
                flux_225_e = flux_225_e_list,
                fluxerr_225_e = fluxerr_225_e_list,
                flux_675_o = flux_675_o_list,
                fluxerr_675_o = fluxerr_675_o_list,
                flux_675_e = flux_675_e_list,
                fluxerr_675_e = fluxerr_675_e_list,
                u=u_list, uerr=uerr_list,
                q=q_list, qerr=qerr_list,
                insrot1=insrot1_list, 
                insrot2=insrot2_list, 
                instpa=instpa_list, 
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
                instpa=instpa_list, 
            ))

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

if __name__ == "__main__":
    main()

