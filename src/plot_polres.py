#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot derived linear polarization degree and position angle 
on U/I-Q/I plane.

In input data (i.e., output of pol_XXX.py),
-pi < theta < pi 
is the range.
In this script, domain of definition is changed to be
0 < theta < 2pi.


The corrections are done in the following order 
(same with Ishiguro+2017 and Geem+2022b, not Kawakami+2021):
    1. Correction of polarization efficiency with 'p_eff'.
    2. Correction of instrumental polarization with 'q_inst' and 'u_inst'.
    3. Correction of position angle offset with 'pa_offset'.

The order of correction is 1. -> 3. -> 2. in Kawakami+2021.
The important thing is not the order itself, but the derived P and theta
are close to those in the literature.
"""
import os 
from argparse import ArgumentParser as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from calcerror import round_error
from polana.visualization import mycolor, myls, mymark
from polana.util import cor_poleff, cor_instpol, cor_paoffset, calc_Ptheta


def plot_obspolres(
    ax, df, key, key_P="P", key_Perr="Perr", 
    key_theta="theta", key_thetaerr="thetaerr", 
    key_q="q", key_qerr="qerr", key_u="u", key_uerr="uerr",
    color="black", marker="o", ls="solid"):
    """
    Plot observational result of polarymetry.
    """
    # Weighted mean
    w_P = [1/x**2 for x in df[key_Perr]]
    w_theta = [1/x**2 for x in df[key_thetaerr]]
    wstd_P = np.sqrt(1/np.sum(w_P))
    wstd_theta = np.sqrt(1/np.sum(w_theta))
    wmean_P = np.average(df[key_P], weights=w_P)
    wmean_theta = np.average(df[key_theta], weights=w_theta)

    # Convert the domain of definition
    if wmean_theta < 0:
        # 2 theta = 2 theta + 2 pi
        wmean_theta = wmean_theta + np.pi

    # in percent 
    wmean_P_percent, wstd_Perr_percent = 100.*wmean_P, 100.*wstd_P
    P_percent, Perr_percent = round_error(wmean_P_percent, wstd_Perr_percent)


    # in degree TODO:check
    wmean_theta, wstd_theta = np.rad2deg(wmean_theta), np.rad2deg(wstd_theta)
    theta, thetaerr = round_error(wmean_theta, wstd_theta)


    label = (
        f"{obj} {key}\n" + r"(P, $\theta$) = " 
        + f"({P_percent}" r"$\pm$" + f"{Perr_percent} %, {theta}" + r"$\pm$" + f"{thetaerr} deg)")

    ax.errorbar(
        df[key_q], df[key_u], xerr=df[key_qerr], yerr=df[key_uerr],
        ms=15, color=color, marker=" ", capsize=0, 
        ls="None", label=None, zorder=1)
    ax.scatter(
        df[key_q], df[key_u], marker=marker, s=300, color=color, 
        facecolor="None", zorder=1, label=label)

    # Add circle pf P=Pobs
    ax.add_collection(PatchCollection(
        [Circle((0, 0), float(wmean_P))],
        color=color, ls=ls, lw=1, facecolor="None")
        )


def plot_litpolres(
    ax, key, P, Perr, theta, thetaerr, color="black", marker="o", ls="solid"):
    """
    Plot observational result of polarymetry.
    """
    # Input theta is in degree
    label = (
        f"{obj} {key}\n" + r"(P, $\theta$) = " 
        + f"({P}" r"$\pm$" + f"{Perr} %, {theta}" + r"$\pm$" + f"{thetaerr} deg)")

    # Convert to radians
    theta = theta*np.pi/180.
    thetaerr = thetaerr*np.pi/180.
    # Convert to P < 1
    P = P*0.01
    Perr = Perr*0.01

    q = P*np.cos(2*theta)
    #np.cos(2*theta)**2*Perr**2 
    qerr = np.sqrt(
        np.cos(2*theta)**2*Perr**2 + 4*P**2*np.sin(2*theta)**2*thetaerr**2
        )
    u = P*np.sin(2*theta)
    uerr = np.sqrt(
        np.sin(2*theta)**2*Perr**2 + 4*P**2*np.cos(2*theta)**2*thetaerr**2
        )

    ax.errorbar(
        q, u, xerr=qerr, yerr=uerr,
        ms=15, color=color, marker=" ", capsize=0, 
        ls="None", label=None, zorder=1)
    ax.scatter(
        q, u, marker=marker, s=300, color=color, 
        facecolor="None", zorder=1, label=label)
    width = 2*np.max([abs(q), abs(u)])
    # Add circle pf P
    ax.add_collection(PatchCollection(
        [Circle((0, 0), float(P))],
        color=color, ls=ls, lw=1, facecolor="None")
        )


if __name__ == "__main__":
    parser = ap(description="Plot P and theta.")
    parser.add_argument(
        "res", type=str, help="*colall*.csv")
    parser.add_argument(
        "--cor", action="store_true", 
        help="Correct instrument polarization etc.")
    parser.add_argument(
        "--key_obs", type=str, default="Obs.",
        help="Keyword for expected vale")
    parser.add_argument(
        "-P", type=float, default=None,
        help="Expected linear polarization degree")
    parser.add_argument(
        "--Perr", type=float, default=None,
        help="Uncertainty of expected linear polarization degree")
    parser.add_argument(
        "--theta", type=float, default=None,
        help="Expected position angle of linear polarization degree")
    parser.add_argument(
        "--thetaerr", type=float, default=None,
        help="Uncertainty of the position angle")
    parser.add_argument(
        "--key", type=str, default="",
        help="Keyword for expected vale")
    parser.add_argument(
        "--outdir", type=str, default="plot",
        help="Output directory")
    parser.add_argument(
        "--out", type=str, 
        help="Output filename")
    args = parser.parse_args()


    # Set output directory
    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    df = pd.read_csv(args.res, sep=" ")
    print(f"  N_obs = {len(df)} (original)")
    obj = df.obj[0]


    fig = plt.figure(figsize=(8, 8)) 
    ax = fig.add_axes([0.20, 0.20, 0.78, 0.78])
    ax.set_xlabel(f"q = Q/I")
    ax.set_ylabel(f"u = U/I")


    # Observation (raw) =======================================================
    plot_obspolres(
        ax, df, args.key_obs, color=mycolor[0], marker=mymark[0], ls="dashed")
    width = 2*np.max([
        abs(np.min(df["q"])), abs(np.max(df["q"])),
        abs(np.min(df["u"])), abs(np.max(df["u"]))
        ])
    # Observation (raw) =======================================================


    # Observation (after correction) ==========================================
    # Plot linear polarization degree after correction
    if args.cor:
        inst = df.inst[0]
        band = df.band[0]

        # 1. Correction of polarization efficiency with 'p_eff'.
        df = cor_poleff(
            df, inst, band, "q", "u", "qerr", "uerr", "q_cor0", "u_cor0", 
            "qerr_cor0", "uerr_cor0")
        df = calc_Ptheta(
            df, "P_cor0", "theta_cor0", "Perr_cor0", "thetaerr_cor0",
            "q_cor0", "u_cor0", "qerr_cor0", "uerr_cor0")
        plot_obspolres(
            ax, df, args.key_obs+"peff corrected", key_P="P_cor0", key_Perr="Perr_cor0",
            key_theta="theta_cor0", key_thetaerr="thetaerr_cor0",
            key_q="q_cor0", key_qerr="qerr_cor0", key_u="u_cor0", key_uerr="uerr_cor0",
            color=mycolor[3], marker=mymark[5], ls="solid")

        # 2. Correction of instrumental polarization with 'q_inst' and 'u_inst'.
        df = cor_instpol(
            df, inst, band, "q_cor0", "u_cor0", "qerr_cor0", "uerr_cor0", 
            "q_cor1", "u_cor1", "qerr_cor1", "uerr_cor1", "insrot")
        df = calc_Ptheta(
            df, "P_cor1", "theta_cor1", "Perr_cor1", "thetaerr_cor1",
            "q_cor1", "u_cor1", "qerr_cor1", "uerr_cor1")
        plot_obspolres(
            ax, df, args.key_obs+"inst corrected", key_P="P_cor1", key_Perr="Perr_cor1",
            key_theta="theta_cor1", key_thetaerr="thetaerr_cor1",
            key_q="q_cor1", key_qerr="qerr_cor1", key_u="u_cor1", key_uerr="uerr_cor1",
            color=mycolor[4], marker=mymark[6], ls="solid")

        # 3. Correction of position angle offset with 'pa_offset'.
        df = cor_paoffset(
            df, inst, band, "q_cor1", "u_cor1", "qerr_cor1", "uerr_cor1", 
            "q_cor2", "u_cor2", "qerr_cor2", "uerr_cor2", "instpa")
        df = calc_Ptheta(
            df, "P_cor2", "theta_cor2", "Perr_cor2", "thetaerr_cor2",
            "q_cor2", "u_cor2", "qerr_cor2", "uerr_cor2")
        plot_obspolres(
            ax, df, args.key_obs+"paoff corrected", key_P="P_cor2", key_Perr="Perr_cor2",
            key_theta="theta_cor2", key_thetaerr="thetaerr_cor2",
            key_q="q_cor2", key_qerr="qerr_cor2", key_u="u_cor2", key_uerr="uerr_cor2",
            color=mycolor[2], marker=mymark[3], ls="solid")
    # Observation (after correction) ==========================================




    # Plot expected linear polarization degree
    if (args.P) and (args.theta):
        if not args.Perr:
            Perr = 0.
        if not args.thetaerr:
            thetaerr = 0.

        P = args.P
        Perr = args.Perr
        theta = args.theta
        thetaerr = args.thetaerr

        plot_litpolres(
            ax, "Literature", P, Perr, theta, thetaerr, 
            color=mycolor[1], marker=mymark[1], ls="dotted")
        

    ax.grid(which="major", axis="both")
    # Set range
    ax.set_xlim([-width, width])
    ax.set_ylim([-width, width])

    ax.legend(fontsize=12)


    if args.out:
        out = args.out
    else:
        out = f"{obj}_polana_res.png"
    out = os.path.join(args.outdir, out)
    fig.savefig(out)
    plt.close()
