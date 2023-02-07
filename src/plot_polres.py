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


if __name__ == "__main__":
    parser = ap(description="Plot P and theta.")
    parser.add_argument(
        "res", type=str, help="*colall*.csv")
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
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.80])
    ax.set_xlabel(f"q = Q/I")
    ax.set_ylabel(f"u = U/I")

    # Observation
    # Weighted mean
    w_P = [1/x**2 for x in df["Perr"]]
    w_theta = [1/x**2 for x in df["thetaerr"]]
    wstd_P = np.sqrt(1/np.sum(w_P))
    wstd_theta = np.sqrt(1/np.sum(w_theta))
    wmean_P = np.average(df["P"], weights=w_P)
    wmean_theta = np.average(df["theta"], weights=w_theta)

    # Convert the domain of definition
    if wmean_theta < 0:
        # 2 theta = 2 theta + 2 pi
        wmean_theta = wmean_theta + np.pi

    # in percent 
    wmean_P_percent, wstd_Perr_percent = 100.*wmean_P, 100.*wstd_P
    P_percent, Perr_percent = round_error(wmean_P_percent, wstd_Perr_percent)


    # in degree TODO:check
    wmean_theta, wstd_theta = np.rad2deg(wmean_theta), 180./np.pi*wstd_theta
    theta, thetaerr = round_error(wmean_theta, wstd_theta)

    label = (
        f"{obj} {args.key_obs}\n" + r"(P, $\theta$) = " 
        + f"({P_percent}" r"$\pm$" + f"{Perr_percent} %, {theta}" + r"$\pm$" + f"{thetaerr} deg)")

    ax.errorbar(
        df["q"], df["u"], xerr=df["qerr"], yerr=df["uerr"],
        ms=15, color=mycolor[0], marker=" ", capsize=0, 
        ls="None", label=None, zorder=1)
    ax.scatter(
        df["q"], df["u"], marker=mymark[0], s=100, color=mycolor[0], 
        facecolor="None", zorder=1, label=label)

    # Add circle pf P=Pobs
    ax.add_collection(PatchCollection(
        [Circle((0, 0), float(wmean_P))],
        color=mycolor[0], ls="solid", lw=1, facecolor="None")
        )

    width = 2*np.max([
        abs(np.min(df["q"])), abs(np.max(df["q"])),
        abs(np.min(df["u"])), abs(np.max(df["u"]))
        ])

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
        
        # Input theta is in degree
        label = (
            f"{obj} {args.key}\n" + r"(P, $\theta$) = " 
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
            ms=15, color=mycolor[1], marker=" ", capsize=0, 
            ls="None", label=None, zorder=1)
        ax.scatter(
            q, u, marker=mymark[1], s=100, color=mycolor[1], 
            facecolor="None", zorder=1, label=label)
        width = 2*np.max([abs(q), abs(u)])
        # Add circle pf P
        ax.add_collection(PatchCollection(
            [Circle((0, 0), float(P))],
            color=mycolor[1], ls="dashed", lw=1, facecolor="None")
            )

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
