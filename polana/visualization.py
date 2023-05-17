#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Color photmetry plot functions
"""
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from polana.util import round_error

# Color
mycolor = ["#AD002D", "#1e50a2", "#006e54", "#ffd900", 
           "#EFAEA1", "#69821b", "#ec6800", "#afafb0", "#0095b9", "#89c3eb"] 
mycolor = mycolor*100

# Linestyle
myls = ["solid", "dashed", "dashdot", "dotted", (0, (5, 3, 1, 3, 1, 3)), 
        (0, (4,2,1,2,1,2,1,2))]
myls = myls*100

# Marker
mymark = ["o", "^", "x", "D", "+", "v", "<", ">", "h", "H"]
mymark = mymark*100


def colmark(idx):
    """
    Return color and marker from idx.
    """
    color = [
        "#AD002D", "#1e50a2", "#006e54", "#ffd900", 
        "#EFAEA1", "#69821b", "#ec6800", "#afafb0", "#0095b9", "#89c3eb"] 
    marker = [
        "o", "^", "x", "D", "+", "v", "<", ">", "h", "H"]
    N_total = len(color)*len(marker)
    assert idx < N_total, "Check the code."

    idx_color = idx%len(color)
    idx_marker = idx//len(marker)
    return color[idx_color], marker[idx_marker]


def plot_obspolres(
    ax, df, key, obj, key_P="P", key_Perr="Perr", 
    key_theta="theta", key_thetaerr="thetaerr", 
    key_q="q", key_qerr="qerr", key_u="u", key_uerr="uerr",
    color="black", marker="o", ls="solid"):
    """
    Plot observational result of polarymetry.
    """

    # Weighted mean
    w_P = [1/x**2 for x in df[key_Perr]]
    print(df[key_thetaerr])
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
    print(label)

    ax.errorbar(
        df[key_q], df[key_u], xerr=df[key_qerr], yerr=df[key_uerr],
        ms=15, color=color, marker=" ", capsize=0, 
        ls="None", label=None, zorder=1)
    ax.scatter(
        df[key_q], df[key_u], marker=marker, s=300, color=color, 
        facecolor="None", zorder=1, label=label)

    # Add circle of P=Pobs
    ax.add_collection(PatchCollection(
        [Circle((0, 0), float(wmean_P))],
        color=color, ls=ls, lw=1, facecolor="None")
        )


def plot_litpolres(
    ax, key, obj, P, Perr, theta, thetaerr, color="black", marker="o", ls="solid"):
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
