#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Color photmetry plot functions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from scipy.stats import sigmaclip
from scipy.signal import medfilt
from matplotlib import cm

from calcerror import adderr, round_error

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


def color_from_band(band):
    if band == "g": return "#69821b"
    if band == "r": return "#AD002D" 
    if band == "i": return "#ec6800"
    if band == "z": return "#f055f0"

#   def myfigure_objmag(n_band):
#     if n_band == 3:
#       # each band has
#       # 1. cat_col vs. cat - inst with CT, 
#       # 2. CT and Z light curve
#       # 3. cat_mag vs. mag_inst
#       # 4. CT ignored magzpt light curve
#       # 5. magzpt histogram without fitting
#       # 6. magpzt from histogram
#       fig = plt.figure(figsize=(24, 12 ))
#       ax1_CTfit = fig.add_axes([0.10, 0.70, 0.1, 0.23])
#       ax2_CTfit = fig.add_axes([0.10, 0.40, 0.1, 0.23])
#       ax3_CTfit = fig.add_axes([0.10, 0.10, 0.1, 0.23])
#       ax1_CTfit_Z = fig.add_axes([0.25, 0.70, 0.1, 0.23])
#       ax2_CTfit_Z = fig.add_axes([0.25, 0.40, 0.1, 0.23])
#       ax3_CTfit_Z = fig.add_axes([0.25, 0.10, 0.1, 0.23])
#       ax1_fit = fig.add_axes([0.40, 0.70, 0.1, 0.23])
#       ax2_fit = fig.add_axes([0.40, 0.40, 0.1, 0.23])
#       ax3_fit = fig.add_axes([0.40, 0.10, 0.1, 0.23])
#       ax1_fit_Z = fig.add_axes([0.55, 0.70, 0.1, 0.23])
#       ax2_fit_Z = fig.add_axes([0.55, 0.40, 0.1, 0.23])
#       ax3_fit_Z = fig.add_axes([0.55, 0.10, 0.1, 0.23])
#       ax1_hist = fig.add_axes([0.7, 0.70, 0.1, 0.23])
#       ax2_hist = fig.add_axes([0.7, 0.40, 0.1, 0.23])
#       ax3_hist = fig.add_axes([0.7, 0.10, 0.1, 0.23])
#       ax1_hist_Z = fig.add_axes([0.85, 0.70, 0.1, 0.23])
#       ax2_hist_Z = fig.add_axes([0.85, 0.40, 0.1, 0.23])
#       ax3_hist_Z = fig.add_axes([0.85, 0.10, 0.1, 0.23])
#   
#       axes_CTfit = [ax1_CTfit, ax2_CTfit, ax3_CTfit]
#       axes_CTfit_Z = [ax1_CTfit_Z, ax2_CTfit_Z, ax3_CTfit_Z]
#       axes_fit = [ax1_fit, ax2_fit, ax3_fit]
#       axes_fit_Z = [ax1_fit_Z, ax2_fit_Z, ax3_fit_Z]
#       axes_hist = [ax1_hist, ax2_hist, ax3_hist]
#       axes_hist_Z = [ax1_hist_Z, ax2_hist_Z, ax3_hist_Z]
#       return fig, axes_CTfit, axes_CTfit_Z, axes_fit, axes_fit_Z, axes_hist, axes_hist_Z


def myfigure4CT(n_band, n_col):
  """
  with cbar axis for CT !
  """
  if n_band == 3:
    if n_col ==3:
        fig = plt.figure(figsize=(24, 14))
        ax1_l = fig.add_axes([0.06,  0.70, 0.2, 0.23])
        ax2_l = fig.add_axes([0.06,  0.40, 0.2, 0.23])
        ax3_l = fig.add_axes([0.06,  0.10, 0.2, 0.23])
        ax1_c = fig.add_axes([0.34,   0.70, 0.2, 0.23])
        ax2_c = fig.add_axes([0.34,   0.40, 0.2, 0.23])
        ax3_c = fig.add_axes([0.34,   0.10, 0.2, 0.23])
        ax1_cbar = fig.add_axes([0.55,   0.70, 0.01, 0.23])
        ax2_cbar = fig.add_axes([0.55,   0.40, 0.01, 0.23])
        ax3_cbar = fig.add_axes([0.55,   0.10, 0.01, 0.23])
        ax1_r = fig.add_axes([0.66,  0.70, 0.2, 0.23])
        ax2_r = fig.add_axes([0.66,  0.40, 0.2, 0.23])
        ax3_r = fig.add_axes([0.66,  0.10, 0.2, 0.23])
        ax1_res = fig.add_axes([0.9, 0.70, 0.08, 0.23])
        ax2_res = fig.add_axes([0.9, 0.40, 0.08, 0.23])
        ax3_res = fig.add_axes([0.9, 0.10, 0.08, 0.23])
        axes_l = [ax1_l, ax2_l, ax3_l]
        axes_c = [ax1_c, ax2_c, ax3_c]
        axes_cbar = [ax1_cbar, ax2_cbar, ax3_cbar]
        axes_r = [ax1_r, ax2_r, ax3_r]
        axes_res = [ax1_res, ax2_res, ax3_res]
        return fig, axes_l, axes_c, axes_cbar, axes_r, axes_res


def myfigure_long1col(n_band):
    if n_band == 3:
        fig = plt.figure(figsize=(20, 16))
        ax1 = fig.add_axes([0.1, 0.66, 0.85, 0.28])
        ax2 = fig.add_axes([0.1, 0.38, 0.85, 0.28])
        ax3 = fig.add_axes([0.1, 0.10, 0.85, 0.28])
        axes = [ax1, ax2, ax3]
        return fig, axes


def myfigure_ncol(n_band, n_col, res=False, cbar=False):
    if n_band == 3:
        if n_col ==1:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_axes([0.1, 0.66, 0.85, 0.28])
            ax2 = fig.add_axes([0.1, 0.38, 0.85, 0.28])
            ax3 = fig.add_axes([0.1, 0.10, 0.85, 0.28])
            axes = [ax1, ax2, ax3]
            return fig, axes

        if n_col ==2:
            if res:
                fig = plt.figure(figsize=(20, 14))
                ax1_l = fig.add_axes([0.1, 0.70, 0.35, 0.23])
                ax2_l = fig.add_axes([0.1, 0.40, 0.35, 0.23])
                ax3_l = fig.add_axes([0.1, 0.10, 0.35, 0.23])
                ax1_r = fig.add_axes([0.5, 0.70, 0.35, 0.23])
                ax2_r = fig.add_axes([0.5, 0.40, 0.35, 0.23])
                ax3_r = fig.add_axes([0.5, 0.10, 0.35, 0.23])
                ax1_res = fig.add_axes([0.9, 0.70, 0.08, 0.23])
                ax2_res = fig.add_axes([0.9, 0.40, 0.08, 0.23])
                ax3_res = fig.add_axes([0.9, 0.10, 0.08, 0.23])
                axes_l = [ax1_l, ax2_l, ax3_l]
                axes_r = [ax1_r, ax2_r, ax3_r]
                axes_res = [ax1_res, ax2_res, ax3_res]
                return fig, axes_l, axes_r, axes_res
            elif cbar:
                fig = plt.figure(figsize=(20, 14))
                ax1_l = fig.add_axes([0.1, 0.70, 0.35, 0.23])
                ax2_l = fig.add_axes([0.1, 0.40, 0.35, 0.23])
                ax3_l = fig.add_axes([0.1, 0.10, 0.35, 0.23])
                ax1_r = fig.add_axes([0.5, 0.70, 0.35, 0.23])
                ax2_r = fig.add_axes([0.5, 0.40, 0.35, 0.23])
                ax3_r = fig.add_axes([0.5, 0.10, 0.35, 0.23])
                ax1_cbar = fig.add_axes([0.9, 0.70, 0.03, 0.23])
                ax2_cbar = fig.add_axes([0.9, 0.40, 0.03, 0.23])
                ax3_cbar = fig.add_axes([0.9, 0.10, 0.03, 0.23])
                axes_l = [ax1_l, ax2_l, ax3_l]
                axes_r = [ax1_r, ax2_r, ax3_r]
                axes_cbar = [ax1_cbar, ax2_cbar, ax3_cbar]
                return fig, axes_l, axes_r, axes_cbar
            
            else:
                fig = plt.figure(figsize=(20, 14))
                ax1_l = fig.add_axes([0.1, 0.7, 0.35, 0.24])
                ax2_l = fig.add_axes([0.1, 0.40, 0.35, 0.24])
                ax3_l = fig.add_axes([0.1, 0.10, 0.35, 0.24])
                ax1_r = fig.add_axes([0.6, 0.70, 0.35, 0.24])
                ax2_r = fig.add_axes([0.6, 0.40, 0.35, 0.24])
                ax3_r = fig.add_axes([0.6, 0.10, 0.35, 0.24])
                axes_l = [ax1_l, ax2_l, ax3_l]
                axes_r = [ax1_r, ax2_r, ax3_r]
                return fig, axes_l, axes_r

        if n_col ==3:
            if res:
                fig = plt.figure(figsize=(24, 14))
                ax1_l = fig.add_axes([0.06,  0.70, 0.2, 0.23])
                ax2_l = fig.add_axes([0.06,  0.40, 0.2, 0.23])
                ax3_l = fig.add_axes([0.06,  0.10, 0.2, 0.23])
                ax1_c = fig.add_axes([0.34,   0.70, 0.2, 0.23])
                ax2_c = fig.add_axes([0.34,   0.40, 0.2, 0.23])
                ax3_c = fig.add_axes([0.34,   0.10, 0.2, 0.23])
                ax1_r = fig.add_axes([0.64,  0.70, 0.2, 0.23])
                ax2_r = fig.add_axes([0.64,  0.40, 0.2, 0.23])
                ax3_r = fig.add_axes([0.64,  0.10, 0.2, 0.23])
                ax1_res = fig.add_axes([0.9, 0.70, 0.08, 0.23])
                ax2_res = fig.add_axes([0.9, 0.40, 0.08, 0.23])
                ax3_res = fig.add_axes([0.9, 0.10, 0.08, 0.23])
                axes_l = [ax1_l, ax2_l, ax3_l]
                axes_c = [ax1_c, ax2_c, ax3_c]
                axes_r = [ax1_r, ax2_r, ax3_r]
                axes_res = [ax1_res, ax2_res, ax3_res]
                return fig, axes_l, axes_c, axes_r, axes_res
            if cbar:
                fig = plt.figure(figsize=(24, 14))
                ax1_l = fig.add_axes([0.06,  0.70, 0.2, 0.23])
                ax2_l = fig.add_axes([0.06,  0.40, 0.2, 0.23])
                ax3_l = fig.add_axes([0.06,  0.10, 0.2, 0.23])
                ax1_c = fig.add_axes([0.34,   0.70, 0.2, 0.23])
                ax2_c = fig.add_axes([0.34,   0.40, 0.2, 0.23])
                ax3_c = fig.add_axes([0.34,   0.10, 0.2, 0.23])
                ax1_r = fig.add_axes([0.64,  0.70, 0.2, 0.23])
                ax2_r = fig.add_axes([0.64,  0.40, 0.2, 0.23])
                ax3_r = fig.add_axes([0.64,  0.10, 0.2, 0.23])
                ax1_res = fig.add_axes([0.9, 0.70, 0.03, 0.23])
                ax2_res = fig.add_axes([0.9, 0.40, 0.03, 0.23])
                ax3_res = fig.add_axes([0.9, 0.10, 0.03, 0.23])
                axes_l = [ax1_l, ax2_l, ax3_l]
                axes_c = [ax1_c, ax2_c, ax3_c]
                axes_r = [ax1_r, ax2_r, ax3_r]
                axes_res = [ax1_res, ax2_res, ax3_res]
                return fig, axes_l, axes_c, axes_r, axes_res

    if n_band == 2:
        if n_col ==1:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_axes([0.1, 0.66, 0.85, 0.28])
            ax2 = fig.add_axes([0.1, 0.38, 0.85, 0.28])
            axes = [ax1, ax2]
            return fig, axes
        if n_col ==2:
            if res:
                fig = plt.figure(figsize=(20, 10))
                ax1_l = fig.add_axes([0.1, 0.55, 0.35, 0.35])
                ax2_l = fig.add_axes([0.1, 0.10, 0.35, 0.35])
                ax1_r = fig.add_axes([0.5, 0.55, 0.35, 0.35])
                ax2_r = fig.add_axes([0.5, 0.10, 0.35, 0.35])
                ax1_res = fig.add_axes([0.9, 0.55, 0.08, 0.35])
                ax2_res = fig.add_axes([0.9, 0.10, 0.08, 0.35])
                axes_l = [ax1_l, ax2_l]
                axes_r = [ax1_r, ax2_r]
                axes_res = [ax1_res, ax2_res]
                return fig, axes_l, axes_r, axes_res
            elif cbar:
                fig = plt.figure(figsize=(20, 10))
                ax1_l = fig.add_axes([0.1, 0.55, 0.35, 0.35])
                ax2_l = fig.add_axes([0.1, 0.10, 0.35, 0.35])
                ax1_r = fig.add_axes([0.5, 0.55, 0.35, 0.35])
                ax2_r = fig.add_axes([0.5, 0.10, 0.35, 0.35])
                ax1_cbar = fig.add_axes([0.9, 0.55, 0.03, 0.35])
                ax2_cbar = fig.add_axes([0.9, 0.10, 0.03, 0.35])
                axes_l = [ax1_l, ax2_l]
                axes_r = [ax1_r, ax2_r]
                axes_cbar = [ax1_cbar, ax2_cbar]
                return fig, axes_l, axes_r, axes_cbar
        if n_col ==3:
            if res:
                fig = plt.figure(figsize=(24, 10))
                ax1_l = fig.add_axes([0.06,  0.55, 0.2, 0.35])
                ax2_l = fig.add_axes([0.06,  0.10, 0.2, 0.35])
                ax1_c = fig.add_axes([0.34,   0.55, 0.2, 0.35])
                ax2_c = fig.add_axes([0.34,   0.10, 0.2, 0.35])
                ax1_r = fig.add_axes([0.64,  0.55, 0.2, 0.35])
                ax2_r = fig.add_axes([0.64,  0.10, 0.2, 0.35])
                ax1_res = fig.add_axes([0.9, 0.55, 0.08, 0.35])
                ax2_res = fig.add_axes([0.9, 0.10, 0.08, 0.35])
                axes_l = [ax1_l, ax2_l]
                axes_c = [ax1_c, ax2_c]
                axes_r = [ax1_r, ax2_r]
                axes_res = [ax1_res, ax2_res]
                return fig, axes_l, axes_c, axes_r, axes_res
            if cbar:
                fig = plt.figure(figsize=(24, 10))
                ax1_l = fig.add_axes([0.06,  0.55, 0.2, 0.35])
                ax2_l = fig.add_axes([0.06,  0.10, 0.2, 0.35])
                ax1_c = fig.add_axes([0.34,   0.55, 0.2, 0.35])
                ax2_c = fig.add_axes([0.34,   0.10, 0.2, 0.35])
                ax1_r = fig.add_axes([0.64,  0.55, 0.2, 0.35])
                ax2_r = fig.add_axes([0.64,  0.10, 0.2, 0.35])
                ax1_res = fig.add_axes([0.9, 0.55, 0.03, 0.35])
                ax2_res = fig.add_axes([0.9, 0.10, 0.03, 0.35])
                axes_l = [ax1_l, ax2_l]
                axes_c = [ax1_c, ax2_c]
                axes_r = [ax1_r, ax2_r]
                axes_res = [ax1_res, ax2_res]
                return fig, axes_l, axes_c, axes_r, axes_res


    if n_band == 1:
        if n_col ==1:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_axes([0.1, 0.2, 0.85, 0.7])
            axes = [ax1]
            return fig, axes
        if n_col ==2:
            if res:
                fig = plt.figure(figsize=(20, 6))
                ax1_l = fig.add_axes([0.1, 0.2, 0.35, 0.7])
                ax1_r = fig.add_axes([0.5, 0.2, 0.35, 0.7])
                ax1_res = fig.add_axes([0.9, 0.2, 0.08, 0.7])
                axes_l = [ax1_l]
                axes_r = [ax1_r]
                axes_res = [ax1_res]
                return fig, axes_l, axes_r, axes_res
            elif cbar:
                fig = plt.figure(figsize=(20, 6))
                ax1_l = fig.add_axes([0.1, 0.2, 0.35, 0.7])
                ax1_r = fig.add_axes([0.5, 0.2, 0.35, 0.7])
                ax1_cbar = fig.add_axes([0.9, 0.2, 0.03, 0.7])
                axes_l = [ax1_l]
                axes_r = [ax1_r]
                axes_cbar = [ax1_cbar]
                return fig, axes_l, axes_r, axes_cbar
            else:
                fig = plt.figure(figsize=(20, 6))
                ax1_l = fig.add_axes([0.1, 0.2, 0.35, 0.7])
                ax1_r = fig.add_axes([0.6, 0.2, 0.35, 0.7])
                axes_l = [ax1_l]
                axes_r = [ax1_r]
                return fig, axes_l, axes_r
        if n_col ==3:
            if res:
                fig = plt.figure(figsize=(24, 6))
                ax1_l = fig.add_axes([0.06,  0.2, 0.2, 0.7])
                ax1_c = fig.add_axes([0.34,   0.2, 0.2, 0.7])
                ax1_r = fig.add_axes([0.64,  0.2, 0.2, 0.7])
                ax1_res = fig.add_axes([0.9, 0.2, 0.08, 0.7])
                axes_l = [ax1_l]
                axes_c = [ax1_c]
                axes_r = [ax1_r]
                axes_res = [ax1_res]
                return fig, axes_l, axes_c, axes_r, axes_res



def myfigure(n):
  if n == 2:
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_axes([0.10, 0.15, 0.35, 0.8])
    ax2 = fig.add_axes([0.6, 0.15, 0.35, 0.8])
    return fig, ax1, ax2
  if n == 4:
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_axes([0.1, 0.6, 0.35, 0.35])
    ax2 = fig.add_axes([0.6, 0.6, 0.35, 0.35])
    ax3 = fig.add_axes([0.1, 0.10, 0.35, 0.35])
    ax4 = fig.add_axes([0.6, 0.10, 0.35, 0.35])
    return fig, ax1, ax2, ax3, ax4
  if n == 6:
    fig = plt.figure(figsize=(16, 18))
    ax1 = fig.add_axes([0.1, 0.72, 0.35, 0.25])
    ax2 = fig.add_axes([0.6, 0.72, 0.35, 0.25])
    ax3 = fig.add_axes([0.1, 0.4, 0.35, 0.25])
    ax4 = fig.add_axes([0.6, 0.4, 0.35, 0.25])
    ax5 = fig.add_axes([0.1, 0.08, 0.35, 0.25])
    ax6 = fig.add_axes([0.6, 0.08, 0.35, 0.25])
    return fig, ax1, ax2, ax3, ax4, ax5, ax6


def plot_fwhm(
    df, bands, smooth="median", n_smooth=5, p_scale=None, out="fwhm.png"):
    """
    Plot time-series of fwhms.

    Parameters
    ----------
    df : pandas.DataFrame
      DataFrame with fwhm info.
    bands : array-like
      observation band
    smooth : str
      smoothing method
    n_smooth : str
      number of data used for smoothing
    p_scale : float
      pixel scale in arcsec/pix
    out : str
      output png filename
    """

    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_axes([0.10, 0.55, 0.8, 0.4])
    ax2 = fig.add_axes([0.10, 0.05, 0.8, 0.4])

    ax1.set_xlabel("JD")
    ax1.set_ylabel("FWHM [pixel]")

    ax2.set_xlabel("Frame")
    ax2.set_ylabel("FWHM [pixel]")

    for idx_b, b in enumerate(bands):
        ax1.errorbar(
            df["jd"], df[f"fwhm_{b}"], df[f"fwhmerr_{b}"],
            marker=None, color=mycolor[idx_b], ls="None")
        ax1.scatter(
            df["jd"], df[f"fwhm_{b}"], label=f"Raw fwhm {b}", 
            ec=mycolor[idx_b], facecolor="None", s=300)

        ax2.errorbar(
            df["nframe"], df[f"fwhm_{b}"], df[f"fwhmerr_{b}"],
            marker=None, color=mycolor[idx_b], ls="None")
        ax2.scatter(
            df["nframe"], df[f"fwhm_{b}"], label=f"Raw fwhm {b}", 
            ec=mycolor[idx_b], facecolor="None", s=300)

        # Smoothing
        if smooth == "median":
            # ToDo: edge handilng
            # ToDo: fwhmerr calculation
            df[f"fwhm_{b}_smooth"] = medfilt(df[f"fwhm_{b}"], n_smooth)
            ax1.errorbar(
                df["jd"], df[f"fwhm_{b}_smooth"], df[f"fwhmerr_{b}"],
                label=f"Smoothed {b} ({smooth}, n={n_smooth})", 
                fmt="o", lw=1, marker="x", color=mycolor[idx_b], ms=10)
            ax2.errorbar(
                df["nframe"], df[f"fwhm_{b}_smooth"], df[f"fwhmerr_{b}"],
                label=f"Smoothed {b} ({smooth}, n={n_smooth})", 
                fmt="o", lw=1, marker="x", color=mycolor[idx_b], ms=10)


    # Add seeing in arcsec
    if p_scale:
        for ax in [ax1, ax2]:
            fwhm_min_pix, fwhm_max_pix = ax.get_ylim()
            fwhm_min_arcsec = fwhm_min_pix * p_scale
            fwhm_max_arcsec = fwhm_max_pix * p_scale
            ax_arcsec = ax.twinx()
            ax_arcsec.set_ylim([fwhm_min_arcsec, fwhm_max_arcsec])
            ax_arcsec.set_ylabel("FWHM [arcsec]")
    ax1.legend()
    ax2.legend()
    plt.savefig(out, dpi=200)
    plt.close()

    return df


def add_circle_with_radius(ax, df, key_x, key_y, rad, color, ls, label):
  """
  Add circle with radius.
  The circle size is fixed (to 10 by default).

  Parameters
  ----------
  df : pandas.DataFrame
    DataFrame for objects
  rad : float
    circle radius in pixel
  color : str
    color of circle(s)
  ls : str
    line style of circle(s)
  label : str
    object label in legend
  """
  s_circle = 10.
  ax.scatter(
    df[key_x], df[key_y], color=color, s=s_circle, lw=1, 
    facecolor="None", alpha=1, label=label)
  ax.add_collection(PatchCollection(
    [Circle((x,y), rad) for x,y in zip(df[key_x], df[key_y])],
    color=color, ls=ls, lw=1, facecolor="None", label=None)
    )


def plot_photregion_ref(
  image, stddev, df_refall=None, df_ref=None, df_sep=None, 
  radius_ref=20, key_x="x", key_y="y", out="photregion.png"):
  """
  Plot region of circle photometry.

  Parameters
  ----------
  image : array-like
    object extracted image
  stddev : float
    image background standard deviation
  df_refll : pandas.DataFrame
    DataFrame for reference stars before merging
  df_ref : pandas.DataFrame
    DataFrame for reference stars
  df_sep : pandas.DataFrame
    DataFrame for sep detected bright stars
  radius : float
    aperture radius 
  key_x, key_y : str
    keywords for x, y
  out : str
    output png filename
  """

  # Plot src image after 5-sigma clipping 
  sigma = 5
  _, vmin, vmax = sigmaclip(image, sigma, sigma)
  ny, nx = image.shape
  fig = plt.figure(figsize=(12,int(12*ny/nx)))
  ax = fig.add_subplot(111)
  ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)

  # Reference stars in orange. Radius is dummy.
  if (df_refall is not None) and not df_refall.empty:
    label_ref = f"All Ref. stars in catalog N={len(df_refall)}"
    add_circle_with_radius(
      ax, df_refall, key_x, key_y, radius_ref/2, "black", "solid", label_ref)


  # Reference stars in orange. Radius is real.
  if (df_ref is not None) and not df_ref.empty:
    label_ref = f"Merged Ref. stars in catalog N={len(df_ref)}"
    add_circle_with_radius(
      ax, df_ref, key_x, key_y, radius_ref, "orange", "dashed", label_ref)

  # Sep detected stars. Radius is real.
  if (df_sep is not None) and not df_sep.empty:
    label_sep = f"Sep detections N={len(df_sep)}"
    add_circle_with_radius(
      ax, df_sep, key_x, key_y, radius_ref, "blue", "dotted", label_sep)

  ax.set_xlim([0, nx])
  ax.set_ylim([0, ny])
  ax.legend().get_frame().set_alpha(1.0)
  ax.invert_yaxis()
  plt.tight_layout()
  plt.savefig(out, dpi=200)
  plt.close()


def plot_photregion(
    image, stddev, df_list=None, rad_list=None, key_x_list=None, key_y_list=None, 
    label_list=None, mask=None, out="photregion.png"):
    """
    Plot region of circle photometry.

    Parameters
    ----------
    image : array-like
      object extracted image
    stddev : float
      image background standard deviation
    df_obj : pandas.DataFrame
      DataFrame for a target 
    df_ref : pandas.DataFrame
      DataFrame for reference stars
    df_ref0 : pandas.DataFrame
      DataFrame for reference stars (original)
    df_sep : pandas.DataFrame
      DataFrame for sep detected bright stars
    radius : float
      aperture radius 
    key_x, key_y : str
      keywords for x, y
    mask: array-like, optional
      mask region
    out : str
      output png filename
    """

    # Plot src image after 5-sigma clipping 
    sigma = 5
    _, vmin, vmax = sigmaclip(image, sigma, sigma)
    ny, nx = image.shape
    fig = plt.figure(figsize=(12,int(12*ny/nx)))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
 
    # Plot mask
    if mask is not None:
        ax.imshow(mask, alpha=0.2)

    mycolor = ["red", "orange", "blue", "magenta"]

    # Reference stars in orange. Radius is real.
    for idx, df in enumerate(df_list):
        if not df.empty:
            rad   = rad_list[idx]
            key_x = key_x_list[idx]
            key_y = key_y_list[idx]
            color = mycolor[idx]
            ls    = myls[idx]
            if label_list is not None:
                label = label_list[idx]
            add_circle_with_radius(
                ax, df, key_x, key_y, rad, color, ls, label)

    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    ax.legend().get_frame().set_alpha(1.0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_photregion_old(
  image, stddev, df_ref_use, df_ref=None, df_ref_all=None, rad_ref=10, 
  df_sep=None, rad_sep=10, df_obj=None, rad_obj=10, out="photregion.png"):
  """Plot region of circle photometry.

  Parameters
  ----------
  image : array-like
    object extracted image
  stddev : float
    image background standard deviation
  df_ref_use : pandas.DataFrame
    DataFrame for used reference stars (remove in edge and close)
  df_ref : pandas.DataFrame
    DataFrame for not in edge reference stars (remove in edge)
  df_ref_all : pandas.DataFrame
    DataFrame for all reference stars (includes in edge)
  rad_ref : float
    aperture radius of reference stars in pixel
  df_obj : pandas.DataFrame
    DataFrame for the target
  rad_obj : float
    aperture radius of the target in pixel
  df_sep : pandas.DataFrame
    DataFrame for the sep detections
  rad_sep : float
    aperture radius of the sep detections in pixel
  out : str
    output png filename
  """

  # Keywords
  key_ref_x, key_ref_y = "x1", "y1"
  key_x, key_y = "x", "y"
  # Plot value range
  vmin  = np.median(image)-1.5*stddev
  vmax  = np.median(image)+5*stddev
  ny, nx = image.shape
  fig = plt.figure(figsize=(12,int(12*ny/nx)))
  ax = fig.add_subplot(111)
  ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)

  # Used reference stars in orange. Radius is real.
  if df_ref is not None:
    label_ref = f"Used ref. stars N={len(df_ref_use)}"
    add_circle_with_radius(
      ax, df_ref_use, key_x, key_y, rad_ref, "orange", "dashed", label_ref)

  # All reference stars in blue Radius is fake.
  if df_ref_all is not None:
    label_ref_all = f"All ref. stars N={len(df_ref_all)}(rad={rad_ref})"
    add_circle_with_radius(
      ax, df_ref_all, key_x, key_y, rad_ref*2, "blue", "dotted", label_ref_all)

  # Not in edge reference stars in red. Radius is fake.
  if df_ref is not None:
    label_ref = f"Not in edge ref. stars N={len(df_ref)}"
    add_circle_with_radius(
      ax, df_ref, key_x, key_y, rad_ref*0.5, "red", "dotted", label_ref)

  # Sep detections in green. Radius is real.
  if df_sep is not None:
    label_sep = f"Sep detections N={len(df_sep)}(rad={rad_sep})"
    add_circle_with_radius(
      ax, df_sep, key_x, key_y, rad_sep*1.5, "green", "dashed", label_sep)

  # Targeet in magenta. Radius is real.
  if df_obj is not None:
    label_obj = f"Target (rad={rad_obj})"
    add_circle_with_radius(
      ax, df_obj, key_x, key_y, rad_sep, "magenta", "solid", label_obj)

  ax.set_xlim([0, nx])
  ax.set_ylim([0, ny])
  ax.legend().get_frame().set_alpha(1.0)
  ax.invert_yaxis()
  plt.tight_layout()
  plt.savefig(out, dpi=200)
  plt.close()


def plot_photregion_iso(
  image, stddev, objects, isomap, out):
  """Plot region of circle photometry.

  Parameters
  ----------
  image : array-like
    object extracted image
  stddev : float
    image background standard deviation
  objects : array of 
    extracted objects
  isomap : array of
    isomap
  out : str
    output png filename
  """
  
  vmin  = np.median(image)-1.5*stddev
  vmax  = np.median(image)+10.0*stddev
  ny, nx = image.shape
  fig = plt.figure(figsize=(12,int(12*ny/nx)))

  ax = fig.add_subplot(111)
  ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
  ax.scatter(
    objects['x'], objects['y'], 
    color="blue", s=50, lw=3, facecolor="None", alpha=0.5,
    label="Final used (green photometry radius)")
  radius = 10
  ax.add_collection(PatchCollection(
    [Circle((x,y), radius) for x,y in zip(objects['x'], objects['y'])],
    color="green", ls="dotted", lw=2, facecolor="None", 
    label=f"Photometry circle (radius={radius})"))

  mask = np.ma.masked_where(isomap.sum(axis=0)==0, isomap.sum(axis=0))
  ax.imshow(mask, cmap='winter_r', alpha=0.5)

  ax.set_xlim([0, nx])
  ax.set_ylim([0, ny])
  ax.legend().get_frame().set_alpha(1.0)
  ax.invert_yaxis()
  plt.tight_layout()
  plt.savefig(out, dpi=100)
  plt.close()


# == For plotting =============================================================

def flux_figure(N_band):
  """Create figures for flux light curves.

  Parameter
  ---------
  N_band : int
    number of observed band

  Returns
  -------
  fig : matplotlib.figure.Figure
    matplotlib.figure.Figure class object
  axes_raw : list of matplotlib.axes._axes.Axes
    Axes for raw light curves
  axes_norm : list of matplotlib.axes._axes.Axes
    Axes for normalized light curves
  """
  if N_band == 2:
    fig = plt.figure(figsize=(20, 8))
    # Raw
    ax1 = fig.add_axes([0.1, 0.55, 0.35, 0.4])
    ax2 = fig.add_axes([0.1, 0.15, 0.35, 0.4])
    # Normalized
    ax3 = fig.add_axes([0.6, 0.55, 0.35, 0.4])
    ax4 = fig.add_axes([0.6, 0.15, 0.35, 0.4])
    axes_raw = [ax1, ax2]
    axes_norm = [ax3, ax4]

  if N_band == 3:
    fig = plt.figure(figsize=(20, 12))
    # Raw
    ax1 = fig.add_axes([0.1, 0.66, 0.35, 0.28])
    ax2 = fig.add_axes([0.1, 0.38, 0.35, 0.28])
    ax3 = fig.add_axes([0.1, 0.1, 0.35, 0.28])
    # Normalized
    ax4 = fig.add_axes([0.6, 0.66, 0.35, 0.28])
    ax5 = fig.add_axes([0.6, 0.38, 0.35, 0.28])
    ax6 = fig.add_axes([0.6, 0.1, 0.35, 0.28])
    axes_raw = [ax1, ax2, ax3]
    axes_norm = [ax4, ax5, ax6]

  return fig, axes_raw, axes_norm


def myfigure_refladder(N_band):
  """Create figures for object light curves with atmospheric templates.

  Parameter
  ---------
  N_band : int
    number of observed band

  Returns
  -------
  fig : matplotlib.figure.Figure
    matplotlib.figure.Figure class object
  ax_flux : list of matplotlib.axes._axes.Axes
    Axes for raw flux light curves
  ax_templates : list of matplotlib.axes._axes.Axes
    Axes for atmospheric template light curves
  ax_mag : list of matplotlib.axes._axes.Axes
    Axes for corrected magnitude light curves
  ax_mag : list of matplotlib.axes._axes.Axes
    Axes for color light curves
  """

  if N_band == 2:
    fig = plt.figure(figsize=(20, 8))
    # Raw
    ax1 = fig.add_axes([0.08, 0.6, 0.25, 0.33])
    ax2 = fig.add_axes([0.08, 0.18, 0.25, 0.33])
    # template
    ax3 = fig.add_axes([0.08, 0.55, 0.25, 0.05])
    ax4 = fig.add_axes([0.08, 0.15, 0.25, 0.05])
    # Normalized mag
    ax5 = fig.add_axes([0.4, 0.55, 0.25, 0.4])
    ax6 = fig.add_axes([0.4, 0.15, 0.25, 0.4])
    # Color (mag diff)
    ax7 = fig.add_axes([0.72, 0.55, 0.25, 0.4])
    ax8 = fig.add_axes([0.72, 0.15, 0.25, 0.4])
    axes_flux = [ax1, ax2]
    axes_template = [ax3, ax4]
    axes_mag = [ax5, ax6]
    axes_col = [ax7, ax8]

  if N_band == 3:
    fig = plt.figure(figsize=(20, 12))
    # Raw
    ax1 = fig.add_axes([0.08, 0.71, 0.25, 0.23])
    ax2 = fig.add_axes([0.08, 0.43, 0.25, 0.23])
    ax3 = fig.add_axes([0.08, 0.15, 0.25, 0.23])
    # template
    ax4 = fig.add_axes([0.08, 0.66, 0.25, 0.05])
    ax5 = fig.add_axes([0.08, 0.38, 0.25, 0.05])
    ax6 = fig.add_axes([0.08, 0.1, 0.25, 0.05])
    # Normalized mag
    ax7 = fig.add_axes([0.4, 0.66, 0.25, 0.28])
    ax8 = fig.add_axes([0.4, 0.38, 0.25, 0.28])
    ax9 = fig.add_axes([0.4, 0.1, 0.25, 0.28])
    # Color (mag diff)
    ax10 = fig.add_axes([0.72, 0.66, 0.25, 0.28])
    ax11 = fig.add_axes([0.72, 0.38, 0.25, 0.28])
    ax12 = fig.add_axes([0.72, 0.1, 0.25, 0.28])
    axes_flux = [ax1, ax2, ax3]
    axes_template = [ax4, ax5, ax6]
    axes_mag = [ax7, ax8, ax9]
    axes_col = [ax10, ax11, ax12]

  return fig, axes_flux, axes_template, axes_mag, axes_col


def myfigure_colcol_lc(n_band, cbar=False):
    """
    Create an ax for color color diagram with lc.
    
    Parameters
    ----------
    n_band : int
        number of band
    cbar : bool
        color bar

    Returns
    -------
    fig : figure
    axes : axes 
        for lc
    axes_colcol : axes 
        for colcol
    """
    
    if n_band==3:
        fig = plt.figure(figsize=(20, 14))
        ax1_l = fig.add_axes([0.1, 0.7, 0.35, 0.24])
        ax2_l = fig.add_axes([0.1, 0.40, 0.35, 0.24])
        ax3_l = fig.add_axes([0.1, 0.10, 0.35, 0.24])
       
        axes = [ax1_l, ax2_l, ax3_l]

        if cbar:
            ax = fig.add_axes([0.55, 0.35, 0.30, 0.30])
            ax_u = fig.add_axes([0.55, 0.65, 0.30, 0.07])
            ax_r = fig.add_axes([0.85, 0.35, 0.07, 0.30])
            ax_cbar = fig.add_axes([0.90, 0.35, 0.02, 0.30])
            axes_colcol = [ax, ax_r, ax_u, ax_cbar]
        else:
            ax = fig.add_axes([0.55, 0.35, 0.30, 0.30])
            ax_u = fig.add_axes([0.55, 0.65, 0.30, 0.07])
            ax_r = fig.add_axes([0.85, 0.35, 0.07, 0.30])
            axes_colcol = [ax, ax_r, ax_u]

    return fig, axes, axes_colcol


def myfigure_colcol(large=False, cbar=False):
  """
  Create an ax for color color diagram.
  """
  
  if large:
      fig = plt.figure(figsize=(16, 16))
  else:
      fig = plt.figure(figsize=(8, 6))

  if cbar:
      ax = fig.add_axes([0.15, 0.15, 0.60, 0.70])
      ax_u = fig.add_axes([0.15, 0.85, 0.6, 0.07])
      ax_r = fig.add_axes([0.75, 0.15, 0.05, 0.70])
      ax_cbar = fig.add_axes([0.82, 0.15, 0.03, 0.70])
      return fig, ax, ax_r, ax_u, ax_cbar
  else:
      ax = fig.add_axes([0.15, 0.1, 0.70, 0.70])
      ax_u = fig.add_axes([0.15, 0.80, 0.70, 0.1])
      ax_r = fig.add_axes([0.85, 0.1, 0.1, 0.70])
      return fig, ax, ax_r, ax_u


def plot_colcol(
    df, fig, ax, ax_r, ax_u, b1, b2, b3, magtype, JD0, rotP, col=mycolor[0], 
    col_hist = "black", marker="o", each=True, mean=True, key_time=None, ax_cbar=None,
    label=None):
    """
    Plot color color diagrams.

    wmean (weighted mean) is the best.

    fig : 
        for cbar plot
    """
    ax.set_xlabel(f"${b1}$-${b2}$")
    ax.set_ylabel(f"${b2}$-${b3}$")

    # xlim

    # Remove invalid data ?
    #df = df[df["eflag_color"]==0]
    # Time window
    #print(f"  {idx} : Delta T = {deltaT:.2f}")

    #  # Mean and std
    #  c_mean = np.mean(df[f"{b1}_{b2}"])
    #  # Photometric uncertainty
    #  c_std_phot = adderr(df[f"{b1}_{b1}err"])/len(df)
    #  # Standard Deviation
    #  c_SD = np.std(df[f"{band_l}_{band_r}"])
    #  # Standard Error
    #  c_SE = c_SD/np.sqrt(len(df))
    #  # Total error 
    #  c_std = np.sqrt(c_std_phot**2 + c_SE**2)

    # Weighted mean
    c1_w = 1/df[f"{b1}_{b2}err"]**2
    c1_wmean = np.average(df[f"{b1}_{b2}"], weights=c1_w)
    # Check !!!!!!
    # SD of weighted mean
    c1_wstd = np.sqrt(1/np.sum(c1_w))
    c2_w = 1/df[f"{b2}_{b3}err"]**2
    c2_wmean = np.average(df[f"{b2}_{b3}"], weights=c2_w)
    # Check !!!!!!
    # SD of weighted mean
    c2_wstd = np.sqrt(1/np.sum(c2_w))

    # Standard mean
    N = len(df)
    c1_mean = np.average(df[f"{b1}_{b2}"])
    c2_mean = np.average(df[f"{b2}_{b3}"])
    # Standard deviation
    c1_SD   = np.std(df[f"{b1}_{b2}"])
    c2_SD   = np.std(df[f"{b2}_{b3}"])
    # Standard error
    c1_SE   = c1_SD/np.sqrt(N)
    c2_SE   = c2_SD/np.sqrt(N)
    # Sum of errors
    c1_sumerr   = adderr(df[f"{b1}_{b2}err"])/np.sqrt(N)
    c2_sumerr   = adderr(df[f"{b2}_{b3}err"])/np.sqrt(N)
    
    ## Round error
    c1_wmean_str, c1_wstd_str = round_error(c1_wmean, c1_wstd)
    c2_wmean_str, c2_wstd_str = round_error(c2_wmean, c2_wstd)
    c1_mean_str1, c1_SD_str = round_error(c1_mean, c1_SD)
    c2_mean_str1, c2_SD_str = round_error(c2_mean, c2_SD)
    c1_mean_str2, c1_SE_str = round_error(c1_mean, c1_SE)
    c2_mean_str2, c2_SE_str = round_error(c2_mean, c2_SE)
    c1_mean_str3, c1_sumerr_str = round_error(c1_mean, c1_sumerr)
    c2_mean_str3, c2_sumerr_str = round_error(c2_mean, c2_sumerr)
   
    # Test 2022-07-07
    label = (
        f"Weighted average (N={len(df)})\n"
        f"{b1}-{b2} ${c1_wmean_str}" + r"\pm" + f"{c1_wstd_str}$\n"
        f"{b2}-{b3} ${c2_wmean_str}" + r"\pm" + f"{c2_wstd_str}$\n"
        #f"Standard Deviation\n"
        #f"{b1}-{b2} ${c1_mean_str1}" + r"\pm" + f"{c1_SD_str}$\n"
        #f"{b2}-{b3} ${c2_mean_str1}" + r"\pm" + f"{c2_SD_str}$\n"
        #f"Standard Error\n"
        #f"{b1}-{b2} ${c1_mean_str2}" + r"\pm" + f"{c1_SE_str}$\n"
        #f"{b2}-{b3} ${c2_mean_str2}" + r"\pm" + f"{c2_SE_str}$\n"
        #f"Sum of uncertainties\n"
        #f"{b1}-{b2} ${c1_mean_str3}" + r"\pm" + f"{c1_sumerr_str}$\n"
        #f"{b2}-{b3} ${c2_mean_str3}" + r"\pm" + f"{c2_sumerr_str}$\n"
        )
    print(label)
    # For open circle
    if key_time is not None:
        # Convert to sec
        df[key_time] = (df[key_time]-np.min(df[key_time]))*24.*3600.
        mapp = ax.scatter(
          df[f"{b1}_{b2}"], df[f"{b2}_{b3}"], c=df[key_time],  cmap=cm.inferno,
          s=70, lw=1, 
          marker=marker, facecolor="None", edgecolor=col, zorder=2, label=label)
        cbar = fig.colorbar(mapp, ax_cbar)
        cbar.set_label("Elapsed time [sec]")
        # Add errorbars
        ax.errorbar(
          df[f"{b1}_{b2}"], df[f"{b2}_{b3}"], xerr=df[f"{b1}_{b2}err"], 
          yerr=df[f"{b2}_{b3}err"], color=col, lw=0.5, ms=3, ls="None",
          marker=None, capsize=0, zorder=1)
    # Plot each point
    elif each:
        ax.scatter(
          df[f"{b1}_{b2}"], df[f"{b2}_{b3}"], color=col, s=70, lw=1, 
          marker=marker, facecolor="None", edgecolor=col, zorder=-1, label=label)
        ax.errorbar(
          df[f"{b1}_{b2}"], df[f"{b2}_{b3}"], xerr=df[f"{b1}_{b2}err"], 
          yerr=df[f"{b2}_{b3}err"], color=col, lw=0.5, ms=3, ls="None",
          marker=None, capsize=0)
    elif mean:
        ax.scatter(
          c1_wmean, c2_wmean,
          color=col, s=70, lw=1, 
          marker=marker, facecolor="None", edgecolor=col, zorder=-1, label=label)
        ax.errorbar(
          c1_wmean, c2_wmean, c1_wstd, c2_wstd, 
          color=col, lw=0.5, ms=3, ls="None",
          marker=None, capsize=0)

    ax_u.hist(
        df[f"{b1}_{b2}"], color=col_hist, histtype="step")
    ax_r.hist(
        df[f"{b2}_{b3}"], orientation="horizontal", color=col_hist, histtype="step")


    if mean:
        # Mean and std
        # Plot mean and std lines in main plot

        # Do not plot since the error of weighted mean/standard error are small
        #x.plot(c1_wmean, c2_wmean, marker="o", color=col_hist)
        #ax.hlines(c2_wmean, c1_wmean-c1_wstd, c1_wmean+c1_wstd, color=col_hist)
        #ax.vlines(c1_wmean, c2_wmean-c2_wstd, c2_wmean+c2_wstd, color=col_hist)

        ax_r_xmax = ax_r.get_xlim()[1]
        ax_r.hlines(c2_wmean, 0, ax_r_xmax, color=col_hist, ls=myls[1])
        # Do not plot since the error of weighted mean/standard error are small
        #ax_r.hlines(c2_wmean-c2_wstd, 0, ax_r_xmax, color=col_hist, ls=myls[1])
        #ax_r.hlines(c2_wmean+c2_wstd, 0, ax_r_xmax, color=col_hist, ls=myls[1])

        ax_u_ymax = ax_u.get_ylim()[1]
        ax_u.vlines(c1_wmean, 0, ax_u_ymax, color=col_hist, ls=myls[1])
        # Do not plot since the error of weighted mean/standard error are small
        #ax_u.vlines(c1_wmean-c1_wstd, 0, ax_u_ymax, color=col_hist, ls=myls[1])
        #ax_u.vlines(c1_wmean+c1_wstd, 0, ax_u_ymax, color=col_hist, ls=myls[1])
 
        # ax_p_hist.set_xlim([p_min, p_max])
        # ax_p_hist.set_ylim([0, ax_p_ymax])
        # ax_dmag_hist.set_ylim([dmag_min, dmag_max])
        # ax_dmag_hist.set_xlim([0, ax_dmag_xmax])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax_u.set_xlim(xmin, xmax)
    ax_r.set_ylim(ymin, ymax)
    ax_u.axes.xaxis.set_visible(False)
    ax_r.axes.yaxis.set_visible(False)
