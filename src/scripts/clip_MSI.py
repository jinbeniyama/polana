#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Clip fits files obtained with MSI.
Original size (nx, ny) = (512, 512)
By default, output fits has (nx, ny) = (350, 230).
"""
from argparse import ArgumentParser as ap
import astropy.io.fits as fits
import sys
import numpy as np
import os
import datetime 


def main(args):
    """This is the main function called by the script.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed from the command-line as defined below.
    """
    
    # Create a directory to save output fits
    outdir = "clip"
    if os.path.isdir(outdir):
        print(f"Already exists {outdir}")
    else:
        os.makedirs(outdir)
    
    fitsname = os.path.basename(args.fits)
    basename = fitsname.split(".")[0]

    # Select sensitive pixels
    xmin, xmax = args.xr
    ymin, ymax = args.yr

    # Open a 2-d fits
    # Header keywords are optimized for Kanata/HONIR
    hdu = fits.open(args.fits)
    hdr = hdu[0].header
    ny0, nx0 = hdu[0].data.shape
    print(f"  Original Data Shape (nx,ny)=({nx0},{ny0})")
    assert nx0 == 512, "Check the input."
    assert ny0 == 512, "Check the input."

    # Clip
    hdu[0].data = hdu[0].data[ymin:ymax, xmin:xmax]
    ny1, nx1 = hdu[0].data.shape
    print(f"  Reduced Data Shape (nx,ny) = ({nx1},{ny1})")
    print(f"    x : {xmin:4d}-{xmax:4d}")
    print(f"    y : {ymin:4d}-{ymax:4d}")

    # Add header keyword
    hdu[0].header.add_history(
      f"[clip_MSI] original fits: {fitsname}")
    hdu[0].header.add_history(
      f"[clip_MSI] dim: ({nx0},{ny0}) to ({nx1},{ny1})")

    # Save the fits
    out = f"{basename}_clip.fits"
    hdu.writeto(os.path.join(outdir, out), overwrite=True)


if __name__ == "__main__":
    parser = ap(
        description="Clip 2d fits cube")
    parser.add_argument(
        "flist", nargs="*", type=str, 
        help="a reduced 2-d fits")
    parser.add_argument(
        "--xr", nargs=2, type=int, default=[150, 400], help="x range")
    parser.add_argument(
        "--yr", nargs=2, type=int, default=[10, 240], help="y range")
    args = parser.parse_args()
 
    for f in args.flist:
        args.fits = f
        main(args)
