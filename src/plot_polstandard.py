#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot polarization standard stars and optional Horizons objects.
TODO: Add time vs. elevation.
"""
import os
from argparse import ArgumentParser as ap
import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord
import astropy.units as u

from polana.util_pol import std_unpol, std_strong


# --- marker/color mapping ---
type_styles = {
    "nonpol": {"color": "blue", "marker": "o", "label": "Unpolarized"},
    "unpol": {"color": "blue", "marker": "o", "label": "Unpolarized"},
    "strong": {"color": "red", "marker": "s", "label": "Polarized"},
    "pol": {"color": "red", "marker": "s", "label": "Polarized"},
    "horizons": {"color": "green", "marker": "^", "label": "Object"}
}

type_dicts = {
    "nonpol": std_unpol,
    "unpol": std_unpol,
    "strong": std_strong,
    "pol": std_strong
}

if __name__ == "__main__":
    parser = ap(
        description="Plot polarization standard stars and optional Horizons object.")
    parser.add_argument(
        "types", type=str, nargs='+', 
        help="one or more types: strong, nonpol")
    parser.add_argument(
        "--obj", type=str, 
        help="Horizons object ID")
    parser.add_argument(
        "--t0", type=str, 
        help="start date (YYYY-MM-DD)")
    parser.add_argument(
        "--t1", type=str, 
        help="stop date (YYYY-MM-DD)")
    parser.add_argument(
        "--tstep", type=str, default="1d", 
        help="time step, e.g., 1d")
    parser.add_argument(
        "--code", type=str, default="500", 
        help="observatory/location code")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- plot standard stars ---
    for t in args.types:
        t_lower = t.lower()
        if t_lower not in type_styles:
            raise ValueError(f"Unknown type '{t}'. Use 'strong' or 'nonpol'.")

        pol_dict = type_dicts[t_lower]
        style = type_styles[t_lower]

        ra_list = []
        dec_list = []
        names = []

        for name, data in pol_dict.items():
            coord = SkyCoord(
                data["ra"], data["dec"],
                unit=(u.hourangle, u.deg),
                frame="icrs"
            )
            ra_list.append(coord.ra.deg)
            dec_list.append(coord.dec.deg)
            names.append(name)

        ax.scatter(ra_list, dec_list, s=50, color=style["color"],
                   marker=style["marker"], label=style["label"])

        for ra, dec, name in zip(ra_list, dec_list, names):
            ax.text(ra, dec, name, fontsize=8)

    # --- optional Horizons object ---
    if args.obj and args.t0 and args.t1:
        obj_query = Horizons(id=args.obj, location=args.code,
                             id_type="smallbody",
                             epochs={'start': args.t0,
                                     'stop': args.t1,
                                     'step': args.tstep})
        eph = obj_query.ephemerides()
        ra_h = eph['RA']
        dec_h = eph['DEC']
        ax.scatter(ra_h, dec_h, s=60, color=type_styles["horizons"]["color"],
                   marker=type_styles["horizons"]["marker"],
                   label=f"{args.obj}")
        for ra, dec in zip(ra_h, dec_h):
            ax.text(ra, dec, args.obj, fontsize=7)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Polarization Standards")
    ax.set_xlim(360, 0)
    ax.set_ylim(-90, 90)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
