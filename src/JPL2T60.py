#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert ephemeris obtained from JPL to T60 style.

T60 format has 76 (+\n) characters in a single line.
Reference: mer20221231.eph
#T60CTRL2.1  (UT)   **h**m**.*s +**d**'**.*" NPACCW ANGRAD dRA      dDEC    
2022-12-31T00:00:00 19h42m17.1s -20d48'03.2" -007.0 0004.4 -0057.60 +0038.28
2022-12-31T00:01:00 19h42m17.0s -20d48'02.6" -007.0 0004.4 -0057.60 +0038.28   

Example
-------
>>> JPL2T60.py ephem/2010XC15_2022-12-24to2022-12-25step1mcodeF51eph.txt --out 2010XC15_UT20221224.eph
"""
import os
from argparse import ArgumentParser as ap
import pandas as pd
import numpy as np
import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord


def deg2hmsdms(ra, dec):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    radec = c.to_string("hmsdms")
    ra = radec.split(" ")[0]
    dec = radec.split(" ")[1]
    return ra, dec


if __name__ == "__main__":
    parser = ap(description="Convert JPL ephem to T60 ephem")
    parser.add_argument(
        "JPL", help="ephemeris obtained from JPL/HORIZONS")
    parser.add_argument(
        "--out", default="T60ephem.txt", help="Output filename")
    parser.add_argument(
        "--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    if args.outdir:
        outdir = args.outdir
    else:
        os.makedirs(outdir, exist_ok=True)

    # RA_rate, DEC_rate in arcsec/hour
    # RA_rate_arcsec_s, DEC_rate_arcsec_s in arcsec/s
    columns = ["targetname", "datetime_str", "datetime_jd", "V", "H", "alpha", 
               "alpha_true", "EL", "RA", "RA_app", "RA_rate", "DEC", 
               "DEC_app", "DEC_rate", "ObsEclLon", "ObsEclLat", "delta", 
               "r", "PABLon", "PABLat", "solar_presence", 
               "RA_3sigma", "DEC_3sigma"]
    
    out = os.path.join(outdir, args.out)
    header = "#T60CTRL2.1  (UT)   **h**m**.*s +**d**m**.*s NPACCW ANGRAD dRA      dDEC    \n"
    with open(out, "w") as f:
        f.write(header)
        df = pd.read_csv(args.JPL, sep=" ")
        for idx, row in df.iterrows():
            # datetime_str like 2022-Dec-24 00:00
            t = row["datetime_str"]
            t_dt = datetime.datetime.strptime(t, "%Y-%b-%d %H:%M")
            # Convert to 2022-12-24T00:00:00
            t = datetime.datetime.strftime(t_dt, "%Y-%m-%dT%H:%M:%S")
            
            # RA_app and DEC_app like 59.412 and 150.59037 
            # without atmospheric corrections. (or airless)
            # RA and DEC are astrometric coordinates.
            # The differences are not ignored (~ a few arcmin).
            ra, dec = row["RA"], row["DEC"]
            # Convert to 19h42m17.1s
            ra, dec = deg2hmsdms(ra, dec)

            ra_hm = ra[:6] 
            ra_s = float(ra[6:-1])
            # Round sec
            ra_s = f"{ra_s:04.1f}s"
            ra = ra_hm + ra_s
            assert len(ra) == 11, ra
            
            # DEC in JPL includes +/-
            dec_dm = dec[:7] 
            dec_s = float(dec[7:-1])
            # Round sec
            dec_s = f"{dec_s:04.1f}s"
            dec = dec_dm + dec_s
            assert len(dec) == 12, dec

            # Rate of the motion in arcsec/h
            # The ra_rate and dec_rate are sky motion in ra and dec coordinates.
            # We should do cosine correction to know the rate of motion in RA and DEC.
            # By default, RA_rate = dRA*cos(DEC) in JPL.
            ra_rate = row["RA_rate"]/np.cos(np.radians(row["DEC"]))
            dec_rate = row["DEC_rate"]
            ra_rate, dec_rate = f"{ra_rate:+08.2f}", f"{dec_rate:+08.2f}"
            assert len(ra_rate) == 8, ra_rate
            assert len(dec_rate) == 8, dec_rate
            
            # NPACCW (6 includes +/-)
            # TODO: What is NPACCW?
            NPACCW = 99
            NPACCW = f"{NPACCW:+06.1f}"
            assert len(NPACCW) == 6, NPACCW

            # ANGRAD: Position angle in radian (6)
            # In degree
            ANG = row["velocityPA"]
            ANGRAD = np.radians(ANG)
            ANGRAD = f"{ANGRAD:06.1f}"

            assert len(ANGRAD) == 6, ANGRAD
    
            line = f"{t} {ra} {dec} {NPACCW} {ANGRAD} {ra_rate} {dec_rate}\n"
            print(line)
            assert len(line) == 77, line
            f.write(line)
