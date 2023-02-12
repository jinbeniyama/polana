#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Color photmetry plot functions
"""

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
