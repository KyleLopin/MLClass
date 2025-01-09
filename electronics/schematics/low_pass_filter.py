# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make a schematic with a simple RC high pass filter
"""

__author__ = "Kyle Vitautas Lopin"


import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="low_pass_filter.svg") as d:
    d += elm.Resistor().right().label('1$\mu$F').label('$V_{in}$', loc='top')
    d += elm.Resistor().down().label("1k$\Omega$")
    d += elm.Ground()

d.draw()
