# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make a schematic with a simple RC high pass filter
"""

__author__ = "Kyle Vitautas Lopin"


import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="high_pass_filter.svg") as d:
    d += elm.Capacitor().right().label('1$\mu$F').label('$V_{in}$', loc='left').label('$V_{out}$', loc='right')
    d += elm.Resistor().down().label("1k$\Omega$")
    d += elm.Ground()

d.draw()
