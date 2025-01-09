# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Build schematic with 100 ohm resistor in series with ammeter
"""

__author__ = "Kyle Vitautas Lopin"


import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="low_pass_filter.svg") as d:
    d += elm.Resistor().down().label("100$\Omega$").label('$V_{in}$', loc='right')
    d += elm.MeterA().down()
    d += elm.Ground()

d.draw()
