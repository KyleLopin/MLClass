# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make a simple band pass filter
"""

__author__ = "Kyle Vitautas Lopin"

import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="band_pass_filter.svg") as d:
    d += elm.Capacitor().right().label('C1').label('$V_{in}$', loc='left')
    d.push()
    d += elm.Resistor().down().label("R1")
    d.pop()
    d += elm.Resistor().right().label('R2').label('$V_{out}$', loc='right')
    d += elm.Capacitor().down().label("C2")
    d += elm.Line().left()
    d.push()
    d += elm.Ground()
    d.pop()
    d += elm.Line().left()

d.save()

