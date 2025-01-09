# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make a simple resistor into a current source for students to make a current mirror
current source
"""

__author__ = "Kyle Vitautas Lopin"
import schemdraw
import schemdraw.elements as elm
FILENAME = "current_source_simple.svg"

with schemdraw.Drawing() as d:
    d += elm.Vdd().label("10 V")


    d += elm.Resistor().down().label("$R_L$")
    d += elm.SourceI().down().label("1 mA")
    d += elm.Ground()

# d.draw()
d.save(fname=FILENAME)

