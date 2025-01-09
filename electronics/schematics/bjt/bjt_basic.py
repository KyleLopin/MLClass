# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make a schematic to explore there relationship between the
collector-emitter Voltage and the collector current
with the base current
"""

__author__ = "Kyle Vitautas Lopin"

import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="bjt_basic.svg") as d:
    d += elm.SourceI().right().label("$I_b$")
    d += (bjt := elm.BjtNpn().right())
    d += (r1 := elm.Resistor().up().label('1kΩ', loc="bottom"))
    d += elm.Vdd().label("V1 (variable)")
    d += elm.Ground().at(bjt.emitter)
    d += elm.CurrentLabel(reverse=True).at(r1).label("$I_c$")


# d.save(fname="bjt_basic")
d.draw()
