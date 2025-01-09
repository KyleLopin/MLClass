# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make circuit for student to calculate the re of a bjt
"""

__author__ = "Kyle Vitautas Lopin"
import schemdraw
import schemdraw.elements as elm
FILENAME = "bjt_re.svg"

with schemdraw.Drawing() as d:
    d += elm.Line().right(0.3).label("$V_{be}$", loc="left")
    d += (bjt := elm.BjtNpn().right())
    d += (res := elm.Resistor().up().label('1kΩ', loc="bot"))
    d += elm.CurrentLabel().at(res).label("$I_c$", loc="top").reverse()
    d += elm.Vdd().label("5 V")

    d += elm.Ground().at(bjt.emitter)

# d.draw()
d.save(fname=FILENAME)
