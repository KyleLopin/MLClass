# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make current mirror for students to solve load current for
"""

__author__ = "Kyle Vitautas Lopin"
import schemdraw
import schemdraw.elements as elm
FILENAME = "current_mirror_start_2_bjt.svg"

with schemdraw.Drawing(file=FILENAME) as d:
    d += elm.Vdd().label("$V_{in}$")
    d += (ref_res := elm.Resistor().down().label("$R_{ref}$"))
    d += (bjt := elm.BjtNpn().right().anchor("collector").reverse())
    d += elm.Line().right(0.3).at(bjt.base)
    d.push()
    d += elm.Line().up(0.8)
    d += elm.Line().tox(bjt.collector)

    d += elm.CurrentLabel().at(ref_res).label("$I_{ref}$", loc="bot")
    d.pop()
    d += elm.Line().right(0.3)
    d += (bjt2 := elm.BjtNpn().right())
    d += (r_l := elm.ResistorIEC().up().label("$R_L$"))
    d += elm.CurrentLabel(ofst=0.7).at(r_l).label("$I_{L}$", loc="bot", ofst=0.3).reverse().flip()
    d += elm.Vdd().label("$V_{in}$")
    d += elm.Ground().at(bjt2.emitter)


    d += elm.Ground().at(bjt.emitter)

# d.draw()
d.save(fname=FILENAME)
