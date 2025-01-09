# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make the differential stage of an op amp
"""

__author__ = "Kyle Vitautas Lopin"



import schemdraw
import schemdraw.elements as elm

FILENAME = "bjt_diff_circuit_Isource2.svg"

with schemdraw.Drawing() as d:
    d += elm.Line().right(0.3).label("$V_{in}^+$", loc="left")
    d += (bjt_plus := elm.BjtNpn().right().label("Q1"))

    # d += elm.Resistor().up().label("Rc1").at(bjt_plus.collector)
    # d += (vcc_line := elm.Line().right().label("+5V"))
    # d += elm.Resistor().down().label("Rc2")
    d += elm.SourceI().up().label(r"$\frac{I_{bias}}{2}$").reverse()
    d += (vcc_line := elm.Line().right().label("+5V"))
    d += elm.SourceI().down().label(r"$\frac{I_{bias}}{2}$")

    d += (bjt_neg := elm.BjtNpn().anchor("collector").left().flip().label("Q2", loc="left"))
    d += elm.Line().right(0.3).label("$V_{in}^-$", loc="right").at(bjt_neg.base)
    d += (bottom_line := elm.Line().left().at(bjt_neg.emitter).to(bjt_plus.emitter)).drop("center").label("Ve")
    # toggle from passive to active load
    # d += elm.Resistor().down().label("Re")
    d += elm.SourceI().down().label("$I_{bias}$")

    d += elm.Vss().label("-5V")
    d += elm.Line().right(2).at(bjt_neg.collector).label("$V_{out}$", loc="right")


# d.draw()
d.save(fname=FILENAME)

