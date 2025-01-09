# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Build the common emitter bjt circuit with an input AC signal
"""

__author__ = "Kyle Vitautas Lopin"

import schemdraw
import schemdraw.elements as elm
from schemdraw import dsp

with schemdraw.Drawing(file="bjt_basic.svg") as d:
    # d += elm.SourceI().right().label("$I_b$")
    # d += elm.Vdd().label("5V")
    # make the DC base bias resistors
    d += (vcc_line := elm.Line().left().label("5V"))
    d += elm.Resistor().down().label("Rb1")
    d.push()
    d += (rb2 := elm.Resistor().down().label("Rb2"))
    d += elm.Ground()
    d += elm.Line().right(2).at(rb2.start)
    d += (bjt := elm.BjtNpn().right())
    d += (rc := elm.Resistor().up().at(bjt.collector).label('Rc', loc="bottom")).toy(vcc_line.start)

    d += (re := elm.Resistor().at(bjt.emitter).down().label('Re', loc="bottom"))
    d += elm.Ground()
    d.pop()
    d += elm.Capacitor().left().label("$C_{blocking}$")
    d += dsp.Oscillator().left().label("Vin", loc='top')

    d += elm.Line().right().at(bjt.emitter)
    d += elm.Resistor().down().label("$R_L$")
    d += elm.Ground()
    # d += elm.CurrentLabel(reverse=True).at(r1).label("$I_c$")


d.save(fname="bjt_vin_2.svg")
# d.draw()

