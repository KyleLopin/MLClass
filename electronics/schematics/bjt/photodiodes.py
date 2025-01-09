# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Set of schematics to make an IR Tx and Rx
"""

__author__ = "Kyle Vitautas Lopin"

import schemdraw
import schemdraw.elements as elm

FILENAME = "photodiode_rx3.svg"

with schemdraw.Drawing(file=FILENAME) as d:
    # ======   photodiode_tx ======
    # d += elm.Vdd().label("5V")
    # d += elm.Button().down()
    # d += elm.Resistor().down().label("220 Ω")
    # d += elm.LED().down()
    # d += elm.Ground()

    # # ======   photodiode_rx ======
    d += elm.Vdd().label("5V")
    d.push()
    d += elm.Line(0.8).left()
    d += elm.Photodiode().down().reverse()
    d += elm.Line(0.8).right()
    d += (bjt := elm.BjtNpn())
    d.pop()
    d += elm.Line().right().tox(bjt.collector)
    d += elm.Line().down().to(bjt.collector)

    d += elm.Resistor().down().label("5.6k Ω").at(bjt.emitter)
    d += elm.Ground()
    d += elm.Line(0.5).right().at(bjt.emitter)
    d += elm.LED().down()
    d += elm.Resistor().down().label("200 Ω")

    d += elm.Ground()


d.save(fname=FILENAME)
# d.draw()
