# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make schematic to measure LED turn on voltage
"""

__author__ = "Kyle Vitautas Lopin"

import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="led_turn_on_voltage.svg") as d:
    d += elm.Vdd().label('$V_{in}$')
    d.push()  # save to make voltmeter for resistor
    d += elm.Resistor().down().label('$R_1$')

    # add voltmeter to side of resistor
    d.pop()
    d += elm.Line().right()
    d += elm.MeterV().down()
    d += elm.Line().left()

    # make LED
    d.push()
    d += elm.LED().label("LED").down()

    # add voltmeter to side of LED
    d.pop()
    d += elm.Line().right()
    d += elm.MeterV().down()
    d += elm.Line().left()

    d += elm.Ground()


d.draw()
d.save()
