# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make 2 circuits to introduce diodes to first semester
ab students, 1 for DC characteristics and 1 for simple
half wave rectifier
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import schemdraw
from schemdraw import elements as elm

# filename = "dc_diodes.svg"
# with schemdraw.Drawing(file=filename) as d:
#     d += elm.Vdd().label("$V_{in}$")
#     d += elm.Diode().down()
#     d += elm.Resistor().down().label('1 kΩ')
#     d += elm.Ground()

# filename = "ac_diodes_basic_1.svg"
# with schemdraw.Drawing(file=filename) as d:
#     d += elm.SourceSin().up().label("$8 V_{p-p}$")
#     d += elm.Diode().right()
#     # d.push()
#     # d += elm.Capacitor().down().label("1 μF")
#     # d += elm.Line().left()
#     # d.pop()
#     # d += elm.Line().right()
#     d += elm.Resistor().down().label('1 kΩ')
#     d += elm.Line().left()

filename = "solar_parallel.svg"
with schemdraw.Drawing(file=filename) as d:
    d += elm.Solar()
    d += elm.Line().right()
    d += elm.Line().right()
    d += elm.Resistor().down().label("$R_1$")
    d += elm.Line().left()
    d.push()
    d += elm.Solar().up()
    d.pop()
    d += elm.Line().left()

d.show()
d.save(fname=filename)
