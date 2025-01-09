# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make some circuits with resistors in series, parrallell and mix for 1rst year electronics students

"""

__author__ = "Kyle Vitautas Lopin"


import schemdraw
from schemdraw import elements as elm


# make series circuits
with schemdraw.Drawing() as d:
    d += elm.Resistor().down().label('2.2kΩ')
    d += elm.Resistor().down().label('3.3kΩ')
    d += elm.Resistor().down().label('5.1kΩ')


# # parallel circuits
# with schemdraw.Drawing() as d:
#     d += elm.Resistor().down().label('750Ω')
#     d += elm.Line().right()
#     d += elm.Resistor().up().label('500Ω')
#     d += elm.Line().left()


# parallel circuits
# with schemdraw.Drawing() as d:
#     d += elm.Line().left()
#     d += elm.Resistor().down().label('1.2kΩ', loc='bottom')
#     d += elm.Line().right()
#     d += elm.Resistor().up().label('1.2kΩ')
#     d += elm.Line().right()
#     d += elm.Resistor().down().label('5Ω')
#     d += elm.Line().left()

d.draw()
