# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make schematic to illustrate a difference amplifier
"""

__author__ = "Kyle Vitautas Lopin"


import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing(file="difference_amplifier.svg") as d:
    # add opamp and top resistors
    d += (op := elm.Opamp(leads=True))
    d += (out := elm.Line(at=op.out).length(.75).label('$V_{out}$', loc='right'))
    d += elm.Line().up().at(op.in1).length(1.5).dot()
    d.push()
    d += elm.Resistor().left().label('R').label('$V_{1}$', loc='left')
    d.pop()
    d += elm.Resistor().tox(op.out).label('$R$')
    d += elm.Line().toy(op.out).dot()
    # add bottom resistors
    d += elm.Line().down().at(op.in2).length(1.5).dot()
    d.push()
    d += elm.Resistor().left().label('$R$').label('$V_{2}$', loc='left')
    d.pop()
    d += elm.Resistor().tox(op.out).label('$R$')
    d += elm.Ground()


d.draw()
# d.save()
