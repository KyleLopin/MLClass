# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make schematics for the circuits for studying super positions in circuits.
Only use 2 voltage supplies for lab.
"""

__author__ = "Kyle Vitautas Lopin"


import schemdraw
from schemdraw import elements as elm
# Set the backend to matplotlib for PNG or PDF output
schemdraw.use('matplotlib')

# filename = "superposition_1.svg"
# with schemdraw.Drawing(file=filename) as d:
#     d += elm.Battery().up().reverse().label("Supply 1\n9 V")
#     d += elm.Resistor().right().label("$R_1$\n150 Ω")
#     d += elm.Resistor().down().label("$R_2$\n300 Ω")
#     d.push()
#     d += elm.Line().left()
#     d.pop()
#     d += elm.Line().right()
#     d += elm.Battery().up().reverse().label("Supply 2\n8 V")
#     d += elm.Resistor().left().label("$R_3$\n300 Ω")


filename = "superposition_2.svg"
with schemdraw.Drawing(file=filename) as d:
    d += elm.Battery().up().reverse().label("Supply 1\n10.8 V")
    d += elm.Resistor().up().label("$R_1$\n30k Ω")
    d += elm.Line().right()
    d.push()
    d += elm.Resistor().down().label("$R_2$\n10k Ω")
    d += elm.Battery().down().label("Supply 2\n10 V")
    d += elm.Line().left()
    d.pop()
    d += elm.Line().right()
    d += elm.Resistor().down().label("$R_3$\n15k Ω")
    d += elm.Line().down()
    d += elm.Line().left()


d.save(fname=filename)
# d.draw()
