# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import schemdraw
from schemdraw import elements as elm



# make series circuits 1
# filename = "DMM_basic_1.svg"
# with schemdraw.Drawing(file=filename) as d:
#     d += (power_supply := elm.Battery().up().label('10V').reverse())
#     d += elm.Line().up(1)
#     d += elm.Line().right()
#     d += elm.Resistor().down().label('$R_1$\n500Ω')
#     d += elm.Resistor().down().label('$R_1$\n500Ω')
#     d += elm.Line().left()
#     d += elm.Line().up().toy(power_supply.start)


# make series circuits 2
# filename = "DMM_basic_2.svg"
# with schemdraw.Drawing(file=filename) as d:
#     d += elm.Vdd().label("6 V")
#     d += elm.Resistor().down().label('$R_1$\n1 kΩ')
#     d += elm.Resistor().down().label('$R_2$\n500 Ω')
#     d += elm.Ground()


# make parallel circuit 1
# filename = "DMM_basic_parallel.svg"
# with schemdraw.Drawing(file=filename) as d:
#     d += (power_supply := elm.Battery().up().label('10V').reverse())
#     d += (positive_rail := elm.Line().right(6))
#     d += elm.Resistor().down().label('$R_1$\n1 kΩ')
#     d += elm.Line().left(3)
#     d.push()
#     d += elm.Resistor().up().label("$R_2$\n500 Ω")
#     d.pop()
#     d.push()
#     d += elm.Ground()
#     d.pop()
#     d += elm.Line().left(3)

# make parallel circuit 1
filename = "DMM_basic_series_and_parallel.svg"
with schemdraw.Drawing(file=filename) as d:
    d += elm.Ground()
    d += elm.Resistor().up().label('$R_3$\n500 Ω')
    d.push()
    d += elm.Line().right(1)
    d += elm.Resistor().up().label('$R_1$\n1 kΩ')
    d += elm.Line().left(1)
    d += elm.Vdd().label("10 V")

    d.pop()
    d += elm.Line().left(1)
    d += elm.Resistor().up().label('$R_2$\n1 kΩ')
    d += elm.Line().right(1)


 #d.save(fname=filename)
d.draw()
