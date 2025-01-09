# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_xlim([10, 10000])
ax.set_ylim([-40, 5])
plt.grid()
plt.grid(b=True, which='minor', linestyle='--')
plt.ylabel("Decibel (db)")
plt.xlabel("Frequency (Hz)")
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig("Blank_bode")
plt.show()
