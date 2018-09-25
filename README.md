# pymagf
Simple simulator of a magnetic field in python. It's a toy, I don't plan to develop it further.
But it works quite well.

Magnetic around a system of wires can be simulated and a-frame (WebGL) visualization constructed
using a simple DSL:

Segment is a continuous wire. Wire is a collections of segments.
Wires can be added and both segments and wires support simple geometric manipulations (translations and rotations).
A Simulation needs one Wire and a collection of calculators.

##Example
See sim1.py
https://orest-d.github.io/pymagf/index.html
