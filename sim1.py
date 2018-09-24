from wire import *

sim = Simulation(Wire(
    Segment.circle(),
    Segment.circle().rot_x(90).move((1,0,0)),

))

sim.visualize(path="out.html",embedded=True)

