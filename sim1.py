from wire import *

sim = Simulation(
    Wire(
        Segment.circle(),
        Segment.circle().rot_x(20).move((1,0,0)).reverse(),
    ),
    #FieldLines(Grid.cube(2,15),scale=0.1)
    #DensitySample(5,100)
    DensityFieldLines(5,lines=200,scale=0.05,segments=20)
)

sim.visualize(path="out.html",embedded=True)

