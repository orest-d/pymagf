from wire import *

sim = Simulation(
    Wire(
        Segment.circle().rot_x(-15).move((-0.6, 0, 0)),
        Segment.circle().rot_x(15).move((0.6, 0, 0)).reverse(),
    ),
    #FieldLines(Grid.cube(2,15),scale=0.1)
    #DensitySample(5,100)
    DensityFieldLines(5,lines=300,scale=0.05,segments=20),
    PlaneTest(10,400)
)

sim.visualize(path="out.html",embedded=True)

