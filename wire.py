import numpy as np
from htmltools import javascript_link, image_link


def Rx(a):
    a*=np.pi/180
    return np.array([[1,0,0],[0,np.cos(a),np.sin(a)],[0,-np.sin(a),np.cos(a)]],np.double)
def Ry(a):
    a*=np.pi/180
    return np.array([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]],np.double)
def Rz(a):
    a*=np.pi/180
    return np.array([[np.cos(a),np.sin(a),0],[-np.sin(a),np.cos(a),0],[0,0,1]],np.double)

class Segment:
    def __init__(self,current=1.0,*points):
        self.points = []
        self.current=current
        for p in points:
            self.add(p)

    def add(self,point):
        x,y,z = point
        self.points.append(np.array([x,y,z],np.double))
        return self
    def __len__(self):
        return len(self.points)
    def __getitem__(self, item):
        return self.points[item]
    def discretize(self,max_d=0.1):
        segment = Segment(self.current)
        for i in range(len(self)):
            if i==0:
                segment=segment.add(self[0])
            else:
                a = self[i-1]
                b = self[i]
                dr = b-a
                d = np.sqrt(np.dot(dr,dr))
                if d<=max_d:
                    segment = segment.add(b)
                else:
                    N=int(d/max_d)+1
                    for j in range(N):
                        fraction = float(j+1)/N
                        v = a+fraction*dr
                        segment = segment.add(v)
        return segment

    def move(self,v):
        x,y,z=v
        v=np.array([x,y,z],np.double)
        segment = Segment(self.current)
        for w in self.points:
            segment=segment.add(w+v)
        return segment

    def transform(self,m):
        segment = Segment(self.current)
        for w in self.points:
            segment=segment.add(np.dot(m,w))
        return segment

    def rot_x(self, a):
        return self.transform(Rx(a))
    def rot_y(self, a):
        return self.transform(Ry(a))
    def rot_z(self, a):
        return self.transform(Rz(a))

    @classmethod
    def circle(cls,current=1.0,radius=1.0,N=16):
        segment = Segment(current)
        for i in range(N+1):
            a = i*2*np.pi/N
            segment=segment.add((radius*np.cos(a),radius*np.sin(a),0))
        return segment
    @classmethod
    def square(cls,current=1.0,radius=1.0):
        return Segment(current,(-radius,-radius,0),(-radius,radius,0),(radius,radius,0),(radius,-radius,0),(-radius,-radius,0))

    def reverse(self):
        segment = Segment(self.current)
        for i in range(len(self)):
            segment = segment.add(self[len(self)-1-i])
        return segment

    def clone(self):
        segment = Segment(self.current)
        for w in self.points:
            segment = segment.add(w)
        return segment

    def tube(self,radius=0.1):
        if self.current>0:
            material = 'material="color: red" '
        else:
            material = 'material="color: blue" '
        path = ", ".join("%f %f %f"%(p[0],p[1],p[2]) for p in self.points)
        return '<a-tube path="%(path)s" radius="%(radius)f" %(material)s></a-frame>\n'%locals()

    def elements(self):
        x = np.array([self[i][0] for i in range(len(self) - 1)], np.double)
        y = np.array([self[i][1] for i in range(len(self) - 1)], np.double)
        z = np.array([self[i][2] for i in range(len(self) - 1)], np.double)

        dx = np.array([self[i + 1][0] - self[i][0] for i in range(len(self) - 1)], np.double)
        dy = np.array([self[i + 1][1] - self[i][1] for i in range(len(self) - 1)], np.double)
        dz = np.array([self[i + 1][2] - self[i][2] for i in range(len(self) - 1)], np.double)
        c = float(self.current)*np.ones(len(self)-1,np.double)
        return x,y,z,dx,dy,dz,c

class Wire:
    def __init__(self,*segments):
        self.segments=segments
    def tube(self,radius=0.1):
        return "".join(segment.tube(radius=radius) for segment in self.segments)
    def elements(self):
        all_x=[]
        all_y=[]
        all_z=[]
        all_dx=[]
        all_dy=[]
        all_dz=[]
        all_c=[]
        for segment in self.segments:
            x,y,z,dx,dy,dz,c = segment.elements()
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
            all_dx.extend(dx)
            all_dy.extend(dy)
            all_dz.extend(dz)
            all_c.extend(c)
        return (
            np.array(all_x),
            np.array(all_y),
            np.array(all_z),
            np.array(all_dx),
            np.array(all_dy),
            np.array(all_dz),
            np.array(all_c)
        )
    def discretize(self,max_d=0.1):
        return Wire(*[segment.discretize(d) for segment in self.segments])
    def calculator(self):
        return Calculator(*self.elements())

class Calculator:
    def __init__(self,x,y,z,dx,dy,dz,c):
        self.data = (x,y,z,dx,dy,dz,c)

    def point_B(self,px,py,pz):
        x,y,z,dx,dy,dz,c=self.data

        rx = px - x
        ry = py - y
        rz = pz - z
        r2 = rx * rx + ry * ry + rz * rz

        lxr_x = dy * rz - dz * ry
        lxr_y = dz * rx - dx * rz
        lxr_z = dx * ry - dy * rx

        cx = c * lxr_x / r2
        cy = c * lxr_y / r2
        cz = c * lxr_z / r2
        Bx = np.sum(cx)
        By = np.sum(cy)
        Bz = np.sum(cz)

        return Bx,By,Bz

    def grid_components_B(self,px,py,pz):
        segment_x, segment_y, segment_z, segment_dx, segment_dy, segment_dz, segment_c = self.data
        Bx = np.zeros(px.shape, np.double)
        By = np.zeros(px.shape, np.double)
        Bz = np.zeros(px.shape, np.double)

        for i in range(len(segment_x)):
            x = segment_x[i]
            y = segment_y[i]
            z = segment_z[i]
            dx = segment_dx[i]
            dy = segment_dy[i]
            dz = segment_dz[i]
            current = segment_c[i]
            rx = px - x
            ry = py - y
            rz = pz - z
            r2 = rx * rx + ry * ry + rz * rz
            r2[r2 == 0] = 1e-6

            #    dy dz dx dy
            #    ry rz rx ry
            lxr_x = dy * rz - dz * ry
            lxr_y = dz * rx - dx * rz
            lxr_z = dx * ry - dy * rx

            cx = current * lxr_x / r2
            cy = current * lxr_y / r2
            cz = current * lxr_z / r2
            Bx += cx
            By += cy
            Bz += cz

        return Bx, By, Bz
    def grid_B(self,grid,name=None):
        Bx,By,Bz = self.grid_components_B(grid.x,grid.y,grid.z)
        if name is None:
            name = 'B_'+str(grid.name)
        return Grid(Bx,By,Bz,name=name)

class Grid:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

    def zero(self):
        return self.scale(0.0)

    def clone(self):
        return self.scale(1.0)

    def scale(self,scale):
        return Grid(self.x*scale,self.y*scale,self.z*scale)

    def move(self,v):
        vx,vy,vz=v
        return Grid(self.x+vx,self.y+vy,self.z+vz)

    def transform(self,m):
        return Grid(
            m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z,
        )
    def rot_x(self,a):
        return self.transform(Rx(a))
    def rot_y(self,a):
        return self.transform(Ry(a))
    def rot_z(self,a):
        return self.transform(Rz(a))

    @classmethod
    def plane(self,x1,y1,x2,y2,level=0.0,nx=100,ny=100):
        span_x = np.linspace(x1, x2, nx)
        span_y = np.linspace(y1, y2, ny)
        x=span_x.repeat(ny).reshape((nx, ny))
        y=span_y.repeat(nx).reshape((ny, nx)).T
        z=level*np.ones((nx,ny),np.double)
        return Grid(x,y,z)

    @classmethod
    def space(self,x1,y1,z1,x2,y2,z1,z2,nx=100,ny=100,nz=100):
        span_z = np.linspace(z1, z2, nz)
        x=[]
        y=[]
        z=[]
        for level in span_z:
            g=Grid.plane(x1,y1,x2,y2,level=level,nx=nx,ny=ny)
            x.append(g.x)
            y.append(g.y)
            z.append(g.z)
        return Grid(np.array(x),np.array(y),np.array(z),name=name)

class Gradient:
    def __init__(self,*vrgba):
        self.vrgba=[]
        for entry in vrgba:
            v=float(entry[0])

        self.v=np.array([entry[0] for entry in vrgba],np.double)
        self.r=np.array([entry[1] for entry in vrgba],np.double)
        self.g=np.array([entry[2] for entry in vrgba],np.double)
        self.b=np.array([entry[3] for entry in vrgba],np.double)
        self.a=np.array([entry[4] for entry in vrgba],np.double)

class GridLines:
    def __init__(self,grid,scale=1.0):
        self.grid=grid
        self.scale=scale
        self.field = grid.zero()
    def calculate(self,calculator):
        self.field = calculator.grid_B(self.grid)

class Simulation:
    def __init__(self,wire):
        self.wire=wire
        self.calculator = wire.calculator()

    def head(self,title,embedded=False):
        return """
        <meta charset="utf-8">
        <title>%s</title>
        <meta name="description" content="%s">
        %s
        %s
        %s
        """ % (
            title,title,
            javascript_link("https://aframe.io/releases/0.8.2/aframe.min.js", embedded=embedded),
            javascript_link("https://cdn.rawgit.com/donmccurdy/aframe-extras/v4.1.3/dist/aframe-extras.min.js",
                            embedded=embedded),
            javascript_link("https://unpkg.com/aframe-orbit-controls@1.2.0/dist/aframe-orbit-controls.min.js",
                            embedded=embedded)
        )

    def visualize(self,title="pymagf",path=None,embedded=False,wire_radius=0.1):
        head = self.head(title,embedded=embedded)
        tube = self.wire.tube(radius=wire_radius)
        html="""<!DOCTYPE html>
<html>
  <head>
%(head)s
  </head>
  <body>
    <a-scene background="color: #000011">
    <a-assets>
    </a-assets>
%(tube)s
    <a-entity camera look-controls orbit-controls="target: 0 1.6 -0.5; minDistance: 0.5; maxDistance: 180; initialPosition: 0 5 15"></a-entity>
    </a-scene>
  </body>
</html>""" % locals()
        if path is not None:
            with open(path,"w") as f:
                f.write(html)
        return html
