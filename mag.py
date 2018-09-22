# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import PIL


segment_x=[]
segment_y=[]
segment_z=[]
segment_dx=[]
segment_dy=[]
segment_dz=[]
segment_c=[]

vectors=[]
R=2.0
N=40
span = np.arange(N+1)
x=R*np.cos(2*span*np.pi/N)
y=R*np.sin(2*span*np.pi/N)
z=np.zeros(len(x),np.double)

def circle(R=1.0,N=8,charge=1):
    span = np.arange(N+1)
    x=R*np.cos(2*span*np.pi/N)
    y=R*np.sin(2*span*np.pi/N)
    z=np.zeros(len(x),np.double)
    c=np.ones(len(x)-1,np.double)*charge
    return x[:-1],y[:-1],z[:-1],x[1:]-x[:-1],y[1:]-y[:-1],z[1:]-z[:-1],c

def offset(shape,ox=0.0,oy=0.0,oz=0.0):
    x,y,z,dx,dy,dz,c=shape
    return x+ox,y+oy,z+oz,dx,dy,dz,c

def join(shape1,shape2):
    x1,y1,z1,dx1,dy1,dz1,c1=shape1
    x2,y2,z2,dx2,dy2,dz2,c2=shape2
    x=np.hstack((x1,x2))
    y=np.hstack((y1,y2))
    z=np.hstack((z1,z2))
    dx=np.hstack((dx1,dx2))
    dy=np.hstack((dy1,dy2))
    dz=np.hstack((dz1,dz2))
    c=np.hstack((c1,c2))
    return x,y,z,dx,dy,dz,c



for ox,c in [(-2,1),(2,-1)]:
    segment_x+=list(x[:-1]+ox)
    segment_y+=list(y[:-1])
    segment_z+=list(z[:-1])
    segment_c+=list(np.ones(len(x)-1,np.double)*c)
    segment_dx+=list(x[1:]-x[:-1])
    segment_dy+=list(y[1:]-y[:-1])
    segment_dz+=list(z[1:]-z[:-1])

shape = segment_x,segment_y,segment_z,segment_dx,segment_dy,segment_dz,segment_c
print(len(segment_x))

def make_aframe(shape):
    return """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>MagVR</title>
    <meta name="description" content="MagVR">
    <script src="https://aframe.io/releases/0.8.2/aframe.min.js"></script>
    <script src="https://cdn.rawgit.com/donmccurdy/aframe-extras/v4.1.3/dist/aframe-extras.min.js"></script>
    <script src="https://unpkg.com/aframe-orbit-controls@1.2.0/dist/aframe-orbit-controls.min.js"></script>    
  </head>
  <body>
    <a-scene background="color: #000011">
    <a-assets>
      <img id="B1" src="B.png">
      <img id="B2" src="B2.png">
    </a-assets>
%s
    <a-plane position="0 0 0" src="#B1" rotation="0 0 0" width="20" height="20" color="#7BC8A4" material="side: double; transparent: false; alphaTest: 0.5;"></a-plane>
    <a-plane position="0 0 0" src="#B2" rotation="90 0 0" width="20" height="20" color="#7BC8A4" material="side: double; transparent: false; alphaTest: 0.5;"></a-plane>
    <a-entity camera look-controls orbit-controls="target: 0 1.6 -0.5; minDistance: 0.5; maxDistance: 180; initialPosition: 0 5 15"></a-entity>
    </a-scene>
  </body>
</html>"""%(shape_as_tube(shape))

def shape_as_tube(shape,radius=0.1):
    x,y,z,dx,dy,dz,c=shape
    s=""
    for i in range(len(x)):
        color = "red" if c[i]>0 else "blue"
        r1 ="%f %f %f"%(x[i],y[i],z[i])
        r2 ="%f %f %f"%(x[i]+dx[i],y[i]+dy[i],z[i]+dz[i])
        s+='      <a-tube path="%(r1)s, %(r2)s" radius="%(radius)s" material="color: %(color)s"></a-tube>\n'%locals()
    return s

span_x = np.linspace(-10,10,400)
span_y = np.linspace(-10,10,400)

plane_u = span_x.repeat(len(span_y)).reshape((len(span_x),len(span_y)))
plane_v = span_y.repeat(len(span_x)).reshape((len(span_y),len(span_x))).T
plane_w = np.zeros(plane_u.shape,np.double)

print(plane_u.shape,plane_v.shape)

plane_x = plane_u
plane_y = plane_v
plane_z = plane_w

def calculateB(plane_x,plane_y,plane_z):
    Bx= np.zeros(plane_x.shape,np.double)
    By= np.zeros(plane_x.shape,np.double)
    Bz= np.zeros(plane_x.shape,np.double)
    
    for i in range(len(segment_x)):
        x=segment_x[i]
        y=segment_y[i]
        z=segment_z[i]
        dx=segment_dx[i]
        dy=segment_dy[i]
        dz=segment_dz[i]
        current = segment_c[i]
        rx=plane_x-x
        ry=plane_y-y
        rz=plane_z-z
        r2=rx*rx+ry*ry+rz*rz
        r2[r2==0]=1e-6
        
    #    dy dz dx dy
    #    ry rz rx ry
        lxr_x = dy*rz-dz*ry
        lxr_y = dz*rx-dx*rz
        lxr_z = dx*ry-dy*rx
        
        cx = current*lxr_x/r2
        cy = current*lxr_y/r2
        cz = current*lxr_z/r2
        Bx+=cx
        By+=cy
        Bz+=cz
    
    B=np.sqrt(Bx*Bx+By*By+Bz*Bz)
        
    return Bx,By,Bz,B

def calculateBscalar(px,py,pz):
    Bx=0
    By=0
    Bz=0
    
    for i in range(len(segment_x)):
        x=segment_x[i]
        y=segment_y[i]
        z=segment_z[i]
        dx=segment_dx[i]
        dy=segment_dy[i]
        dz=segment_dz[i]
        current = segment_c[i]
        rx=px-x
        ry=py-y
        rz=pz-z
        r2=rx*rx+ry*ry+rz*rz

        lxr_x = dy*rz-dz*ry
        lxr_y = dz*rx-dx*rz
        lxr_z = dx*ry-dy*rx
        
        cx = current*lxr_x/r2
        cy = current*lxr_y/r2
        cz = current*lxr_z/r2
        Bx+=cx
        By+=cy
        Bz+=cz
    
    B=np.sqrt(Bx*Bx+By*By+Bz*Bz)
        
    return Bx,By,Bz,B

Bx,By,Bz,B = calculateB(plane_x,plane_y,plane_z)
Bxy=np.sqrt(Bx*Bx+By*By)

def absimage(gridval,maxval=None,transparent=None, transparentfactor=0.01):
    g=gridval.copy().T
    g[g<0]=0
    if maxval is None:
        maxval = np.max(g)
    g[g>maxval]=maxval
    if transparent is None:
        transparent = transparentfactor*maxval
    transparentindex = g<transparent
    g*=255/maxval
    g=np.array(g,np.uint8)
    a=np.zeros((g.shape[0],g.shape[1],4),np.uint8)
    a[:,:,0]=g
    a[:,:,1]=g
    a[:,:,2]=g
    a[:,:,3]=255-255*transparentindex
    return PIL.Image.fromarray(a,"RGBA")
maxval=calculateBscalar(-1,0,0)[3]
absimage(B).save("B.png",maxval=maxval)

plane_x = plane_u
plane_y = plane_w
plane_z = plane_v

Bx,By,Bz,B = calculateB(plane_x,plane_y,plane_z)

absimage(B).save("B2.png",maxval=maxval)


with open("out.html","w") as f:
    f.write(make_aframe(shape))
    
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
ax1.plot(segment_x,segment_y)
ax1.plot(np.array(segment_x)+np.array(segment_dx),np.array(segment_y)+np.array(segment_dy))
#q = ax1.quiver(plane_x,plane_z, plane_u, plane_v)
print(Bx.shape,Bz.shape)
#q = ax.quiver(X=plane_u, Y=plane_v,U=Bx,V=Bz)
#ax1.quiverkey(q, X=1.0, Y=1.0, U=1,
#             label='Quiver key, length = 10', labelpos='E')
ax2.imshow(Bx.T)
ax3.imshow(By.T)
ax4.imshow(Bz.T)
ax5.imshow(B.T)
ax6.imshow(Bxy.T)

fig.show()
