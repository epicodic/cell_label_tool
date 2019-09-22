#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:59:39 2017

@author: epicodic
"""

import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt

import scipy.misc
import sys
import os

from enum import Enum

#import skimage
#from skimage import morphology
#from skimage import io
#from skimage import img_as_float
#from skimage import exposure 

import skimage.draw

#im_raw = io.imread('keepme.tif',mode='RGBA')
import math
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
from matplotlib.patches import Polygon
from matplotlib import colors
#from scipy import interpolate
from scipy.interpolate import CubicSpline




def color_for_class(cl):
    if(cl==1):
        return '#FF00FF'
    else:
        return '#FFFF00'


class PolygonInteractor(object):
    """
    An polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y,  color = 'yellow', marker='o', markerfacecolor='yellow', animated=True)
        self.ax.add_line(self.line)
        #self._update_line(poly)

        #cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.set_class(0)
        self.cid=[]
        self.cid.append( canvas.mpl_connect('draw_event', self.draw_callback) )
        self.cid.append( canvas.mpl_connect('button_press_event', self.button_press_callback) )
        self.cid.append( canvas.mpl_connect('key_press_event', self.key_press_callback) )
        self.cid.append( canvas.mpl_connect('button_release_event', self.button_release_callback) )
        self.cid.append( canvas.mpl_connect('motion_notify_event', self.motion_notify_callback) )

        self.canvas = canvas
        
    def remove(self):
        self.line.remove()
        for i in range(4):
            self.canvas.mpl_disconnect(self.cid[i])
        
    def __del__(self):
        #print ("deling", self)
        pass

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        #self.canvas.blit(self.ax.bbox)
        pass

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        print("Poly Changed")
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        
        if(self.canvas.manager.toolbar._active != None):
            return
        
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':           
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
            
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.array(
                        list(self.poly.xy[:i]) +
                        [(event.xdata, event.ydata)] +
                        list(self.poly.xy[i:]))
                    self.line.set_data(zip(*self.poly.xy))
                    break

        elif event.key == 'k':
            self.set_class(1-self.cl)
        #elif event.key == 'l':
        #    self.set_class(0)

            
        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        
    def set_class(self,cl):
        self.cl = cl
        
        self.line.set_color(color_for_class(self.cl))
        self.line.set_markerfacecolor(color_for_class(self.cl))
        self.poly.set_color(color_for_class(self.cl))
        

class Mode(Enum):
        IDLE = 1
        CREATE = 2
        EDIT = 3

class ImageToggle(Enum):
        FULL = 0
        R = 1
        G = 2
        B = 3
        RN = 4
        
        
class Label:
     def __init__(self):
         self.jd  = None
         self.tx  = None
         self.ty  = None
         self.h   = None
         self.ht  = None
         self.D   = None
         self.cl  = 0

        
class Tool(object):
    
    
        
    def __init__(self, filename):
    
        self.mode = Mode.IDLE
        
        self.filename = filename    
        self.im_raw = scipy.misc.imread(filename,mode='RGBA')
        
        self.im_r = self.im_raw[:,:,0] 
        self.im_r_norm = colors.LogNorm(self.im_r.mean() + 0.5 * self.im_r.std(), self.im_r.max(), clip='True')
        
        self.im_g = self.im_raw[:,:,1] 
        self.im_b = self.im_raw[:,:,2] 
        
        self.img_median = np.array([0,0,0])
        self.img_median[0] = np.median(self.im_r)
        self.img_median[1] = np.median(self.im_g)
        self.img_median[2] = np.median(self.im_b)

        self.fig, self.ax = plt.subplots()
        self.infotext = None
        
        self.fig.tight_layout()
        self.canvas = self.fig.canvas
        
        self.imgh = None
        self.imgtoggle = ImageToggle.FULL
        
        self.update_image()
        
        self.ax.set_ylim(bottom=10000, top=-10000,auto=True)

        self.marks = []
        self.clicks = []
        self.p = None
        self.labels = []
        self.nextjd = 1
        
        self.edit_label = None
        
        
        #canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.canvas.mpl_connect('key_release_event', self.key_release_callback)
        self.canvas.mpl_connect('resize_event', self.resize_callback)
        
        self.load_labels(filename+".txt")
        
        self.update()


    def update_image(self):
        if(self.imgh):
            self.imgh.remove()
        
        if(self.imgtoggle == ImageToggle.FULL):
            self.imgh = self.ax.imshow(self.im_raw)
        elif(self.imgtoggle == ImageToggle.R):
            self.imgh = self.ax.imshow(self.im_r,cmap='gray')
        elif(self.imgtoggle == ImageToggle.RN):
            self.imgh = self.ax.imshow(self.im_r,cmap='gray', norm=self.im_r_norm)
        elif(self.imgtoggle == ImageToggle.G):
            self.imgh = self.ax.imshow(self.im_g,cmap='gray')
        elif(self.imgtoggle == ImageToggle.B):
            self.imgh = self.ax.imshow(self.im_b,cmap='gray')
        self.update()
        
        
    def resize_callback(self, event):
       
        aspect = event.width / event.height
        
        x1,x2 = self.ax.get_xlim()
        y1,y2 = self.ax.get_ylim()
        
        cx = (x1+x2)/2
#        cy = (y1+y2)/2
        
#        w = x2-x1
        h = y1-y2
        
        tw = h*aspect
#        th = w/aspect

        self.ax.set_xlim(left=cx-tw/2, right=cx+tw/2)
     
    def button_press_callback(self, event):
         
        if(self.canvas.manager.toolbar._active != None):
            return
        
        if(self.mode == Mode.EDIT): # we are in ellipse edit mode
            return
        
        if(event.button==3):
            return
            
        if event.dblclick:
            print(event.button)
            self.clear_clicks()
            
            # find clicked label
            clickedLabelIdx = -1
            for i in range(len(self.labels)):
                dx = abs(event.xdata-self.labels[i].tx)
                dy = abs(event.ydata-self.labels[i].ty)
                r = 10
                if(dx<r and dy<r):
                    print("CLICKED LABEL: %d" % self.labels[i].jd)
                    clickedLabelIdx = i
                    break
                
            if(clickedLabelIdx>=0):
                self.mode = Mode.EDIT
                l = self.labels[clickedLabelIdx]
                xx = l.D[0,:]
                yy = l.D[1,:]
                self.add_ellipse_from_poly(xx,yy)
                self.p.set_class(l.cl)
                l.h.remove()
                l.h = None
                l.ht.remove()
                l.ht = None
                self.edit_label = l
                self.update()
            else:
                self.mode = Mode.IDLE
                
            self.update()
            return
        
        self.mode = Mode.CREATE
        
        line = Line2D([event.xdata], [event.ydata], marker='+', color='r')
        event.inaxes.add_line(line)
        self.marks.append(line)
        self.clicks.append((event.xdata, event.ydata))
         
        if(len(self.clicks)>=3):
            if(self.add_ellipse(self.clicks)):
                self.edit_label = None
                self.mode = Mode.EDIT
            else:
                self.mode = Mode.IDLE
            self.clear_clicks()
         
        self.update()
        
    def key_press_callback(self, event):
        
       
        if(event.key==' ' and self.p!=None):
            if(self.edit_label != None):
                self.update_label(self.p.poly.xy.T, self.p.cl, self.edit_label,False)
            else:
                self.add_label(self.p.poly.xy.T, self.p.cl)
            self.p.remove()
            self.p = None
            self.mode = Mode.IDLE
            self.update()
            
        if(event.key=='escape' or event.key=='delete'):
            self.clear_clicks()
            self.mode = Mode.IDLE
            if(self.p!=None):
                self.p.remove()
                self.p = None
                if(self.edit_label != None):
                    if(event.key=='delete'):
                        self.delete_label(self.edit_label)
                    else:
                        self.update_label(None, self.p.cl,self.edit_label,True)
                self.update()
                
        if(event.key=='r' or event.key=='v' or event.key=='x'):
            self.imgtoggle = ImageToggle.FULL if self.imgtoggle==ImageToggle.R else ImageToggle.R
            self.update_image()
#        if(event.key=='v' or event.key=='x'):
#            self.imgtoggle = ImageToggle.FULL if self.imgtoggle==ImageToggle.RN else ImageToggle.RN
#            self.update_image()
        if(event.key=='g'):
            self.imgtoggle = ImageToggle.FULL if self.imgtoggle==ImageToggle.G else ImageToggle.G
            self.update_image()
        if(event.key=='b'):
            self.imgtoggle = ImageToggle.FULL if self.imgtoggle==ImageToggle.B else ImageToggle.B
            self.update_image()
            
            
        if(event.key=='e'):    
            self.evaluate()
            
    def key_release_callback(self, event):    
        self.update()
             
    def clear_clicks(self):
        for mark in self.marks:
            mark.remove()
        self.marks = []
        self.clicks = []
        self.update()
        
    def add_ellipse_from_poly(self,xx,yy):
        poly = Polygon(list(zip(xx, yy)), animated=True, fill=False, color='yellow', linewidth=0.5)        
        self.ax.add_patch(poly)
        self.p = PolygonInteractor(self.ax, poly)


    def add_ellipse(self,clicks):
        pts=np.array(clicks)
        x=pts[:,0]
        y=pts[:,1]

        cx=(x[0]+x[1])/2
        cy=(y[0]+y[1])/2
         
        phi = math.atan2(y[1]-y[0],x[1]-x[0])
         
        a = math.cos(phi)*(x[0]-cx) +  math.sin(phi)*(y[0]-cy)
         
        p3x =   math.cos(phi)*(x[2]-cx) +  math.sin(phi)*(y[2]-cy)
        p3y = - math.sin(phi)*(x[2]-cx) +  math.cos(phi)*(y[2]-cy)
                 
        sqdenum = a**2-p3x**2;
        if(sqdenum<=0):
            print("Invalid points")
            return False
        
        
        b = abs( (a*p3y)/math.sqrt(a**2-p3x**2) )
         
        #R = np.arange(0,2.0*np.pi, 0.01)
        R = np.linspace(0,2.0*np.pi, 9)
         
        xx = cx + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
        yy = cy + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
        self.add_ellipse_from_poly(xx,yy)
        return True


    def to_smooth_poly(self, D):
        theta = np.linspace(0,1,D.shape[1])
        cs = CubicSpline(theta,D.T, bc_type='periodic')        
        ts = np.linspace(0, 1, 50)
        out=cs(ts);
        return out[:,0],out[:,1]


    def draw_label(self, l):
        xx,yy = self.to_smooth_poly(l.D)
        l.h, = self.ax.plot(xx,yy, color = color_for_class(l.cl))
        if(l.ht==None):
            l.ht = self.ax.text(l.tx, l.ty, ("%d" % l.jd), fontsize=10, color=color_for_class(l.cl), ha='center', va='center')       
        
    def insert_label(self, D, jd, cl):
        l = Label()
        l.jd = jd
        l.cl = cl
        l.tx = np.mean(D[0,:])
        l.ty = np.mean(D[1,:])               
        l.D = D
        self.labels.append(l)
        self.draw_label(l)
        if(jd>=self.nextjd):
            self.nextjd=jd+1
        return l
    
    def update_label(self,D,cl,l,revert):
        if(not revert):
            l.D = D
            l.cl = cl
            l.tx = np.mean(D[0,:])
            l.ty = np.mean(D[1,:])
        
        self.draw_label(l)
        self.save_labels(self.filename+".txt")
        
    def delete_label(self,l):
        self.labels.remove(l)
        
        if(l.h!=None):
            l.h.remove()
            
        if(l.ht!=None):
            l.ht.remove()
                
        self.save_labels(self.filename+".txt")
        

    def add_label(self, D, cl):
        l = self.insert_label(D, self.nextjd, cl)
        self.ensure_header(self.filename+".txt")
        with open(self.filename+".txt", 'a') as file:
            self.save_label(file,l)

    def ensure_header(self, filename):
        try:
            with open(filename, 'r') as file:
                header = file.readline();
                if(not header.startswith("#ILT VERSION 1")):
                    print("ERROR: existing file does not have the correct header")

        except IOError:
            with open(filename, 'a') as wfile:
                wfile.write("#ILT VERSION 1\n")

    def save_label(self, file, l):
        file.write("%d %d" % (l.jd, l.cl))
        for i in range(l.D.shape[1]-1):
            file.write(" %f %f" % (l.D[0,i],l.D[1,i]))
        file.write("\n")
        
    def save_labels(self, filename):
        with open(filename, 'w') as file:
            file.write("#ILT VERSION 1\n")
            for i in range(len(self.labels)):
                self.save_label(file,self.labels[i])
        
    def load_labels(self, filename):
        try:
            with open(filename, 'r') as file:
                header = file.readline();
                if(header.startswith("#ILT VERSION 1")):
                    self.load_labels_v1(filename)
                else:
                    print("Found old file. Converting ...")
                    self.load_labels_v0(filename)
                    os.rename(filename,filename+".bak")
                    self.save_labels(filename)

        except IOError:
            print("Failed to open '%s'" % filename )
        
    def load_labels_v0(self, filename):
        try:
            with open(filename, 'r') as file:
                for line in file:
                    vals = line.split()
                    
                    if(len(vals)<1):
                        print("Invalid line. Skipping")
                        continue
                    
                    jd = int(vals[0])
                    n = int( (len(vals)-1)/2 )
                    D = np.zeros((2,n),dtype=np.float)
                    for i in range(n):
                        D[0,i] = vals[1+2*i]
                        D[1,i] = vals[2+2*i]
                                                   
                    D=np.hstack((D,D[:,0:1]))
                    self.insert_label(D,jd, 0) # no class, so use 0
        except IOError:
            print("Failed to open '%s'" % filename )

    def load_labels_v1(self, filename):
        try:
            with open(filename, 'r') as file:
                header = file.readline(); # skip header line
                for line in file:
                    vals = line.split()
                    
                    if(len(vals)<1):
                        print("Invalid line. Skipping")
                        continue
                    
                    jd = int(vals[0])
                    cl = int(vals[1])
                    n = int( (len(vals)-2)/2 )
                    D = np.zeros((2,n),dtype=np.float)
                    for i in range(n):
                        D[0,i] = vals[2+2*i]
                        D[1,i] = vals[3+2*i]
                                                   
                    D=np.hstack((D,D[:,0:1]))
                    self.insert_label(D,jd, cl) 
        except IOError:
            print("Failed to open '%s'" % filename )

                
    def update(self):
        if(self.infotext):
            self.infotext.remove()
            
        text=""
        
        if(self.canvas.manager.toolbar._active=="PAN"):
            text = "PAN: Left drag:  Move image, Right drag: Zoom"
        elif(self.mode == Mode.IDLE):
            text = "IDLE: Click: add new label, Double Click: edit existing label"
        elif(self.mode == Mode.CREATE):
            text = "CREATE: Click: add next point (%d remaining), Esc: Abort" % (3-len(self.clicks))
        elif(self.mode == Mode.EDIT):
            text = "EDIT: Edit ellipse by dragging markers, Space: save changes, Esc: discard changes, Del: remove label"
        
        textext="P: Toggle Pan/Zoom mode, X/R: Toggle Tip60 channel, G: Toggle Marker, B: Toggle DAPI"
        
        self.infotext = plt.figtext(.02, .02, text+"\n"+textext)
        self.canvas.draw()
     
        
        
    def evaluate(self):
        

        imagefilename = os.path.basename(filename)
        
        with open(filename+".csv", 'w') as file:
            file.write("filename,id,class,area_px,r_imgmedian,r_mean,r_median,r_min,r_max,g_imgmedian,g_mean,g_median,g_min,g_max,b_imgmedian,b_mean,b_median,b_min,b_max\n")
            for l in self.labels:
                xx,yy = self.to_smooth_poly(l.D)
                xx,yy = skimage.draw.polygon(xx,yy)
    
                s_count  = len(xx)
                
                file.write("%s" % imagefilename)
                file.write(",%d" % l.jd)
                file.write(",%d" % l.cl)
                file.write(",%d" % s_count)
                                
                for i in range(3):
                    s_mean   = np.mean(self.im_raw[yy,xx,i])
                    s_median = np.median(self.im_raw[yy,xx,i])
                    s_min    = np.min(self.im_raw[yy,xx,i])
                    s_max    = np.max(self.im_raw[yy,xx,i])
                    file.write(",%f,%f,%f,%f,%f" % (self.img_median[i],s_mean,s_median,s_min,s_max) )
                file.write("\n")
                   
        print("Written to '%s'" % (filename+".csv"))
        #self.im_raw[yy,xx,0]=255
        
        
            
plt.rcParams['keymap.all_axes']=''
plt.rcParams['keymap.back']=''
plt.rcParams['keymap.forward']=''
plt.rcParams['keymap.grid']=''
#plt.rcParams['keymap.grid_minor']=''
plt.rcParams['keymap.home']='home'
plt.rcParams['keymap.save']=''
plt.rcParams['keymap.xscale']=''
plt.rcParams['keymap.yscale']=''
plt.rcParams['keymap.zoom']=''
plt.rcParams['keymap.fullscreen']=''
#keymap.fullscreen: ['f', 'ctrl+f']

filename = sys.argv[1];
tool = Tool(filename)
plt.show()
print("Byebye.")

