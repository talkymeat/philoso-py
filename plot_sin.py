
# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt

def samp_sin(x, wl, a=1, ph=0):
    return a*(np.sin((x*np.pi*2/wl)+((np.pi*ph*2))))

def samp2sin(wl1, a1, ph1, wl2, a2, ph2, st=0, end=1, samples=1000):
    x = np.linspace(st,end,samples)
    y1 = samp_sin(x, wl1, a=a1, ph=ph1)
    y2 = samp_sin(x, wl2, a=a2, ph=ph2)
    return x, y1 + y2

# creating initial data values
# of x and y
x = np.linspace(0, 5, 200)
y = np.sin(x)

x, y = samp2sin(1,1,0,0.02,0.2, 0, samples=1000)
 
# to run GUI event loop
plt.ion()
 
# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)
 
# setting title
plt.title("1000 Samples", fontsize=20)
 
# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
 
# Loop
for i in range(998):
    # creating new Y values

    x, new_y = samp2sin(1,1,0,0.02,0.2, 0, samples=1000-i)
    plt.title(f"{1000-i} Samples", fontsize=20)
 
    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(new_y)
 
    # drawing updated values
    figure.canvas.draw()
 
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()
 
    time.sleep(0.1)