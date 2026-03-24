import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# =========================
# USER INPUT
# =========================
image_path = "/home/elanca/Documents/GitHub/segmenteverygrain/MA0,5FA0,5Pb(I0,8Br0,2).jpg"
known_length_m = 200e-6   # e.g. 10 µm scale bar

# =========================
# LOAD IMAGE
# =========================
img = np.array(Image.open(image_path))

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click TWO points along the scale bar")

points = []

def onclick(event):
    if fig.canvas.manager.toolbar.mode != '':
        return
    if event.xdata is None or event.ydata is None:
        return
    
    points.append((event.xdata, event.ydata))
    ax.plot(event.xdata, event.ydata, 'ro')
    fig.canvas.draw()

    if len(points) == 2:
        fig.canvas.mpl_disconnect(cid)

        (x1, y1), (x2, y2) = points
        pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        pixel_to_meter = known_length_m / pixel_dist

        print("\n==== Calibration Result ====")
        print(f"Pixel distance: {pixel_dist:.2f} px")
        print(f"Known length: {known_length_m:.2e} m")
        print(f"Conversion: {pixel_to_meter:.3e} meters / pixel")
        print(f"Inverse: {1/pixel_to_meter:.3e} pixels / meter")

        # Draw the scale line
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        ax.text((x1+x2)/2, (y1+y2)/2, f"{known_length_m*1e6:.1f} µm",
                color='red', fontsize=12)
        fig.canvas.draw()
        plt.close(fig)   # this usually lets the cell finish cleanly

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()