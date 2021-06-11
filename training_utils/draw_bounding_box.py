import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_detection(image, boxes):
	image = image.astype(float)
	image = Image.fromarray(image)
	fig, ax = plt.subplots()
	ax.imshow()

	for box in boxes:
		w, h = image.size
		x_min = int(box[0]*w)
		y_min = int(box[1]*h)
		x_max = int(box[2]*w)
		y_max = int(box[3]*h)
		rect = patches.Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

	plt.show()


