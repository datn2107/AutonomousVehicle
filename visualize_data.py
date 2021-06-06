import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def plot_bounding_box(image_path, xmin, ymin, xmax, ymax, id_category):
	'''
		Draw bouding box
			(xmin, ymin): left top 
			(xmax, ymax): right bottom
			id_category: name of category
	''' #
	im = cv2.imread(image_path)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	# Create Bounding Box and Text for each box
	for x1,y1,x2,y2,id_cat in zip(xmin, ymin, xmax, ymax, id_category):
		# Draw Bouding Box
		im = cv2.rectangle(im ,(int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
		# Put Text for each box
		#im = cv2.putText(im, str(id_cat), (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2

	# Show image
	plt.imshow(im)
	plt.show()
	# Save image
	# plt.imsave('D:\AIP391 Autonomous Driving\Data\Object Detection\errortest.jpg', im)


# Define path for file csv and folder image
csv_path = r'D:\Autonomous Driving\Data\Object Detection\label\train.csv'
folder_path = r'D:\Autonomous Driving\Data\Object Detection\image\train'

# Read csv
df = pd.read_csv(csv_path)

# Get specific image to drawing box

# Create List contain coordinate of bounding box in one image
xmin = []
ymin = []
xmax = []
ymax = []
id_category = []

print(len(df.index))

# Get information of each bounding box of image
for id in range(len(df.index)):
	if (df.iloc[id]['id_category'] != 13):
		continue

	image_name = df.iloc[id]['name']
	print(image_name)
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	id_category = []

	for bounding_box in df.loc[df['name'] == image_name, ['x1', 'y1', 'x2', 'y2', 'id_category']].values:
		if (bounding_box[4] != 13):
			continue
		xmin.append(bounding_box[0]) # x1
		ymin.append(bounding_box[1]) # y1
		xmax.append(bounding_box[2]) # x2
		ymax.append(bounding_box[3]) # y2
		id_category.append(bounding_box[4]) # id_category
		print(bounding_box)

	plot_bounding_box(folder_path + '\\' + image_name, xmin, ymin, xmax, ymax, id_category)

print(df.head())

#plot_bounding_box(folder_path + '\\' + image_name, xmin, ymin, xmax, ymax, id_category)