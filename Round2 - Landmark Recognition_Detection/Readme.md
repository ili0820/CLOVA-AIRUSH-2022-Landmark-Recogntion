# Landmark Recognition - Detection

Task
----
Find a bounding box and it's Landmark id in an image

one image contains one landmark.

Dataset
----
Train
- 16,290 images with maximum 2705x2705 size 

Test
- 12,179 images

Label
- 143 classes
- each image has labels with fields given as follow

image_idx|class_idx|roi_x0|roi_y0|roi_x1|roi_y1|image_file_name|width|height|center_x|center_y|img_width|img_height
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | 

- image_idx : unique image idx
- class_idx : landmark id
- roi_x0 , roi_y0 : bbox left top
- roi_x1 , roi_y1 : bbox right bottom
- center_x , center_y : center point of bbox 

Evaluation
----
[mAP(mean Average Precision)](https://www.v7labs.com/blog/mean-average-precision)

![image](https://user-images.githubusercontent.com/65278309/229880791-fb95cb4d-5221-4075-88b7-983327f20ed5.png)

- Average of [AP](https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b) of each class

- average over number of classes


Restrictions
----
- Can't open the actual image files and see with bare eyes, due to security reasons 
- 0.8 IOU

Result
----
5th Place with 0.853 mAP
