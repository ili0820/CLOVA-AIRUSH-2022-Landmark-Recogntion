# Landmark Recognition - Classification

Task
----
Image Classification of Landmarks of a city
one image contains one landmark.

Dataset
----
Train
- 100,207 images with maximum 2560x1440 size

Test
- 12,179 images

Label
- each image has labels with fields given as follow

Image ID | Class ID | Original Image ID | Original Class ID | Image File Name
-- | -- | -- | -- | --

only Image ID, Class ID, Image File Name are used
- Image ID : unique image id
- Class ID : unique landmark id
- Image File name: name of a image file

Evaluation
----
[GAP(Global Average Preciiision)](https://evaluations.readthedocs.io/en/latest/kaggle_2020/global_average_precision.html)

![image](https://user-images.githubusercontent.com/65278309/229817368-17a959ce-1d36-48f2-8e85-55f3af74912f.png)
- N is the total number of predictions returned by the system, across all queries

- M is the total number of queries with at least one sample from the training set visible in it (note that some queries may not depict samples)

- P(i) is the precision at rank i. (example: consider rank 3 - we have already made 3 predictions, and 2 of them are correct. Then P(3) will be 2/3)

- rel(i) denotes the relevance of prediciton i: it’s 1 if the i-th prediction is correct, and 0 otherwise

Restrictions
----
- Can't open the actual image files and see with bare eyes, due to security reasons 
- Model FLOPs must not exceed 2G Flops. ( numbers of Flops are calculated using [ptflops](https://github.com/sovrasov/flops-counter.pytorch))

Result
----
2nd Place with 0.9852417 GAP 455.59M FLOPS

Used [Tinynet](https://arxiv.org/abs/2010.14819v2) and [Arcfaceloss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)
