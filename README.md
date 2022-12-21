# LbFS-Visual-Tracking
Correlation Filter-based Visual Tracking using VGG Features with Loss based Feature Selection

[![Video Label](https://www.youtube.com/watch?v=-jUjDpLfOw4)

## Framework

![graphical_abstract](https://user-images.githubusercontent.com/67869508/208832701-64a59ac7-4153-47e6-818c-7b6c5b5e19e7.png)


## Results
![result](https://user-images.githubusercontent.com/67869508/208832716-405d931b-e5c2-4344-bdb9-60691ade47a5.png)

Our system use VGG features for correlation filter-based visual tracking. To select some features, we also propose Loss-based Feature Selection(LbFS) that extract top-4 feature base on some loss functions. Our result is notated to green box and red box by comparing existing result notated to blue box. As a results, our system can track a target object between full frames in video, but the traditional method is not (resulting tracked box is outside the target object area in middle frame. 
