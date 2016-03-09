In order to evaluate the performance of detection, descriptor extraction and matching methods, we need to use several quantities [^heinly] :

- the **detected keypoints (features)**, here we use the number in the reference image (i.e. the unaltered one) ;
- the **matches** ;
- the ***correct* matches**, which we estimate by evaluating distances between keypoints (a threshold must be set, it's `Alter::eps`) ;
- the **correspondences** : given the features we have on both images, how many pairs should be found ? We look for the ones that are close enough (using the tolerance `Alter::eps`).

Using these quantities, performance