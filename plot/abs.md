---
title: Image analysis through *keypoints*
---

# Abstract

...


# Theory of *keypoints*

## Detecting & describing
<!-- What are keypoints ?
 -->
In order to analyze images, computers need to process big amounts of data. However, in many cases, such as embedded systems, there are strong boundaries in memory availability, processing power, and time.

In order to alleviate this problem, one technique aims at extracting significant features from images, such that processing can be done on a reduced set of data. This is called *keypoint* analysis. Once these relevant points are found, one needs to describe them, that is, to find relevant information that could characterize that point --- whatever image it belongs to.

How do we find the *keypoints* in an image ?
How can we *describe* them ?

The solutions both these problems are respectively called the *detector* and the *descriptor*. Several algorithms have aimed at defining these methods.

## History

The first *keypoint* detection technique dates back to ... . It was developped by ..., who aimed at ...

SURF, (SIFT)
Binary !
rotation, scale, FREAK AKAZE



# Evaluation

Our first goal was to compare all the current methods, and possibly combining detector/secriptor pairs. We were inspired by Jared Heinly et al., [^4].

What is a *good* detector/descriptor ?
Could we combine methods provided by different authors ?

## Methodology

In order to evaluate the quality of a detection/description method, we applied a transformation to an image and analyzed the variation of the locations of *keypoints*, as well as how efficiently the algorithm paired the *keypoints* between the two images.

Some measures were provided in the article, 

## Results



# Application

