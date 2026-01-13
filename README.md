# mri-alzheimers-detection-mdst

MDST Alzheimers Detection Project

//Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC8927715
## Problem
Alzheimers Disease is not an unfamiliar term to many: it affects millions of older adults around the world.
It accounts for more than half of dementia cases.

The most scary thing, however, is its slow and burdening progression. Initially starting off by taking the affected's short-term memory, it erases language processing, faces, even bodily functions as it slowly grinds the body to a halt.
But the silver lining to its slow progression is that Alzheimer's can be detected, and thus treated, early. The earlier it can be treated, the less harm the disease can do. 

That's why Alzheimer's detection has been one of the most prevalent issues in machine learning discourse: if we could have a computer be able to detect the disease as soon as it starts, we could use it to save the lives of millions. Accuracy is quite important in this, as the higher chance we can prevent a false diagnosis(or worse, not diagnosing when there is a real problem) the better.

## Dataset
link: https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy
"This dataset consists of four classes—No Impairment, Very Mild Impairment, Mild Impairment, and Moderate Impairment—each containing 2,560 axial MRI scans in the training set, with synthetic MRIs demonstrating quality and diversity as good as the original scans, as evidenced by a mean FID Score of 0.13, mean SSIM of 0.97, mean PSNR of 32 dB, and mean Sharpness Difference (SD) of 0.04 across all minority classes, all of which are close to their ideal values."- Luke Chugh, the compiler of this dataset

## Architectures
Our team tested out both VGG and ResNet architectures. Each person created their own architecture, so we chose the best accuracy from each of the architectures to compare. Below, we both explain how each architecture works and our results were.


### VGG
Essentially, our goal with VGG is to take in our image and determine several parameters from it, use gradient descent to essentially identify which paramater changes the image the most in context of our labels. After pairing features of an MRI scan with stage of Alzheimers, we can input an image, and based on what features it has, it is assigned a specific stage.

VGG employs the use of 16 layers- 13 convolutional layers and 3 connecting layers. The architecture follows a straightforward and repetitive pattern, and the lower amount of layers allows us to prevent overfitting (the model doing so well on the training data that it just doesn't work with the testing data)


Accuracy: .9,
Precision: .92,
Recall: .91,
f1-score: .92

### ResNet 
When training a model, if you increase the amount of processing layers in a neural network (which allows it to handle even more specific differences in brains between Alzheimers and non-Alzheimers), a known side effect is the vanishing/exploding gradient. Essentially, how gradient descent works is that for each layer, you calculate the gradient (the direction of steepest descent in a multivariable function). Sometimes, however, the gradient falsely either vanishes to 0 or explodes to infinity (which sounds quite funny, but may not be as fun). ResNet simply skips any layers that trigger a vanishing/exploding gradient, allowing us to use higher layers with lower risk.

Our ResNet model used 50 layers (almost 3x higher than the VGG). At the higher risk of overfitting, we can potentially retrieve more advanced and minute differences between the brains for edge cases. 

Accuracy: .81,
Precision: .82,
Recall: .82,
f1-score: .82

## Conclusion

Between the two we have tested, its VGG that worked better for this dataset. 

