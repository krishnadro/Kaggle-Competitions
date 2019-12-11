# MiFID
Submissions are evaluated on MiFID (Memorization-informed Fréchet Inception Distance), which is a modification from [Fréchet Inception Distance (FID).](https://arxiv.org/abs/1706.08500)

The smaller MiFID is, the better your generated images are.

## What is FID?

Originally published [here](https://arxiv.org/abs/1706.08500) ([github](https://github.com/bioinf-jku/TTUR)), FID, along with Inception Score (IS), are both commonly used in recent publications as the standard for evaluation methods of GANs.

In FID, we use the Inception network to extract features from an intermediate layer. Then we model the data distribution for these features using a multivariate Gaussian distribution with mean µ and covariance Σ. The FID between the real images r and generated images g is computed as:

$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr} (\Sigma_r + \Sigma_g - 2 (\Sigma_r \Sigma_g)^{1/2})$

![](https://latex.codecogs.com/gif.latex?%5Ctext%7BFID%7D%20%3D%20%7C%7C%5Cmu_r%20-%20%5Cmu_g%7C%7C%5E2%20&plus;%20%5Ctext%7BTr%7D%20%28%5CSigma_r%20&plus;%20%5CSigma_g%20-%202%20%28%5CSigma_r%20%5CSigma_g%29%5E%7B1/2%7D%29)


where Tr sums up all the diagonal elements. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

## What is MiFID (Memorization-informed FID)?
In addition to FID, Kaggle takes training sample memorization into account.

The memorization distance is defined as the minimum cosine distance of all training samples in the feature space, averaged across all user generated image samples. This distance is thresholded, and it's assigned to 1.0 if the distance exceeds a pre-defined epsilon.

In mathematical form:

![](https://latex.codecogs.com/gif.latex?d_%7Bij%7D%20%3D%201%20-%20cos%28f_%7Bgi%7D%2C%20f_%7Brj%7D%29%20%3D%201%20-%20%5Cfrac%7Bf_%7Bgi%7D%20%5Ccdot%20f_%7Brj%7D%7D%7B%7Cf_%7Bgi%7D%7C%20%7Cf_%7Brj%7D%7C%7D)

where fg and fr represent the generated/real images in feature space (defined in pre-trained networks); and fgi and frj represent the ith and jth vectors of fg and fr, respectively.

![](https://latex.codecogs.com/gif.latex?d%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%7D%20%5Cmin_j%20d_%7Bij%7D)

defines the minimum distance of a certain generated image (i) across all real images ((j), then averaged across all the generated images.

![](https://storage.googleapis.com/kaggle-media/competitions/GAN/latex-img-150.png)

defines the threshold of the weight only applies when the (d) is below a certain empirically determined threshold.

Finally, this memorization term is applied to the FID:

![](https://latex.codecogs.com/gif.latex?MiFID%20%3D%20FID%20%5Ccdot%20%5Cfrac%7B1%7D%7Bd_%7Bthr%7D%7D)

## Kaggle's workflow calculating MiFID for public and private scores

Kaggle calculates public and private MiFID scores with the same code, but with different pre-trained models and evaluation images. The public pre-train neural network is [Inception](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz), and the public images used for evaluation are the ImageNet Dogs (all 120 breeds). We will not be sharing what private model or dataset is used for the private MiFID score.

A demo of our MiFID evaluation code can be seen [here](https://www.kaggle.com/wendykan/demo-mifid-metric-for-dog-image-generation-comp).

Kaggle workflow of computing the public/private MiFID is demonstrated below:

![](https://storage.googleapis.com/kaggle-media/competitions/GAN/Kaggle%20GAN%20Diagram%20(3).png)


## Submission File
You are going to generate 10,000 images that are in `PNG` format. Their sizes should be 64x64x3 (RGB). Then you need to zip those images and your output from your Kernel should only have ONE output file named `images.zip.`

Please note that Kaggle Kernels has a number of output files capped at 500. We highly encourage you to either directly write to a `zip` file as you generate images, or create a folder at `../tmp` as your `temporary directory.`
