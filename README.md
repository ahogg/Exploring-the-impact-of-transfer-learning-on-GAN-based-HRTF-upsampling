# Exploring the impact of transfer learning on GAN-based HRTF upsampling

> :warning: **This code base is in active development**: Bugs are very likely!

A. O. T. Hogg, M. Jenkins, H. Liu, and L. Picinali: Exploring the impact of transfer learning on GAN-based HRTF upsampling. *In: Proc. EAAForumAcusticum, Eur. Congress on Acoust.*, 2023.

A. O. T. Hogg, M. Jenkins, H. Liu, I. Squires, S. J. Cooper and L. Picinali: HRTF upsampling with a generative adversarial network using a gnomonic equiangular projection. *In: Proc. IEEE/ACM Transactions on Audio Speech and Language Processing*, 2024.

First, run:
> Note: generate_projection only needs to be run once per dataset.
```sh
main.py generate_projection --hpc False --tag ari-upscale-4
```
to find the barycentric coordinates for each point in the cubed sphere. The --tag flag specifies the folder location of the results, and the --hpc flag changes the paths depending on whether the code is being executed remotely or locally (see the config.py).

Next run:
```sh
main.py preprocess --hpc False --tag ari-upscale-4
```
which interpolates data to find the HRIRs on the cubed sphere. It then obtains the HRTFs using the FFT and splits the data into train and validation sets. 

Then to train the GAN on the cubed sphere data run: 
```sh
main.py train --hpc False --tag ari-upscale-4
```
> Note: The training parameters can be modified in the config.py file.

To test the GAN on the valuation set run:
```sh
main.py test --hpc False --tag ari-upscale-4
```

To compare the performance against the barycentric baseline run:
```sh
main.py baseline --hpc False --tag ari-upscale-4
```
