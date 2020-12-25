# acl-gan

- Github: https://github.com/hyperplane-lab/ACL-GAN
- Arxiv: https://arxiv.org/pdf/2003.04858.pdf

- dataset: https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view
    - 3400 images

## steps

- download dataset into `data` folder
- unzip

## todo

- replace lists with jax indexing

- anything with AdaIN means and variances being passed into modules as second and third parameters not getting tracked by autograd?

- fid
- make tfrecords for tpu training?

## structure

- generator and discriminator architecture from munit

- two generators
    - take in (x, z)
    - G_S and G_T
    - image encoder, noise encoder, and decoder
- three discriminators
    - D_S, D_T, and D_hat
        - D^hat is the acl discriminator
    - multiscale 
- two noise/style encoders
    - E^z_S and E^z_T
    - x -> Z
    - only used for generating z/style for identity loss
    - otherwise randomly sampled
- losses
    - least squares loss for L_adv and L_acl
    - all losses are weighted
    - L^T_adv
        - optimizes G_T, D_T in D_T(x_S), D_T(G_T(x_S, z))
        - tries to go from S -> T
    - L^S_adv
        - takes output of G_T(x_S, z) as x^bar_T
        - optimizes G_S, D_S in D_S(x_S), 1/2 * [D_S(G_S(x^bar_T, z)) + D_S(G_S(x_S, z))]
        - tries to go from (S -> T) -> S and S -> S
    - L_acl
        - takes output of G_S(x^bar_T, z) as x^hat_S and output of G_S(x_S, z) as x^tilde_S
        - optimizes D^hat in D^hat(x_S, x^hat_s) + D^hat(x_S, x^tile_S)
        - tries to make sure that images that go from (S -> T -> S) and (S -> S) look like the originals and aren't just in the same distribution
    - L_idt
        - l1 loss between (x_S, G_S(x_S, E^z_S(x_S))) and (x_S, G_T(x_T, E^z_T(x_T)))
        - makes sure that you can learn the "unique" z vector for an image in a distribution
    - focus mask loss
        - makes the generator output another channel with values in [0, 1] so you can lerp between the original and generated image for more stability and quality
        - full loss derivation in section 3.3

## hyperparameters

- Adam
    - beta_1 = 0.5
    - beta_2 = 0.999
- batch size: 3
- lr: 0.0001
- learning rate annealed by 1/2 every 100k iterations
- trained for 350k iterations
- update generator every other iteration
- gamma (for focus mask loss): 0.001
- epsilon (for focus mask loss): 0.01
- loss weights
    - L_idt: 1
    - L_acl: 0.5
    - L_mask: gamma_min = gamma_max = 0
- data augmentation from councilGAN

## training

## notes

- z vectors in generators mean that there isn't *one* S -> T or T -> S mapping
- augmentations
    - council gan uses:
        - flip l-r; 0.5
        - color jitter; hue = 0.15
        - random rotation 35deg
        - random perspective transform: 0.35; prob 0.5
- its unpaired so you don't need to have paired augmentations
- bce with logits loss is a little different from plain bce loss because of the 
- adain norm
    - mlp outputs a list of concatenated means and stddevs for each feature in the residual blocks
    - residual blocks have same dimensionality to make it easier to implement
    - adain takes the splitted means and stddevs and uses it to rescale inputs
- alpha in dec is just 1
- adain style mean and variance network can output negative values so sqrt(-) can be nan, look into more later if there ends up being a problem
    - temporarily just disabling AdaIN
- ones are real zeros are fake
- cant vmap over loss_g since the param dicts aren't vmappable

## replication

- councilgan only trains on original images for the last 100k iterations
- councilgan uses a prob of 0.5 on the random perspective aug (https://github.com/Onr/Council-GAN/blob/7fe8f8a72ab1b00d4024dd09f414f53781f27eaa/utils.py#L170) while acl-gan doesn't(https://github.com/hyperplane-lab/ACL-GAN/blob/de319019a5c3cbf48a786b8248aaca21d39cbda1/utils.py#L108)
- converting from np -> torch -> pil -> torch -> np takes 300us
- color jitter and random perspective uses ~6ms per image
- training on un-augmented data means that it overfits on 3k images at 32x32 within a few hundred iterations with large batch sizes.
- currently using constant zero padding instead of reflect padding, wait until reflect padding is merged in then use (https://github.com/google/jax/issues/5010)
- ~~currently using jax.image.resize in the multiscale discriminator until i can figure out how to use avgpool in a vmap (https://github.com/google/flax/discussions/738)~~