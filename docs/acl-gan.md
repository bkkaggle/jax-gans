# acl-gan

- Github: https://github.com/hyperplane-lab/ACL-GAN
- Arxiv: https://arxiv.org/pdf/2003.04858.pdf

- dataset: https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view

## steps

- download dataset into `data` folder
- unzip

## todo

- fid

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
- two noise encoders
    - E^z_S and E^z_T
    - x -> Z
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

## replication

- councilgan only trains on original images for the last 100k iterations
- councilgan uses a prob of 0.5 on the random perspective aug (https://github.com/Onr/Council-GAN/blob/7fe8f8a72ab1b00d4024dd09f414f53781f27eaa/utils.py#L170) while acl-gan doesn't(https://github.com/hyperplane-lab/ACL-GAN/blob/de319019a5c3cbf48a786b8248aaca21d39cbda1/utils.py#L108)
- converting from np -> torch -> pil -> torch -> np takes 300us
- color jitter and random perspective uses 700us