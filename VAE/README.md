# Auto Encoders

<br>

Autoencoders are a type of neural network architecture used for unsupervised learning, where the goal is to learn a compressed representation (encoding) of the input data. The basic idea behind autoencoders is to learn a function that maps the input data to a lower-dimensional representation and then reconstructs the original data from the encoded representation. 

<br>

![auto-encoders](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/60ba3ecb0057e25cf8317ede_autoencoder1.png)

<br>

# Variational Auto Encoders

<br>

Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation. 

<br>

![vae](https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-18-at-12.24.19-AM.png)

<br>

# Latent Space 

<br>

![vael](https://www.jeremyjordan.me/content/images/2018/03/6-vae.png)

<br>

# Loss

<br>

```python
def gaussian_likelihood(self, mean, logscale, sample):
    scale = torch.exp(logscale)
    dist = torch.distributions.Normal(mean, scale)
    log_pxz = dist.log_prob(sample)
    return log_pxz.sum(dim=(1, 2, 3))
```

<br>

```python
def kl_divergence(self, z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl
```