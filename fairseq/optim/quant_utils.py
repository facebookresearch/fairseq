import logging

logger = logging.getLogger(__name__)


def estimate_omega(latent_p):
    """Compute layer-wise scaling factor estimated from latent weights.
    XNOR-Net: https://arxiv.org/abs/1603.05279
    """
    return latent_p.norm(p=1).div(latent_p.nelement())


def scaled_sign_(latent_p, omega):
    """In-place sign function scaled by layer-wise factor."""
    return latent_p.sign_().mul_(omega)
