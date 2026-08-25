"""DiffLens-style multiplicative IG-steering baseline on SAE latents.

Two-step pipeline:

1. ``locate_features.py`` — collect in-window (t_range) residual-layer-1
   activations for samples whose caption targets one (stage, shape) class
   (default: Beginning / single peak, i.e. "Begging_peak"), encode them with
   the frozen Top-K SAE, then attribute the frozen latent classifier's
   target-class log-probability with Integrated Gradients (baseline = the
   all-zero latent vector).  The top-k positive-IG latent dimensions become
   the *boost* set, the top-k negative-IG dimensions the *suppress* set.
   Scores and both sets are saved as a JSON artifact.

2. ``evaluate_steering.py`` — during DDIM sampling the SAE wrapper applies a
   fixed multiplicative edit inside the SAE timestep window to samples whose
   caption targets the located class::

       z_new[d] = z[d] * boost_factor     for boost dims
       z_new[d] = z[d] * suppress_factor  for suppress dims
       z_new[d] = z[d]                    otherwise

   The edited latents are decoded through the frozen SAE decoder.  Curves are
   scored with the frozen segment CNN and MSE against the test ground truth,
   paired with a Pure VerbalTS control sharing the same DDIM random stream.
"""

from __future__ import annotations

from Baseline import ig_attribution, multiplicative_steer  # noqa: F401

__all__ = ["ig_attribution", "multiplicative_steer"]
