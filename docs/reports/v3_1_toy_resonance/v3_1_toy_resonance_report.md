New-Item -ItemType Directory -Force docs\reports\v3_1_toy_resonance | Out-Null

@'
# OpenReco v3.1 Toy Resonance Report

## Purpose

OpenReco v3.1 adds a downstream physics-observable study on top of the v3.0 detector-effects framework.

The goal is to show that detector assumptions and reconstruction choices can propagate from track-level reconstruction into a reconstructed invariant-mass observable.

The controlled toy channel is:

```text
J/psi -> mu+ mu-