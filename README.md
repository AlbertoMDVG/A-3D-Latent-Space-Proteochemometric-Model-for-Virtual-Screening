# A 3D Latent-Space Proteochemometric Model for Virtual Screening

This code follows a proteochemometrics approach to virtual screening using BigBind Dataset. A frozen
[MolFLAE](https://github.com/kirito-cpu/MolFLAE) encoder turns a ligand into a latent vector.
The same encoder turns a protein pocket into a latent vector (in three different ways). The two vectors are concatenated together, and a classifier is trained on them to predict binding.

## Layout

```text
src/ligands_encoding/    encode the BigBind ligands
src/pockets_encoding/    encode the BigBind pockets
src/screening/           train the models, score them, run the benchmarks
scripts/                 Slurm scripts for each step
vendor/MolFLAE/          the encoder, copied from its own repository
models/molflae/          the frozen MolFLAE weights
```


## BayesBind

The files `src/screening/encode_bayesbind.py` and `src/screening/metrics.py` repeat the
evaluation used in the BayesBind repository
([molecularmodelinglab/bigbind](https://github.com/molecularmodelinglab/bigbind),
[arXiv:2403.10478](https://arxiv.org/abs/2403.10478)).

Where the source code has been used to ensure results matching, and adapted it to
score the models in this repository. The aim was to get the same values they report, to get a 
comparinson.

## Vendored code

`vendor/MolFLAE/` is code from the original MolFLAE encoder and is included here for completeness. See `vendor/VENDOR.md`. Nothing inside `vendor/` is my own work.
