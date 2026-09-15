# Vendored upstream code

Third-party research code, copied in so the pipeline is reproducible from one checkout.
**Nothing here is my own work** except the patches recorded below.
---


## MolFLAE

| | |
|---|---|
| Upstream | https://github.com/MuZhao2333/MolFLAE.git |
| Licence | none shipped upstream — no LICENSE file, and the README states no terms |
| History | `archive/vendor-git-history/MolFLAE.git` |

Working tree is unmodified upstream. My encoder code that used to live inside
`Latent_Experiments/` now lives in `src/molflae/` and imports this tree by path.

The `ckpt-zinc9M/` checkpoint (67 MB) is gitignored. An identical copy is at
`models/molflae/`, which is what the scripts default to.

**Licence note:** absent a licence file, upstream MolFLAE is technically all-rights-reserved.
