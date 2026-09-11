Tutorial: Inferring a SARS-CoV-2 Phylogeny
============================================

This tutorial walks through the SARS-CoV-2 analysis described in
the VINE paper (Siepel, Hassett & Staklinski, *"VINE: Variational inference
for scalable Bayesian reconstruction of species and cell-lineage
phylogenies,"* `bioRxiv <https://doi.org/10.64898/2025.12.24.696405>`_), in
the section "Applications to real DNA data" and Figure 5A-C. It reproduces
VINE's analysis of the smaller of the paper's two SARS-CoV-2 subsets: 364
whole genomes (~30kb each) drawn from Nextstrain.

Background
----------

The paper obtained the latest SARS-CoV-2 whole-genome sequences from
`Nextstrain <https://nextstrain.org>`_ (downloaded February 25, 2026),
consisting of roughly 74,000 genomes after data-quality filtering, and
extracted two subsets by stratified random sampling: a larger subset of
1060 genomes and a smaller subset of 364 genomes. This tutorial follows
the 364-genome subset, analyzed under the general time-reversible (GTR)
substitution model with the discrete gamma model for among-site rate
variation.

In the paper, VINE and BEAST 2 produced broadly similar trees for this
data set, with highly correlated pairwise distances and only minor
differences in branching pattern -- but VINE finished in about 13 minutes
where BEAST 2 required over 22 hours (both using 8 threads).

Getting the data
-----------------

The paper's data are a continuously-updated public feed, so this tutorial
reconstructs an equivalent subset rather than pinning to a byte-identical
file. Any similarly filtered/subsampled collection of a few hundred
aligned SARS-CoV-2 genomes will work fine for following along; only the
exact set of taxa (and therefore the exact tree) will differ from the
paper's original run.

1. Download the sequence alignment and metadata from Nextstrain's public
   "open" SARS-CoV-2 data feed:

   .. code-block:: console

       curl -O https://data.nextstrain.org/files/ncov/open/aligned.fasta.xz
       curl -O https://data.nextstrain.org/files/ncov/open/metadata.tsv.xz
       xz -d aligned.fasta.xz metadata.tsv.xz

2. Use `augur filter <https://docs.nextstrain.org/projects/augur/en/stable/usage/cli/filter.html>`_
   (part of the `Nextstrain <https://nextstrain.org>`_ toolchain) to
   discard low-quality/poorly-dated records and draw a stratified random
   sample by region and month, following the paper's Methods:

   .. code-block:: console

       augur filter \
         --sequences aligned.fasta \
         --metadata metadata.tsv \
         --min-length 29000 \
         --exclude-ambiguous-dates-by any \
         --group-by region year month \
         --sequences-per-group 1 \
         --subsample-seed 2 \
         --output-sequences small-subset.fasta \
         --output-metadata small-subset.tsv

   ``--min-length 29000`` excludes partial genomes, ``--exclude-ambiguous-dates-by
   any`` requires complete collection dates, and ``--group-by region year
   month --sequences-per-group 1`` caps the sample at one genome per
   region/year/month stratum, which is what produced the smaller
   (364-genome) subset in the paper (``--sequences-per-group 3`` was used
   instead for the paper's larger 1060-genome subset). ``--subsample-seed
   2`` makes the random draw reproducible for a given input snapshot.

Running VINE
------------

With ``small-subset.fasta`` in hand, run :doc:`vine </cli/vine>` with the
GTR substitution model and discrete gamma rate variation, the same
settings used in the paper:

.. code-block:: console

    vine small-subset.fasta \
      --gtr --dgamma 4 \
      --parallel 8 \
      --nsamples 1000 \
      --logfile vine.log \
      --nexus samples.nex \
      --mean mean.nwk \
      --embedding embedding.tsv

Option summary (see :doc:`the full vine reference </cli/vine>` for
details on any of these):

``--gtr``
    Use the general time-reversible substitution model, with all free
    parameters estimated from the data.

``--dgamma 4``
    Use Yang's discrete gamma model for rate variation across sites, with
    4 rate categories.

``--parallel 8``
    Use 8 OpenMP threads to parallelize likelihood calculations (requires
    a build with OpenMP support).

``--nsamples 1000``
    Draw 1000 samples from the approximate posterior after convergence
    (the paper's comparisons of tree distributions, e.g. Robinson-Foulds
    distances, were based on 1000 posterior samples).

``--logfile vine.log``
    Write per-iteration convergence metrics (log likelihood, ELBO, KL
    divergence, GTR rate and gamma-shape estimates, etc.) to
    ``vine.log``.

``--nexus samples.nex``
    Write all 1000 sampled trees, in NEXUS format, to ``samples.nex``.

``--mean mean.nwk``
    Write a single Newick tree reflecting the mean of the approximate
    posterior to ``mean.nwk`` -- this is the tree shown in Figure 5A/C of
    the paper.

``--embedding embedding.tsv``
    Write the final high-dimensional embedding (one row per taxon) to
    ``embedding.tsv``. Figure 5E of the paper shows the first two
    principal components of this embedding for the larger 1060-taxon
    run.

Note that ``--dimensionality`` was left unset here, so VINE estimates a
default embedding dimension from the number of taxa and reports it to the
terminal at startup.

Monitoring convergence
-----------------------

While running, and afterward in ``vine.log``, VINE reports one line per
iteration with columns including:

``ll``
    Log likelihood of the current mean tree under the substitution
    model.

``elbo``
    The evidence lower bound being maximized by stochastic gradient
    ascent (expected log likelihood minus KL divergence from the prior).

``kld``
    The KL-divergence term of the ELBO.

``gradnorm``
    Norm of the stochastic gradient, useful for diagnosing convergence
    (should shrink over iterations, on average).

``gtr[0..5]``, ``alpha``
    The six free GTR rate parameters and the gamma-shape parameter,
    updated jointly with the tree by stochastic gradient ascent (a
    nuisance-parameter feature enabled by ``--gtr``/``--dgamma``).

VINE monitors the running average ELBO and stops once it has stabilized
(governed by ``--miniter`` and ``--niterconv``; see :doc:`/cli/vine`),
then reports which iteration's parameters were retained. On a 364-taxon
alignment with 8 threads, this analysis typically converges in well under
half an hour -- consistent with the paper's reported ~13-minute runtime
for this data set (exact time depends on hardware).

Output files
------------

After convergence, you should have:

- ``mean.nwk`` -- a single Newick tree, the posterior mean estimate.
- ``samples.nex`` -- 1000 posterior-sample trees in NEXUS format, one
  ``TREE`` statement per sample, that can be loaded into standard tree
  viewers (e.g. `IcyTree <https://icytree.org>`_, FigTree, or
  ``ete3``/``dendropy`` in Python) to inspect the posterior cloud, as in
  the gray tree clouds of Figure 5A/D.
- ``embedding.tsv`` -- one row per taxon giving its estimated mean
  position in the learned embedding space.
- ``vine.log`` -- per-iteration convergence metrics, useful for
  diagnosing whether a run has converged or needs more iterations.

Next steps
----------

- Compare this tree (or its posterior sample) against a reference tree
  (e.g. from BEAST 2 or MrBayes) using :doc:`compareTrees </cli/compareTrees>`,
  which reports the mean split KL divergence between two posterior
  samples, as used to produce Figure 5B of the paper.
- Summarize properties of the posterior sample -- pairwise distance
  distributions, topological uncertainty, or branch-score accuracy
  against a reference -- using :doc:`evalTrees </cli/evalTrees>`.
- Re-run with ``--posterior`` to use the paper's recommended combination
  of covariance and normalizing-flow settings for a richer posterior
  approximation (see :doc:`/cli/vine`).
