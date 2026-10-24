# ProtGEM training checkpoint — 2026-10-24

## Final configuration decision

- Model name: **ProtGEM (Protein General Encoder with Multimodal Fusion)**.
- Paper title: **From Ontology-Aware Protein Prediction to Biological Insight: Decoding Cyanobacterial Heterocyst Formation through Multimodal GO Annotation and PPI Inference**.
- The final configuration uses **3 GVP convolution blocks** (`StructEncoder.num_layers = 3`). Deep, 15-block experiments are not the final configuration.
- The previously identified final AF2DB experiment is `workspace/AF2DB/no_l2/`. Its saved configuration, label caches and dataset splits belong with its weights; current defaults are not a substitute for that snapshot.
- Existing workspaces load their own configuration copies. Changing `config/ProteinEncoder.conf` does not change the configuration of an existing experiment or a running process.

## Numerical stability changes

- Preserve the user's `1e-6` addition to the vector normalization denominator.
- Rewrite the protein–GO contrastive loss with an equivalent log-sum-exp expression; validate temperatures and positive-label availability.
- Leave BCE inputs, weighting and loss computation unchanged.
- Check predictions and losses for non-finite values. Check and clip gradients before the Adam update, with a default global norm threshold of `1.0` and a `--max-grad-norm` override.
- Stop on invalid gradients before updating optimizer state; log the failing batch and protein IDs. Do not silently replace NaNs or discard bad batches.
- Add optional `--detect-anomaly` for backward diagnostics and log the maximum gradient norm before clipping.
- Skip the auxiliary GO loss when its weight is zero.

## Evidence and limits

- The reported traceback stopped in `roc_curve` after non-finite predictions had already accumulated. It does not identify the original failing operator.
- The inspected BP/CC/MF caches in `deepstruct` contained no samples without positive labels, and the saved checkpoints had no non-finite tensors.
- Small CPU tests, including a 15-block structure encoder, did not reproduce the original intermittent failure. Greater depth is a suspected contributor, not a confirmed root cause.
- 38 tests passed in the `ev1` environment on 2026-10-23. No full CUDA training run was performed in this environment.
- Clipping changes large-gradient updates; stability changes should not be presented as an exact replay of historical training.

## Other changes included in this checkpoint

The checkpoint also preserves the user's existing model/pipeline figures, updated GO-pretraining ablation values and figure, and ProtGEM naming edits. README restructuring follows this checkpoint as a separate documentation change. No datasets, experiment caches or model weights are included in this commit.

## Follow-up found during README command verification

The shared `ContrastLoss` is also called by `train_ppi.py` with a one-dimensional batch of pair scores. The new two-dimensional-only validation would reject that input. A separate compatibility fix restores support for `[batch]` inputs while preserving the original reduction over the entire pair batch. A regression test verifies loss and gradient equivalence through the import used by `train_ppi.py`.
