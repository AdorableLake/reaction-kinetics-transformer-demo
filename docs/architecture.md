# Architecture

This document describes only the independently implemented public Transformer demo, not the unpublished SACs research architecture or innovations. All data and metrics are synthetic demonstration outputs, not research or publication results.

```text
Independent artificial formula + seeded random features
                       |
             train / validation / test split
                       |
             fit standardizer on train only
                       |
      scalar feature tokens + learned feature identities
                       |
         standard PyTorch Transformer encoder
                       |
              mean pool + linear + sigmoid
                       |
            synthetic trajectory prediction
                       |
       train MSE / validation state selection / test MAE, RMSE
```

Inputs have four dimensionless features. Each feature becomes a token through a shared scalar projection and a feature-identity embedding. The standard Transformer encoder mixes the tokens. A pooled representation predicts the configured number of time points simultaneously. There is no research-specific prior, kinetic descriptor module, residual research head, or custom scientific loss.

## Artificial data definition

Each of four features is independently uniform on [-1, 1]. Time is evenly spaced on [0, 1]. The deliberately chosen rate is `0.4 + 0.3*(x0+1) + 0.2*(x1+1)`; the plateau is `0.1 + 0.1*(x2+1)`. Targets are `plateau + (1-plateau)*exp(-rate*time)`. Feature `x3` is a nuisance variable. Constants and distributions were manually defined for this demo, without sampling, perturbing, or fitting any research dataset. They are not empirical estimates or chemical parameter claims.

The sample split is seeded and disjoint. Feature means and standard deviations are fitted on training features only; the same transformation is applied to all three partitions. The Transformer receives a `[batch, 4]` standardized feature matrix, forms `[batch, 4, d_model]` tokens, and returns `[batch, time_points]` predictions. Tokens represent features, not time steps; the time grid defines the target curve and is not a separate model input.

Training updates model parameters with mini-batch MSE and Adam. After each epoch, validation MSE is computed without gradient updates. Whenever it improves, the model state is copied in memory. Training runs for the configured number of epochs, then restores the state with the lowest validation MSE before test inference.

The restored model predicts the held-out test partition once. MAE and RMSE are computed over all test samples and time points. For comparison, the training-mean baseline averages training target curves at each time point and repeats that curve for every test sample; it is scored against the same test targets with the same metrics. Test data do not participate in preprocessing fitting, parameter updates, or model selection. Neither predictions nor model state are persisted.
