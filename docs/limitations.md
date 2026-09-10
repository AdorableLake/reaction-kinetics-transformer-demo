# Limitations and disclosure boundary

- The public demo uses an independently implemented, simplified Transformer baseline. It does not reproduce the unpublished SACs architecture or research-specific innovations and is not a substitute for a research or production model.
- Synthetic curves follow a small, noiseless, manually defined dimensionless family. They do not represent collected experiments, real chemical conditions, or a validated reaction mechanism.
- Evaluation is limited to a held-out partition of the same single synthetic task; no external validation is performed. These metrics do not represent SACs research performance, paper results, scientific validation, or real-world model performance, and do not establish chemical generalization or suitability for process control.
- Sigmoid bounds predictions between zero and one but does not enforce monotonicity or an exact initial value. The architecture is deliberately generic.
- No raw research data, real targets, research configurations, trained research weights, manuscripts, patent materials, or result-based explanations are included.
- The demo is newly implemented for public demonstration. It makes no claim about who historically implemented any research component or about individual project contributions.
- Inference uses a model trained in the current process; model serialization, deployment APIs, GPU benchmarking, and experiment tracking are outside this version.
- CPU execution is the supported demo path. Repeatability tests cover one CPU software environment; bitwise agreement across operating systems, hardware, or dependency versions is not guaranteed. Dependency pins cover direct dependencies only, not the full transitive dependency set. An isolated folder test is not an operating-system security sandbox.
- Intellectual-property review of any future research-derived additions and a repository license decision remain separate review items.
