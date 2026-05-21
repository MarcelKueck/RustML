# RustML — Project Plan

A Rust machine learning library spanning classical methods, modern deep learning interop, and a path into robot learning — built with type-safe pipelines, first-class tensor and dataframe interop, and a shared trait foundation with RustyRobots for embodied AI.

**Tagline:** Machine learning in Rust — from linear regression to learned policies, with the type system on your side.

---

## 1. Vision and Identity

### What RustML is

A broad-coverage, framework-agnostic Rust machine learning library covering classical supervised and unsupervised methods, model selection and evaluation, preprocessing and feature engineering, classical RL, and — in Year 2 — a learned-policy layer built on Burn that bridges into robotics via the shared `embodied-core` trait crate. Inspired by scikit-learn's API coherence and Linfa's Rust-native pragmatism, but designed from day one to interoperate with both the Rust scientific-computing stack (`ndarray`, `polars`, `nalgebra`) and the Rust deep learning stack (`burn`), with a trait surface that makes classical models and learned policies interchangeable downstream.

### What RustML is not

- **Not a deep learning framework.** Burn is the substrate for neural networks; RustML doesn't compete with it. In Year 2, RustML *uses* Burn to provide the algorithm layer (BC, PPO, SAC, etc.) on top.
- **Not a notebook environment.** No REPL, no plotting, no kernel. `cargo add rustml` and use it from a binary, library, or service.
- **Not a Linfa replacement.** Linfa is the existing classical ML effort in Rust; RustML's scope is broader (RL, dataframe-first workflows, learned-policy bridge) and its design constraints are different (shared traits with robotics, Polars at the edges). Where overlap exists, the relationship should be cordial — credit prior art, learn from their API decisions, don't disparage.
- **Not Python-compatible.** No PyO3 bindings, no goal of being callable from Python. Rust users, end to end.

### Why it should exist

The Rust ML ecosystem in 2026 has Burn and Candle for deep learning, Linfa for classical methods, narrow crates for specific algorithms, and inference runtimes for deploying ONNX or GGUF models. It has no equivalent of scikit-learn — no single coherent library that takes you from data loading and preprocessing through model selection, evaluation, and deployment, with a consistent API across algorithm families. Linfa is closest but has stalled in coverage and never made a dataframe-first bet. The gap is real, and the moment is right: Polars is mature, ndarray is stable, Burn is in active development, and the Rust scientific computing stack has converged enough to build on.

The second reason is the embodied AI angle. Robotics is moving toward learned policies, and Python's grip on that work is loosening as performance and deployment requirements push teams toward typed, compiled languages. A Rust ML library designed from day one to share traits with a Rust robotics library is a uniquely well-positioned thing. Neither half exists yet. This plan, together with the RustyRobots plan, builds both halves on a shared foundation.

### Differentiators (the moat)

- **Polars at the edges, ndarray in the middle.** First-class `DataFrame` input/output adapters for data loading, feature engineering, and joins; `ndarray` as the canonical internal tensor type that pipelines and estimators operate on. Zero-copy where possible, explicit conversion where not. No other Rust ML library has committed to this split clearly.
- **Type-safe pipelines.** Estimator composition is checked at compile time — a pipeline that feeds the wrong shape or dtype into the next stage doesn't compile. Borrowed from sklearn's `Pipeline` ergonomically, made type-safe in the Rust way.
- **Shared trait foundation with RustyRobots.** `Policy`, `DynamicsModel`, `Observation`, `Action`, `Environment` live in a neutral `embodied-core` crate. Classical controllers from RustyRobots and learned policies from RustML implement the same traits. Year-1 design constraint, Year-2 payoff.
- **Best-in-class documentation.** Every public item documented with a runnable example, conceptual guides for every algorithm family in an mdBook, and a "rosetta stone" for users coming from scikit-learn. Same standard as RustyRobots — it's not a competitive advantage if both projects do it, but it is a credibility floor.
- **Feature-gated heavy dependencies.** Burn is gated behind a `deep` (or `learning`) feature flag. Polars is gated behind a `dataframe` flag. Users doing classical tabular ML with `rustml-linear` shouldn't compile a deep learning framework or a columnar engine to get a logistic regression.

---

## 2. Workspace Structure

Cargo workspace with one crate per concern. Users `cargo add rustml` for the umbrella crate with sensible default features, or pick individual crates for tight dependency control.

```
rustml/
├── Cargo.toml                       # workspace root
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── CHANGELOG.md
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                   # test, clippy, fmt, docs, MSRV, miri
│   │   ├── release.yml              # cargo-release automation
│   │   └── book.yml                 # mdBook deploy
│   └── ISSUE_TEMPLATE/
├── book/                            # mdBook source — the conceptual guide
├── docs/adr/                        # Architecture Decision Records
├── crates/
│   ├── rustml/                      # umbrella re-export crate
│   ├── rustml-core/                 # traits, errors, tensor aliases, prelude
│   ├── rustml-data/                 # Dataset, DataLoader, Polars↔ndarray adapters
│   ├── rustml-preprocessing/        # scalers, encoders, imputers, feature selection
│   ├── rustml-linear/               # linear/logistic regression, ridge, lasso, elastic net
│   ├── rustml-trees/                # decision trees, random forests, gradient boosting
│   ├── rustml-cluster/              # k-means, DBSCAN, hierarchical, GMM
│   ├── rustml-decomposition/        # PCA, ICA, NMF, t-SNE, UMAP
│   ├── rustml-metrics/              # classification, regression, clustering metrics
│   ├── rustml-modelselection/       # CV, grid/random search, train_test_split
│   ├── rustml-pipeline/             # type-safe pipelines, column transformers
│   ├── rustml-rl-classical/         # tabular Q-learning, SARSA, MDP utilities
│   ├── rustml-models/               # toy datasets (iris, boston, mnist loaders)
│   └── rustml-deep/                 # YEAR 2 — Burn-backed BC, PPO, SAC, diffusion, ACT
├── examples/                        # standalone runnable examples
└── benches/                         # criterion benchmarks
```

`embodied-core` lives **outside this workspace** as a sibling crate in its own repository, co-owned with RustyRobots. `rustml-core` depends on it from v0.1, and `rustml-deep` will depend on it heavily in Year 2 for the `Policy` and `Environment` implementations.

---

## 3. The Two Foundations: Type-Safe Pipelines + Tensor/DataFrame Split

These are the differentiators, so they're designed first and built right. Everything else sits on top.

### Tensor and dataframe split

The canonical internal representation is `ndarray`. Polars sits at the I/O boundary. Conversion is explicit but ergonomic.

```rust
// In rustml-core
pub type Tensor1<F = f64> = ndarray::Array1<F>;
pub type Tensor2<F = f64> = ndarray::Array2<F>;
pub type TensorView2<'a, F = f64> = ndarray::ArrayView2<'a, F>;

// In rustml-data — Polars adapters behind the `dataframe` feature
#[cfg(feature = "dataframe")]
pub mod polars_interop {
    use polars::prelude::*;
    use ndarray::Array2;

    pub fn dataframe_to_features(df: &DataFrame, columns: &[&str]) -> Result<Array2<f64>>;
    pub fn dataframe_to_target(df: &DataFrame, column: &str) -> Result<Array1<f64>>;
    pub fn features_to_dataframe(arr: ArrayView2<f64>, names: &[&str]) -> DataFrame;
}
```

The rationale, documented in ADR-002: `ndarray` is what every existing Rust ML crate expects internally, what BLAS-backed linear algebra works on directly, and what `nalgebra` interops with cheaply. Polars is unmatched for tabular wrangling but is a heavy dependency and the wrong abstraction for an inner training loop. Forcing the split early prevents the worst outcome: a library that's half-committed to both and good at neither.

`nalgebra` interop ships in `rustml-core` from day one for the same reason it matters in RustyRobots: zero-copy views in both directions via `ArrayView2` ↔ `MatrixView`. This is what makes `embodied-core` traits work seamlessly across both libraries.

### Type-safe pipelines

```rust
// In rustml-core
pub trait Estimator {
    type Input;
    type Output;
    type Fitted: FittedEstimator<Input = Self::Input, Output = Self::Output>;
    type Error;

    fn fit(self, x: &Self::Input, y: Option<&Self::Output>) -> Result<Self::Fitted, Self::Error>;
}

pub trait FittedEstimator {
    type Input;
    type Output;
    type Error;

    fn predict(&self, x: &Self::Input) -> Result<Self::Output, Self::Error>;
}

// In rustml-pipeline
pub struct Pipeline<H, T> { head: H, tail: T }

impl<H, T> Pipeline<H, T> {
    pub fn then<N>(self, next: N) -> Pipeline<Self, N>
    where
        N: Estimator<Input = <H::Fitted as FittedEstimator>::Output>,
        H: Estimator,
    { ... }
}

// Usage — type errors at compose time, not at fit time
let pipeline = Pipeline::new(StandardScaler::default())
    .then(PCA::new().n_components(10))
    .then(LogisticRegression::default());

let fitted = pipeline.fit(&x_train, Some(&y_train))?;
let preds = fitted.predict(&x_test)?;
```

The contract: every `Estimator` declares input and output associated types, and the pipeline combinator verifies adjacency at the type level. If `PCA` outputs `Array2<f64>` and the next stage expects `Array2<f32>`, it doesn't compile. If a transformer is placed after a final estimator, it doesn't compile. The escape hatch — a dynamic `BoxedEstimator` trait object — exists for users who genuinely need runtime composition (e.g., AutoML loops), with the explicit understanding that they're trading the guarantee for the flexibility.

---

## 4. What Users Can Do With It

```rust
use rustml::prelude::*;

// Load tabular data via Polars, hand to ndarray-based estimators
let df = polars::prelude::CsvReader::from_path("housing.csv")?.finish()?;
let x = dataframe_to_features(&df, &["sqft", "bedrooms", "age"])?;
let y = dataframe_to_target(&df, &["price"])?;

// Train-test split, fit, evaluate
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42))?;

let model = Ridge::new().alpha(1.0).fit(&x_train, Some(&y_train))?;
let preds = model.predict(&x_test)?;
println!("R²: {}", r2_score(&y_test, &preds));

// Type-safe pipeline
let pipeline = Pipeline::new(StandardScaler::default())
    .then(PolynomialFeatures::new().degree(2))
    .then(Lasso::new().alpha(0.1))
    .fit(&x_train, Some(&y_train))?;

// Cross-validation
let scores = cross_val_score(
    || Ridge::new().alpha(1.0),
    &x, &y,
    KFold::new(5),
    scoring::neg_mean_squared_error,
)?;

// Hyperparameter search
let best = GridSearchCV::new(RandomForest::default())
    .param("n_estimators", &[100, 200, 500])
    .param("max_depth", &[5, 10, None])
    .cv(KFold::new(5))
    .fit(&x_train, Some(&y_train))?;

// Clustering
let labels = KMeans::new().n_clusters(8).fit(&x)?.predict(&x)?;

// Dimensionality reduction
let embedded = UMAP::new().n_components(2).fit_transform(&x)?;

// Year 2 — learned policies sharing traits with RustyRobots
#[cfg(feature = "deep")]
{
    use embodied_core::{Policy, Environment};
    use rustml::deep::{BC, PPO};

    let policy = BC::new(network).fit(&expert_trajectories)?;
    let action = policy.act(&observation);  // Same trait RustyRobots controllers implement
}
```

---

## 5. Roadmap

### Year 1 — Classical ML, done right

**Month 1 — Foundations and infrastructure**

Workspace skeleton, dual MIT/Apache license, CI (test/clippy/fmt/docs/miri/MSRV), cargo-release, mdBook scaffold at the documentation domain. `rustml-core`: `Estimator` and `FittedEstimator` traits, tensor aliases, error types, prelude. Property-based testing infrastructure with `proptest`. First three ADRs (estimator trait surface, ndarray-vs-Polars split, dependency and feature-flag policy). Publish `rustml-core` 0.1.0. Announcement blog post.

**Month 2 — rustml-data and rustml-preprocessing**

`Dataset` and `DataLoader` abstractions. Polars↔ndarray adapters behind the `dataframe` feature. CSV, Parquet, JSON loaders. Standard scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), encoders (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`), imputers (`SimpleImputer`, `KNNImputer`). Property tests for invertibility where applicable.

**Month 3 — rustml-linear**

Ordinary least squares, ridge, lasso, elastic net. Logistic regression (binary and multiclass via one-vs-rest and multinomial). Numerical stability properly handled — QR for OLS, coordinate descent for L1, L-BFGS for logistic. Benchmarks vs Linfa and scikit-learn (via process boundary). First example notebooks in `/examples` covering regression and classification end-to-end.

**Month 4 — rustml-metrics and rustml-modelselection, first release**

Classification metrics (accuracy, precision, recall, F1, ROC-AUC, log loss, confusion matrix). Regression metrics (MSE, RMSE, MAE, R², MAPE). Clustering metrics (silhouette, ARI, NMI). `train_test_split`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`. `cross_val_score`, `cross_validate`. First public release: `rustml` 0.1.0 umbrella crate covering linear models and preprocessing.

**Also Month 4: `embodied-core` v0.1.** Co-shipped with RustyRobots. `Policy`, `DynamicsModel`, `Observation`, `Action`, `Environment`, plus the conformance test suite. `rustml-core` adopts it as a dependency from this release on. Both libraries' kickoff posts cross-reference it.

Announcement post to r/rust, r/MachineLearning, This Week in Rust.

**Month 5 — rustml-pipeline and rustml-trees part 1**

Type-safe `Pipeline` and `ColumnTransformer` combinators. `BoxedEstimator` escape hatch. Decision trees (classification and regression) with CART splitting. Random forests. Feature importance.

**Month 6 — rustml-trees part 2 and rustml-cluster**

Gradient boosting from scratch (commit to either histogram-based à la LightGBM or exact à la XGBoost — ADR required, lean histogram for performance and memory). K-means (with k-means++ init), MiniBatchKMeans, DBSCAN, hierarchical (agglomerative), Gaussian mixture models. Release 0.2.0.

**Months 7–8 — rustml-decomposition and hyperparameter search**

PCA, randomized PCA, kernel PCA. ICA, NMF. t-SNE, UMAP (wrap an existing crate if mature, implement otherwise — ADR required). `GridSearchCV`, `RandomizedSearchCV`, `HalvingSearchCV`. Optuna-style Bayesian search is a Year-2 stretch goal, not a Year-1 commitment.

**Months 9–10 — rustml-rl-classical**

Tabular Q-learning, SARSA, expected SARSA, double Q-learning. MDP and bandit utilities. `Environment` trait implementations from `embodied-core` for the standard suite (FrozenLake, Taxi, CartPole-with-discretization). Deliberately scoped to classical methods — anything requiring function approximation waits for Year 2 and Burn.

**Months 11–12 — Polish, profiling, 1.0**

Performance pass with `criterion` benchmarks across the library. Profile-guided optimization of hot loops. Documentation completeness audit — every public item, every example. Release **1.0.0** — API stability promise on the classical surface area.

### Year 2 — Learned policies and the deep bridge

**Theme:** add `rustml-deep` as a Burn-backed crate behind the `deep` feature flag, implementing learned policies that share traits with RustyRobots' classical controllers via `embodied-core`.

**Commitments:**

- **Behavior cloning (BC).** Tractable, well-understood, the natural starting point. Imitation from demonstration trajectories.
- **PPO and SAC.** The two most-used policy-gradient and actor-critic methods. Tractable in Burn as it stands.
- **Trajectory data abstractions.** `Trajectory`, `Transition`, `ReplayBuffer`, `DemonstrationDataset`. These belong in `rustml-deep` (or possibly `rustml-data` with a `trajectories` sub-module) and are the data-side counterpart to `embodied-core`'s `Environment`.
- **The integration story.** A worked example: a robot defined in RustyRobots (Franka Panda URDF, FK/IK, dynamics) trained with PPO in RustML against an `Environment` whose `DynamicsModel` is RustyRobots' analytic dynamics. This is the killer demo. It is the entire point of the trait-sharing exercise.

**Stretch goals, contingent on Burn's state at end of Year 1:**

- **Diffusion policies.** Tractable only if Burn's diffusion primitives have matured. Flagged as a risk; reassess at the Year-1 retrospective.
- **ACT (Action Chunking Transformer).** Requires solid transformer support in Burn. Also a risk; same reassessment.

If Burn isn't ready for diffusion or ACT by end of Year 1, don't fake it — defer to Year 3 and be honest in the roadmap update.

**Other Year-2 work:**

- Begin the textbook companion — a "Machine Learning in Rust" book that uses RustML as its working library, similar in spirit to what Aurélien Géron did for scikit-learn.
- Bayesian hyperparameter search (TPE, GP-based) if community demand warrants.

---

## 6. Documentation Standard

Enforced in CI. Same standard as RustyRobots — explicitly so, because both projects are competing for the same kind of user trust and inconsistency between them would undermine both.

- Every public item has a doc comment — `#![deny(missing_docs)]` at every crate root.
- Every public function has a runnable doc example — `cargo test --doc` in CI.
- Every algorithm has a conceptual page in the mdBook explaining the math (LaTeX via KaTeX), the assumptions, the failure modes, and the computational complexity, linked from the rustdoc.
- Every example in `/examples` has a README, a runnable `main`, and where applicable produces a plot or a serialized model artifact.
- "Rosetta stone" page with side-by-side translations from scikit-learn. Same marketing-tool-disguised-as-documentation move as RustyRobots' Corke and PythonRobotics rosetta stones. People come from scikit-learn; meet them where they are.

---

## 7. RustyRobots Coordination

The shared trait crate is the load-bearing thing. Get it right early or pay for it forever.

- **`embodied-core` is co-owned**, dual MIT/Apache, lives in its own repo, depended on by both projects from their respective v0.1 releases.
- **Ship a minimal v0.1 at end of Month 4.** `Policy` (with associated `ActionDistribution` type for stochastic policies, and a `DeterministicPolicy` sub-trait that auto-impls `Policy` for the deterministic case), `DynamicsModel` (same trait for analytic and learned — interchangeability is the feature), `Observation`, `Action`, `Environment` (wraps a `DynamicsModel` plus reward, termination, and observation logic). No fancy features. Lock the contract while both libraries are still malleable.
- **Conformance test suite as a feature flag.** A set of property tests any implementer of `Policy`, `DynamicsModel`, etc. can run against their own implementation to verify they're honoring the contract — e.g., a stochastic `Policy`'s sampled actions are in the declared action space; a `DynamicsModel`'s step is deterministic given a fixed RNG seed; an `Environment`'s reward is finite for any reachable state. This is what makes a shared trait crate actually trustworthy rather than nominal.
- **One shared design doc** linked from both repos, kept in the `embodied-core` repo. Changes to the trait surface go through that doc before they go through code.
- **Reserve `rustml-deep` on crates.io** in Month 1 as a placeholder to prevent squatting.
- **The Year-2 integration demo** — RustyRobots robot trained with RustML PPO — is the deliverable that justifies the whole trait-sharing exercise. Treat it as a release blocker for `rustml-deep` 0.1.0.

---

## 8. The First Ten GitHub Issues

1. `[meta]` Set up Cargo workspace with empty crates
2. `[meta]` CI: clippy, fmt, test, doc-build matrix on stable + MSRV + nightly
3. `[meta]` Set up mdBook, deploy via GitHub Pages
4. `[meta]` Choose and document MSRV policy (proposal: stable - 6 months)
5. `[design]` ADR-001: Estimator and FittedEstimator trait surface
6. `[design]` ADR-002: ndarray as canonical tensor type, Polars at the edges
7. `[design]` ADR-003: Dependency and feature-flag policy (Burn and Polars feature-gated, no GPL)
8. `[core]` Implement `Estimator`, `FittedEstimator`, and tensor type aliases
9. `[core]` Error types and Result aliases
10. `[docs]` CONTRIBUTING.md with PR template, commit conventions, license decision

---

## 9. Working With Claude as Co-Developer

- **Per-crate kickoff sessions:** paste the relevant ADRs and ask for API design review before writing code.
- **Pre-commit reviews:** paste significant diffs before pushing for idiom and edge-case review.
- **Stuck-on-math sessions:** walk through derivations together. Coordinate descent for L1, the EM steps for GMM, the policy-gradient theorem, the reparameterization trick — these all reward an interlocutor.
- **API ergonomics gut-checks:** before publishing a function, show three example usages. If they read awkwardly, redesign. Especially important for pipeline combinators where the type signatures are doing real work and ugly signatures destroy adoption.
- **Don't outsource the core thinking.** This is a learning project first. Use Claude to discuss tradeoffs, not to write the code.

---

## 10. Quality Bar

Machine learning has a particular failure mode where wrong answers look like right answers — the model fits, the loss decreases, the predictions are plausible, and the bug is in the gradient or the loss function or the sampler and nobody notices for six months. "MVP quality" in ML means silently shipping a logistic regression with the wrong link function. Don't do that.

- Numerical correctness is verified against reference implementations (scikit-learn via cross-language test fixtures, hand-derived ground truth, or published numerical examples from textbooks).
- Every estimator has property tests covering invariants (predictions on training data after fitting, deterministic output given a fixed seed, expected behavior at boundary conditions like single-sample input or perfectly separable classes).
- Every loss, gradient, and metric has at minimum a closed-form unit test and a finite-difference sanity check.
- Breadth comes from many small correct pieces, never from rushing a single piece. A library with five trustworthy algorithms beats a library with fifty plausible ones.