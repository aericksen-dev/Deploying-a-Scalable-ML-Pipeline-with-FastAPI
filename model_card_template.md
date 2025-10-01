Model Details

Model type: RandomForestClassifier

Parameters: 200 estimators, default max depth, random state 42, n_jobs=-1

Framework: scikit-learn

Purpose: Predict whether an individual’s income exceeds $50K/year based on demographic features.

Data

Dataset: UCI Census Income (Adult) dataset

Features:

Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

Continuous: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

Target: Income category (<=50K vs >50K)

Metrics (Overall)

Precision: 0.7338

Recall: 0.6365

F1-score: 0.6817

Performance on Slices:

Precision: 0.5484 | Recall: 0.4722 | F1: 0.5075
workclass: ?, Count: 363

Precision: 0.7183 | Recall: 0.6711 | F1: 0.6939
workclass: Federal-gov, Count: 195

Precision: 0.7353 | Recall: 0.6250 | F1: 0.6757
workclass: Local-gov, Count: 389

Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000
workclass: Never-worked, Count: 1

Precision: 0.7345 | Recall: 0.6224 | F1: 0.6738
workclass: Private, Count: 4597

To evaluate fairness and robustness, we computed metrics across categorical slices.
Below are examples (replace with actual values from slice_output.txt):

Sex = Female → Precision: 0.71, Recall: 0.62, F1: 0.66

Sex = Male → Precision: 0.75, Recall: 0.65, F1: 0.70

Race = White → Precision: 0.74, Recall: 0.64, F1: 0.69

Race = Black → Precision: 0.70, Recall: 0.59, F1: 0.64

Marital-status = Never-married → Precision: 0.72, Recall: 0.60, F1: 0.65

(These numbers illustrate expected disparities; replace with your real slice outputs.)

Ethical Considerations & Limitations

Fairness concerns: Model performance varies across subgroups (e.g., sex, race, marital-status). This may propagate or amplify social inequities if deployed directly.

Dataset bias: The Census dataset reflects historical socioeconomic patterns in the U.S. and may encode structural biases.

Interpretability: Random forests provide feature importance, but predictions are not inherently explainable.

Usage: Should not be used for high-stakes decisions (e.g., employment, credit, housing) without additional fairness analysis, explainability tools, and mitigation strategies.

Recommendations

Monitor performance on sensitive subgroups when deploying.

Consider post-processing techniques or rebalancing for fairness.

Retrain periodically to reflect up-to-date data.
