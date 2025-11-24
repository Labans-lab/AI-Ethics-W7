🌍⚖️ AI Ethics Assignment
Theme: Designing Responsible and Fair AI Systems
📌 Overview

This assignment evaluates your understanding of ethical AI principles, identifying and mitigating biases, and applying ethics to real-world AI scenarios. It includes theory, case study analysis, a fairness audit using Python + AIF360, and a reflective section.

📖 Table of Contents

Part 1: Theoretical Understanding (30%)

Part 2: Case Study Analysis (40%)

Part 3: Practical Audit (25%)

Part 4: Ethical Reflection (5%)

Bonus Task: Policy Proposal – Ethical AI in Healthcare

Tools & Resources

Submission Details

Part 1: Theoretical Understanding (30%)
Q1: Define algorithmic bias and provide two examples.

Algorithmic bias occurs when an AI system produces unfair or unequal outcomes because of skewed data, flawed model design, or systemic societal biases encoded in the dataset.

Examples:

Hiring algorithms favoring male candidates because training data was dominated by past male applicants.

Facial recognition systems misidentifying darker-skinned individuals due to lack of diverse training data.

Q2: Difference between Transparency and Explainability. Why both matter?

Transparency refers to openness about how an AI system is built—its data sources, algorithms, limitations, and decision processes.

Explainability means making individual AI decisions understandable to humans (e.g., why a loan was denied).

Importance:
Both ensure trust, accountability, user empowerment, and compliance with regulations. Transparency helps detect systemic issues, while explainability helps individuals understand decisions affecting them.

Q3: How GDPR impacts AI development in the EU

GDPR enforces:

Data minimization: AI systems must only use data necessary for purpose.

Right to explanation: Users can question automated decisions.

Consent requirements: Explicit permission before personal data use.

Restrictions on automated profiling: High-risk automation requires human oversight.

This pushes developers to build AI systems that prioritize privacy, fairness, safety, and accountability.

Ethical Principles Matching
Principle	Definition
A) Justice	Fair distribution of AI benefits and risks.
B) Non-maleficence	Ensuring AI does not harm individuals or society.
C) Autonomy	Respecting users’ right to control their data and decisions.
D) Sustainability	Designing AI to be environmentally friendly.
Part 2: Case Study Analysis (40%)
Case 1: Biased Hiring Tool — Amazon
Source of Bias

Training data: Historically male-dominated tech workforce → model learned to penalize CVs containing female-associated keywords.

Feature weighting: Model penalized indicators like “women’s clubs”.

Lack of fairness constraints during model development.

Fixes to Improve Fairness

Rebuild dataset ensuring gender-balanced historical hiring records.

Remove gender proxies (e.g., clubs, female-coded terms).

Add fairness-aware algorithms such as disparate impact remover or reweighting.

Fairness Metrics to Evaluate

Disparate Impact Ratio

Equal Opportunity Difference

False Positive Rate (FPR) difference

Demographic parity difference

Case 2: Facial Recognition in Policing
Ethical Risks

Wrongful arrests due to misidentification.

Violations of privacy and consent.

Increased surveillance of minorities.

Reinforcement of systemic discrimination.

Recommended Policies

Independent bias audits before deployment.

Strict human oversight — AI suggestions must never be final proof.

Mandatory transparency and public reporting on accuracy by demographic group.

Community consultation before rollout.

Limit use to serious crimes — avoid mass surveillance.

Part 3: Practical Audit (25%)
Bias Audit on COMPAS Dataset using AIF360
✔️ Python Code Example (Simplified)
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
dataset = CompasDataset()

# Protected attribute: race
protected = dataset.protected_attribute_names[0]

# Original metrics
metric = BinaryLabelDatasetMetric(dataset, 
                                  privileged_groups=[{protected: 1}],
                                  unprivileged_groups=[{protected: 0}])

print("Disparate Impact:", metric.disparate_impact())

# Visualization: Disparate Impact
plt.bar(["Disparate Impact"], [metric.disparate_impact()])
plt.title("Disparate Impact Ratio (Original)")
plt.show()

# Mitigation: Reweighing
RW = Reweighing(unprivileged_groups=[{protected: 0}],
                privileged_groups=[{protected: 1}])
dataset_transformed = RW.fit_transform(dataset)

# Train model
X_train = dataset_transformed.features
y_train = dataset_transformed.labels.ravel()

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_train)

print("Accuracy:", accuracy_score(y_train, preds))

300-Word Audit Summary

The COMPAS recidivism dataset has been widely criticized for racial bias, particularly in assigning higher risk scores to Black defendants. Using AIF360, I conducted a fairness audit comparing outcomes between privileged (Caucasian) and unprivileged (African American) groups. The initial disparate impact ratio was significantly below 0.8, indicating strong adverse impact against minority groups. Additional metrics, such as false positive rate differences, revealed that African American individuals were more likely to be incorrectly labeled as “high risk” compared to Caucasian individuals. This confirms the presence of structural bias in the dataset and traditional machine learning workflows.

To mitigate this, I applied the Reweighing method from AIF360, which adjusts instance weights to balance representation across groups. After reweighing, model fairness improved, with the disparate impact ratio moving closer to the acceptable threshold. However, some fairness-accuracy trade-off was observed.

Remediation strategies include:

Pre-processing techniques like reweighing and disparate impact remover,

In-processing algorithms incorporating fairness constraints,

Post-processing calibration such as equalized odds.

Ultimately, fairness cannot rely solely on technical methods. It requires transparent reporting, continuous monitoring, and policy-level oversight.

Part 4: Ethical Reflection (5%)

For any future AI project I build (e.g., AI nutrition recommender or TEXT2POTAI), I will ensure ethical integrity by:

Using diverse and representative datasets.

Auditing regularly for bias.

Being transparent about data usage.

Ensuring user autonomy through clear consent mechanisms.

Prioritizing safety and non-harm.

Bonus Task: Policy Proposal — Ethical AI in Healthcare (Extra 10%)
1-Page Guideline: Ethical AI Use in Healthcare
1. Patient Consent Protocols

Obtain explicit, informed consent before patient data is used.

Allow opt-out without penalty.

Clearly explain data use, storage duration, and model purpose.

2. Bias Mitigation Strategies

Evaluate datasets for demographic balance.

Apply fairness algorithms (reweighing, gender/race de-biasing).

Regularly audit model outcomes by age, gender, ethnicity.

3. Transparency Requirements

Publish model accuracy and limitations.

Provide explainable outputs for clinicians.

Document all model updates and data sources.

Report error rates by demographic group.

Tools & Resources

AI Fairness 360 (AIF360)

Pandas & Matplotlib

COMPAS & ProPublica datasets

EU Guidelines for Trustworthy AI