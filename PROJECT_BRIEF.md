# Project Brief: India Youth Career

## Non-negotiable rules

- Product flow: Student Profile -> ML Career Prediction -> Gemma 4 roadmap. Gemma never overrides the ML prediction.
- Never hard-code statistics, fabricate ML confidence, salary data, repositories, issue numbers or links.
- Show confidence/alternatives only from predict_proba.
- Never claim correlation is causation; label data as synthetic.
- Training is separate (`train_model.py`); the app only loads joblib artifacts and never refits.
- Required disclaimer text: "Career predictions are data-driven matches based on the model and information provided. They are not guarantees of career success."

## Dataset

The dataset is synthetic. The source workbook sheet is `Youth_Career_Data`; the prepared CSV is `data/india_youth_career_choices.csv`. It has no salary or satisfaction columns. The target is `Preferred_Career` with nine classes.
