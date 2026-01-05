# mlops2025_sandy_chloe
This project implements a complete machine-learning pipeline to predict **NYC Taxi Trip Duration**.  
It covers:

- Data preprocessing
- Feature engineering
- Model training (RandomForest & GradientBoosting)
- Batch inference
- Reproducible setup using `uv` and Python `src/` layout

**Goal:** Provide a fully reproducible pipeline from raw data to predictions.
## Project Structure
src/mlproject/ 
├── data/ 
├── preprocess/ 
├── features/ 
├── train/ 
├── inference/ 
├── pipelines/ 
├── utils/ 
└── __init__.py 
 
scripts/ 
├── preprocess.py 
├── feature_engineering.py 
├── train.py 
├── batch_inference.py 
  
Dockerfile 
docker-compose.yml 
pyproject.toml 
uv.lock 
README.md


Large files (datasets, model artifacts, predictions) are ignored in .gitignore.

Small outputs like metrics.json and model_report.txt are pushed.

# Selected Metrics and Models:

Metrics: RMSE, MAE

Best Model: GradientBoostingRegressor

Reasoning: GradientBoosting gave better validation RMSE and MAE compared to RandomForest
