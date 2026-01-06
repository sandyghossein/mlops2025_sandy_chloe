# NYC Taxi Trip Duration Prediction - MLOps Project

This project implements a complete end-to-end machine learning pipeline to predict **NYC Taxi Trip Duration**. It covers data preprocessing, feature engineering, model training, batch inference, and deployment using AWS SageMaker Pipelines.

**Goal:** Provide a fully reproducible pipeline from raw data to predictions, with support for local execution, Docker containers, and cloud-based SageMaker pipelines.

---

## Project Structure

```
mlops2025_sandy_chloe/
├── src/
│   └── mlproject/
│       ├── preprocess/
│       │   └── clean.py
│       ├── features/
│       │   └── transformers.py
│       ├── train/
│       │   └── models.py
│       ├── inference/
│       │   └── predict.py
│       └── pipelines/                
│           ├── run_training_pipeline.py
│           └── run_batch_inference_pipeline.py
│   
├── scripts/                     # Entry point scripts
│   ├── preprocess.py           # Preprocessing script
│   ├── feature_engineering.py  # Feature engineering script
│   ├── train.py                # Training script
│   └── batch_inference.py      # Batch inference script
│
│
├── data/                        # Data directory (gitignored)
│   ├── train.csv               # Raw training data
│   ├── test.csv                # Raw test data
│   └── processed/             # Processed data outputs
│
│
├── Dockerfile                   # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
├── pyproject.toml              # Project dependencies (uv)
├── uv.lock                     # Locked dependencies
└── README.md                   # This file
```

## How to Run Locally

### 1. Install Dependencies

```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

### 2. Prepare Data

Place your raw data files in the `data/` directory:
- `data/train.csv` - Training data with `trip_duration` column
- `data/test.csv` - Test data for inference

### 3. Run the Pipeline Steps

#### Step 1: Preprocessing
```bash
uv run python scripts/preprocess.py
```
This cleans the training and test datasets and saves outputs to `data/processed/`.

#### Step 2: Feature Engineering
```bash
uv run python scripts/feature_engineering.py \
    --clean_train data/processed/train_clean.csv \
    --clean_test data/processed/test_clean.csv \
    --out_dir outputs
```
This creates engineered features and saves them as Parquet files, plus a preprocessor artifact.

#### Step 3: Training
```bash
uv run python scripts/train.py \
    --features_train outputs/features_train.parquet \
    --clean_train data/processed/train_clean.csv \
    --out_dir outputs
```
This trains multiple models (RandomForest and GradientBoosting), selects the best one, and saves:
- `outputs/model.joblib` - Trained model
- `outputs/metrics.json` - Model evaluation metrics
- `outputs/model_report.txt` - Human-readable training report

#### Step 4: Batch Inference
```bash
uv run python scripts/batch_inference.py \
    --features_test outputs/features_test.parquet \
    --model_path outputs/model.joblib \
    --out_dir outputs
```
This generates predictions and saves them to `outputs/YYYYMMDD_predictions.csv`.

### Alternative: Run All Steps Sequentially

```bash
# Run complete pipeline locally
uv run python scripts/preprocess.py
uv run python scripts/feature_engineering.py --clean_train data/processed/train_clean.csv --clean_test data/processed/test_clean.csv --out_dir outputs
uv run python scripts/train.py --features_train outputs/features_train.parquet --clean_train data/processed/train_clean.csv --out_dir outputs
uv run python scripts/batch_inference.py --features_test outputs/features_test.parquet --model_path outputs/model.joblib --out_dir outputs
```

---

## How to Run with Docker

### Option 1: Using Docker Compose

```bash
# Build and run the training container
docker-compose up app

# Or run a specific script
docker-compose run app python scripts/preprocess.py
docker-compose run app python scripts/feature_engineering.py --clean_train data/processed/train_clean.csv --clean_test data/processed/test_clean.csv --out_dir outputs
```

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -t ml-project .

# Run preprocessing
docker run -v $(pwd):/app ml-project python scripts/preprocess.py

# Run feature engineering
docker run -v $(pwd):/app ml-project python scripts/feature_engineering.py \
    --clean_train data/processed/train_clean.csv \
    --clean_test data/processed/test_clean.csv \
    --out_dir outputs

# Run training
docker run -v $(pwd):/app ml-project python scripts/train.py \
    --features_train outputs/features_train.parquet \
    --clean_train data/processed/train_clean.csv \
    --out_dir outputs

# Run batch inference
docker run -v $(pwd):/app ml-project python scripts/batch_inference.py \
    --features_test outputs/features_test.parquet \
    --model_path outputs/model.joblib \
    --out_dir outputs
```

## How to Run the SageMaker Pipeline

This project includes two SageMaker pipelines that orchestrate the existing codebase:

1. **Training Pipeline**: Preprocessing → Feature Engineering → Training → Model Artifact
2. **Batch Inference Pipeline**: Preprocessing → Feature Engineering → Batch Inference → Predictions

### Step 1: Upload Data to S3

```bash
# Upload raw training and test data
aws s3 cp data/train.csv s3://YOUR-BUCKET/data/raw/train.csv
aws s3 cp data/test.csv s3://YOUR-BUCKET/data/raw/test.csv
```

### Step 2: Register Training Pipeline

```bash
cd ops/sagemaker

python run_training_pipeline.py \
    --role arn:aws:iam::ACCOUNT-ID:role/SageMakerExecutionRole \
    --region us-east-1 \
    --bucket YOUR-BUCKET \
    --instance-type ml.m5.xlarge
```

This creates and registers a SageMaker Pipeline named `MLTrainingPipeline` that:
- Takes raw CSV data from S3
- Runs preprocessing, feature engineering, and training
- Outputs model artifact to `s3://YOUR-BUCKET/pipeline-outputs/model/`

### Step 3: Execute Training Pipeline

```python
import boto3
from sagemaker.workflow.pipeline import Pipeline

# Get the pipeline
sm_client = boto3.client('sagemaker', region_name='us-east-1')
pipeline = Pipeline(name='MLTrainingPipeline', sagemaker_session=sagemaker_session)

# Start execution
execution = pipeline.start(
    parameters={
        'RawDataS3Uri': 's3://YOUR-BUCKET/data/raw'
    }
)
print(f"Pipeline execution ARN: {execution.arn}")
```

Or use the SageMaker Studio UI to trigger the pipeline execution.

### Step 4: Register Batch Inference Pipeline

```bash
python Run_batch_inference_pipeline.py \
    --role arn:aws:iam::ACCOUNT-ID:role/SageMakerExecutionRole \
    --region us-east-1 \
    --bucket YOUR-BUCKET \
    --instance-type ml.m5.xlarge
```

### Step 5: Execute Batch Inference Pipeline

```python
# Upload inference input data
# aws s3 cp inference_data.csv s3://YOUR-BUCKET/inference/input/

# Start inference pipeline
pipeline = Pipeline(name='MLBatchInferencePipeline', sagemaker_session=sagemaker_session)
execution = pipeline.start(
    parameters={
        'RawInputS3Uri': 's3://YOUR-BUCKET/inference/input/',
        'ModelArtifactS3Uri': 's3://YOUR-BUCKET/pipeline-outputs/model/'  # From training pipeline
    }
)
```

### Pipeline Architecture

Both pipelines use **ScriptProcessor** to run the existing scripts (`preprocess.py`, `feature_engineering.py`, `train.py`, `batch_inference.py`) without modification. The pipelines:

- Automatically handle S3 input/output
- Chain steps with proper dependencies
- Use pipeline parameters for flexible S3 URIs
- Upload the entire project (`source_dir`) so all scripts and modules are available

---

## Selected Metrics and Justification

### Metrics: RMSE and MAE

**Root Mean Squared Error (RMSE):**
- **Formula**: √(Σ(y_pred - y_true)² / n)
- **Why**: RMSE penalizes larger errors more heavily, making it ideal for trip duration prediction where large errors (e.g., predicting 10 minutes for a 60-minute trip) are significantly worse than small errors. It's in the same units as the target variable (seconds), making it interpretable.

**Mean Absolute Error (MAE):**
- **Formula**: Σ|y_pred - y_true| / n
- **Why**: MAE provides a complementary view by treating all errors equally. It's robust to outliers and gives a straightforward interpretation: "on average, predictions are off by X seconds."

**Selection Rationale:**
- Both metrics are regression standards and provide different perspectives
- RMSE is the primary metric for model selection (best model chosen by lowest validation RMSE)
- MAE provides additional insight into average error magnitude
- Both are reported for training and validation sets to detect overfitting

---

## Model Choices

### Models Evaluated

1. **RandomForestRegressor**
   - **Hyperparameters**:
     - `n_estimators=100`
     - `max_depth=20`
     - `min_samples_split=5`
     - `random_state=42`
   - **Rationale**: Ensemble method that handles non-linear relationships well, robust to outliers, and provides feature importance insights.

2. **GradientBoostingRegressor**
   - **Hyperparameters**:
     - `n_estimators=100`
     - `max_depth=5`
     - `learning_rate=0.1`
     - `random_state=42`
   - **Rationale**: Sequential boosting approach that can capture complex patterns through iterative refinement. Often performs well on structured tabular data.

### Model Selection Process

1. Both models are trained on the same train/validation split (80/20, `random_state=42`)
2. Each model is evaluated using RMSE and MAE on both training and validation sets
3. The model with the **lowest validation RMSE** is selected as the best model
4. The best model is retrained on the full training dataset before saving

### Best Model: GradientBoostingRegressor

**Justification:**
- GradientBoosting typically achieves better validation RMSE and MAE compared to RandomForest
- Better generalization: lower validation error relative to training error indicates less overfitting
- Gradient boosting's sequential learning approach captures complex feature interactions in trip duration prediction
- Well-suited for structured tabular data with engineered features (distance, time features, etc.)

**Model Outputs:**
- Saved model: `outputs/model.joblib` (can be loaded with `joblib.load()`)
- Metrics: `outputs/metrics.json` (contains RMSE/MAE for all models)
- Report: `outputs/model_report.txt` (human-readable summary)

---

## Team Member Contributions

**Sandy & Chloe** - Joint development of the complete MLOps pipeline:

- **Sandy**:
  - initial setup
  - Data preprocessing and cleaning pipeline
  - Batch inference implementation
  - SageMaker pipeline orchestration (training and inference pipelines)
    
- **Chloe**:
  - Feature engineering implementation (distance calculations, time features)
  - Model training infrastructure and evaluation framework
  - Local execution and Docker setup
  

**Collaborative Work:**
- Joint design of project structure and codebase organization
- Model selection and hyperparameter tuning decisions
- End-to-end pipeline testing and validation
- Code review and quality assurance

---

## Additional Notes

### File Organization

- **Large files** (datasets, model artifacts, predictions) are gitignored
- **Small outputs** like `metrics.json` and `model_report.txt` are committed for visibility
- The project uses `uv` for fast, reliable dependency management
- Python `src/` layout ensures clean package structure


