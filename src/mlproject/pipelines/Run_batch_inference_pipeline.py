#!/usr/bin/env python3
"""
SageMaker Batch Inference Pipeline Orchestration Script.

This script creates and registers a SageMaker Pipeline that:
1. Preprocesses raw CSV input
2. Engineers features
3. Loads trained model from S3
4. Runs batch inference
5. Writes predictions to S3
"""

import argparse
import boto3
from pathlib import Path

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.functions import Join


def get_pipeline(
    role: str,
    bucket: str,
    region: str,
    instance_type: str = "ml.m5.xlarge",
    image_uri: str = None,
):
    """
    Creates a SageMaker Pipeline for batch inference.
    
    Args:
        role: IAM role ARN for SageMaker execution
        bucket: S3 bucket name for storing artifacts
        region: AWS region
        instance_type: Instance type for processing jobs
        image_uri: Optional custom Docker image URI (if None, uses SageMaker default)
    
    Returns:
        Pipeline object
    """
    # Pipeline parameters
    raw_input_s3_uri = ParameterString(
        name="RawInputS3Uri",
        default_value=f"s3://{bucket}/inference/input",
    )
    
    model_artifact_s3_uri = ParameterString(
        name="ModelArtifactS3Uri",
        default_value=f"s3://{bucket}/models",
    )
    
    # S3 paths for outputs
    base_path = f"s3://{bucket}/pipeline-outputs/inference"
    
    # Get SageMaker session
    from sagemaker.session import Session
    sagemaker_session = Session(boto3.Session(region_name=region))
    
    # Get project root directory (parent of ops/)
    project_root = Path(__file__).parent.parent.parent
    
    # Use default SKLearn image if not provided
    if image_uri is None:
        from sagemaker import image_uris
        image_uri = image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.0-1",
            instance_type=instance_type,
        )
    
    # ========== STEP 1: PREPROCESSING ==========
    # Create a wrapper script to handle file copying for preprocess.py
    wrapper_dir = Path(__file__).parent
    preprocess_wrapper_path = wrapper_dir / "_preprocess_inference_wrapper.py"
    preprocess_wrapper_code = """#!/usr/bin/env python3
import subprocess
import sys
import shutil
import glob
from pathlib import Path

# Create data directory and copy files
Path("data").mkdir(exist_ok=True)
# Copy all CSV files from input
csv_files = glob.glob("/opt/ml/processing/input/raw/*.csv")
for csv_file in csv_files:
    shutil.copy(csv_file, f"data/{Path(csv_file).name}")

# Ensure we have train.csv and test.csv (for inference, test.csv is the input)
if not Path("data/train.csv").exists():
    # If only one CSV file, use it as both train and test
    csv_files = list(Path("data").glob("*.csv"))
    if csv_files:
        shutil.copy(csv_files[0], "data/train.csv")
        shutil.copy(csv_files[0], "data/test.csv")

# Run the original preprocessing script
sys.path.insert(0, str(Path("scripts").absolute()))
from scripts.preprocess import main as preprocess_main
preprocess_main()

# Copy outputs to SageMaker output directory
Path("/opt/ml/processing/output/data/processed").mkdir(parents=True, exist_ok=True)
shutil.copytree("data/processed", "/opt/ml/processing/output/data/processed", dirs_exist_ok=True)
"""
    preprocess_wrapper_path.write_text(preprocess_wrapper_code)
    
    preprocess_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        source_dir=str(project_root),  # Upload entire project so scripts/ and src/ are available
    )
    
    preprocess_step = ProcessingStep(
        name="PreprocessInferenceData",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                source=raw_input_s3_uri,
                destination="/opt/ml/processing/input/raw",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="cleaned_data",
                source="/opt/ml/processing/output/data/processed",
                destination=f"{base_path}/preprocessed",
            ),
        ],
        code=str(preprocess_wrapper_path.relative_to(project_root)),
    )
    
    # ========== STEP 2: FEATURE ENGINEERING ==========
    feature_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        source_dir=str(project_root),  # Upload entire project so scripts/ and src/ are available
    )
    
    feature_step = ProcessingStep(
        name="FeatureEngineeringInference",
        processor=feature_processor,
        inputs=[
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "cleaned_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/cleaned",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="features",
                source="/opt/ml/processing/output",
                destination=f"{base_path}/features",
            ),
        ],
        code="scripts/feature_engineering.py",
        job_arguments=[
            "--clean_train", "/opt/ml/processing/input/cleaned/train_clean.csv",
            "--clean_test", "/opt/ml/processing/input/cleaned/test_clean.csv",
            "--out_dir", "/opt/ml/processing/output",
        ],
        depends_on=[preprocess_step],
    )
    
    # ========== STEP 3: BATCH INFERENCE ==========
    # Create a wrapper script to handle model extraction and run batch inference
    inference_wrapper_path = wrapper_dir / "_inference_wrapper.py"
    inference_wrapper_code = """#!/usr/bin/env python3
import subprocess
import sys
import tarfile
from pathlib import Path

# Extract model artifact (SageMaker training outputs model.tar.gz)
model_input_dir = Path("/opt/ml/processing/input/model")
model_tar = list(model_input_dir.glob("*.tar.gz"))
if model_tar:
    model_extract_dir = Path("/opt/ml/processing/model_extracted")
    model_extract_dir.mkdir(exist_ok=True)
    with tarfile.open(model_tar[0], "r:gz") as tar:
        tar.extractall(path=model_extract_dir)
    # Find the model file (usually model.joblib or similar)
    model_files = list(model_extract_dir.rglob("*.joblib")) + list(model_extract_dir.rglob("*.pkl"))
    if model_files:
        model_path = model_files[0]
    else:
        # If no joblib/pkl, use the first file in the extracted directory
        all_files = [f for f in model_extract_dir.rglob("*") if f.is_file()]
        if all_files:
            model_path = all_files[0]
        else:
            raise FileNotFoundError("No model file found in extracted archive")
else:
    # If no tar.gz, look for direct model file
    model_files = list(model_input_dir.rglob("*.joblib")) + list(model_input_dir.rglob("*.pkl"))
    if model_files:
        model_path = model_files[0]
    else:
        raise FileNotFoundError("No model file found in model input directory")

# Run batch inference script
sys.path.insert(0, str(Path("scripts").absolute()))
from scripts.batch_inference import main as inference_main

# Set up arguments for batch_inference.py
sys.argv = [
    "batch_inference.py",
    "--features_test", "/opt/ml/processing/input/features/features_test.parquet",
    "--model_path", str(model_path),
    "--out_dir", "/opt/ml/processing/output",
]

inference_main()
"""
    inference_wrapper_path.write_text(inference_wrapper_code)
    
    inference_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        source_dir=str(project_root),  # Upload entire project so scripts/ and src/ are available
    )
    
    inference_step = ProcessingStep(
        name="BatchInference",
        processor=inference_processor,
        inputs=[
            ProcessingInput(
                source=feature_step.properties.ProcessingOutputConfig.Outputs[
                    "features"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/features",
            ),
            ProcessingInput(
                source=model_artifact_s3_uri,
                destination="/opt/ml/processing/input/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output",
                destination=f"{base_path}/predictions",
            ),
        ],
        code=str(inference_wrapper_path.relative_to(project_root)),
        depends_on=[feature_step],
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name="MLBatchInferencePipeline",
        parameters=[raw_input_s3_uri, model_artifact_s3_uri],
        steps=[preprocess_step, feature_step, inference_step],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Create and register SageMaker batch inference pipeline")
    parser.add_argument("--role", type=str, required=True, help="IAM role ARN for SageMaker")
    parser.add_argument("--region", type=str, required=True, help="AWS region")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge", help="Instance type")
    parser.add_argument("--image-uri", type=str, default=None, help="Custom Docker image URI (optional)")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = get_pipeline(
        role=args.role,
        bucket=args.bucket,
        region=args.region,
        instance_type=args.instance_type,
        image_uri=args.image_uri,
    )
    
    # Register/update pipeline
    pipeline.upsert(role_arn=args.role)
    print(f"✅ Pipeline '{pipeline.name}' registered successfully!")
    print(f"Pipeline ARN: {pipeline.arn}")


if __name__ == "__main__":
    main()

