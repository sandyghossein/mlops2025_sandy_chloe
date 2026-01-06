#!/usr/bin/env python3
"""
SageMaker Training Pipeline Orchestration Script.

This script creates and registers a SageMaker Pipeline that:
1. Preprocesses raw data
2. Engineers features
3. Trains a model
4. Outputs model artifact to S3
"""

import argparse
from pathlib import Path

import boto3
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
    Creates a SageMaker Pipeline for training.
    
    Args:
        role: IAM role ARN for SageMaker execution
        bucket: S3 bucket name for storing artifacts
        region: AWS region
        instance_type: Instance type for processing/training jobs
        image_uri: Optional custom Docker image URI (if None, uses SageMaker default)
    
    Returns:
        Pipeline object
    """
    # Pipeline parameters
    raw_data_s3_uri = ParameterString(
        name="RawDataS3Uri",
        default_value=f"s3://{bucket}/data/raw",
    )
    
    # S3 paths for outputs
    base_path = f"s3://{bucket}/pipeline-outputs"
    
    # Get SageMaker session
    from sagemaker.session import Session
    sagemaker_session = Session(boto3.Session(region_name=region))
    
    # Use default SKLearn image if not provided
    if image_uri is None:
        from sagemaker import image_uris
        image_uri = image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.0-1",
            instance_type=instance_type,
        )
    
    # Get project root directory (parent of ops/)
    project_root = Path(__file__).parent.parent.parent
    
    # ========== STEP 1: PREPROCESSING ==========
    # Create a wrapper script to handle file copying for preprocess.py
    # Since preprocess.py expects hardcoded paths, we need to copy files first
    wrapper_dir = project_root / "ops" / "sagemaker"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    preprocess_wrapper_path = wrapper_dir / "_preprocess_wrapper.py"
    preprocess_wrapper_code = """#!/usr/bin/env python3
import subprocess
import sys
import shutil
from pathlib import Path

# Create data directory and copy files
Path("data").mkdir(exist_ok=True)
shutil.copy("/opt/ml/processing/input/train/train.csv", "data/train.csv")
shutil.copy("/opt/ml/processing/input/test/test.csv", "data/test.csv")

# Run the original preprocessing script
sys.path.insert(0, str(Path("scripts").absolute()))
from scripts.preprocess import main as preprocess_main
preprocess_main()

# Copy outputs to SageMaker output directory
Path("/opt/ml/processing/output/data/processed").mkdir(parents=True, exist_ok=True)
shutil.copytree("data/processed", "/opt/ml/processing/output/data/processed", dirs_exist_ok=True)
"""
    preprocess_wrapper_path.write_text(preprocess_wrapper_code)
    
    # Make path relative to project root for ScriptProcessor
    preprocess_wrapper_relative = preprocess_wrapper_path.relative_to(project_root)
    
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
        name="PreprocessData",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                source=Join(values=[raw_data_s3_uri, "/train.csv"], on=""),
                destination="/opt/ml/processing/input/train",
            ),
            ProcessingInput(
                source=Join(values=[raw_data_s3_uri, "/test.csv"], on=""),
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="cleaned_data",
                source="/opt/ml/processing/output/data/processed",
                destination=f"{base_path}/preprocessed",
            ),
        ],
        code=str(preprocess_wrapper_relative),
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
        name="FeatureEngineering",
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
    
    # ========== STEP 3: TRAINING ==========
    train_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        source_dir=str(project_root),  # Upload entire project so scripts/ and src/ are available
    )
    
    train_step = ProcessingStep(
        name="TrainModel",
        processor=train_processor,
        inputs=[
            ProcessingInput(
                source=feature_step.properties.ProcessingOutputConfig.Outputs[
                    "features"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/features",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "cleaned_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/cleaned",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output",
                destination=f"{base_path}/model",
            ),
        ],
        code="scripts/train.py",
        job_arguments=[
            "--features_train", "/opt/ml/processing/input/features/features_train.parquet",
            "--clean_train", "/opt/ml/processing/input/cleaned/train_clean.csv",
            "--out_dir", "/opt/ml/processing/output",
        ],
        depends_on=[feature_step],
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name="MLTrainingPipeline",
        parameters=[raw_data_s3_uri],
        steps=[preprocess_step, feature_step, train_step],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Create and register SageMaker training pipeline")
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

