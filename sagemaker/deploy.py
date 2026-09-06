import time

import boto3


# ============================================================
# Configuration
# ============================================================

AWS_REGION = "us-east-1"

ROLE_NAME = "BankGoodCreditSagemakerExecutionRole"

MODEL_DATA = (
    "s3://bank-goodcredit-mlops-480151323504/"
    "models/bank-goodcredit/model.tar.gz"
)

# Corrected single-platform Linux/AMD64 ECR image
IMAGE_URI = (
    "480151323504.dkr.ecr.us-east-1.amazonaws.com/"
    "bank-goodcredit-risk-sagemaker:latest"
)

ENDPOINT_NAME = "bank-goodcredit-risk-endpoint"

SERVERLESS_MEMORY_MB = 2048
MAX_CONCURRENCY = 5


# ============================================================
# AWS clients
# ============================================================

session = boto3.Session(region_name=AWS_REGION)

iam = session.client("iam")
sm = session.client("sagemaker")


# ============================================================
# Get SageMaker execution role
# ============================================================

def get_role_arn():
    response = iam.get_role(
        RoleName=ROLE_NAME
    )

    return response["Role"]["Arn"]


# ============================================================
# Wait for endpoint
# ============================================================

def wait_for_endpoint(endpoint_name):

    print("\nWaiting for endpoint to become InService...")

    while True:

        response = sm.describe_endpoint(
            EndpointName=endpoint_name
        )

        status = response["EndpointStatus"]

        print(f"Endpoint status: {status}")

        if status == "InService":
            return response

        if status == "Failed":

            reason = response.get(
                "FailureReason",
                "No failure reason returned by SageMaker."
            )

            raise RuntimeError(
                "SageMaker endpoint failed.\n"
                f"Reason: {reason}"
            )

        time.sleep(20)


# ============================================================
# Main deployment
# ============================================================

def main():

    timestamp = time.strftime(
        "%Y%m%d-%H%M%S"
    )

    model_name = (
        f"bank-goodcredit-risk-model-{timestamp}"
    )

    endpoint_config_name = (
        f"bank-goodcredit-risk-config-{timestamp}"
    )

    # --------------------------------------------------------
    # Get execution role
    # --------------------------------------------------------

    role_arn = get_role_arn()

    print("\n========================================")
    print("Bank GoodCredit SageMaker Deployment")
    print("========================================")

    print("\nUsing SageMaker execution role:")
    print(role_arn)

    print("\nUsing model artifact:")
    print(MODEL_DATA)

    print("\nUsing custom ECR container:")
    print(IMAGE_URI)

    # --------------------------------------------------------
    # 1. Create SageMaker model
    # --------------------------------------------------------

    print("\nCreating SageMaker model...")

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": MODEL_DATA,
        },
    )

    print(
        f"Model created: {model_name}"
    )

    # --------------------------------------------------------
    # 2. Create Serverless endpoint configuration
    # --------------------------------------------------------

    print(
        "\nCreating SageMaker Serverless "
        "endpoint configuration..."
    )

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB":
                        SERVERLESS_MEMORY_MB,
                    "MaxConcurrency":
                        MAX_CONCURRENCY,
                },
            }
        ],
    )

    print(
        "Endpoint configuration created: "
        f"{endpoint_config_name}"
    )

    # --------------------------------------------------------
    # 3. Create SageMaker Serverless endpoint
    # --------------------------------------------------------

    print(
        "\nCreating SageMaker "
        "Serverless endpoint..."
    )

    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=endpoint_config_name,
    )

    print(
        "Endpoint creation started: "
        f"{ENDPOINT_NAME}"
    )

    # --------------------------------------------------------
    # 4. Wait for endpoint
    # --------------------------------------------------------

    response = wait_for_endpoint(
        ENDPOINT_NAME
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print("\n========================================")
    print("DEPLOYMENT SUCCESSFUL")
    print("========================================")

    print(
        f"Endpoint name: "
        f"{ENDPOINT_NAME}"
    )

    print(
        f"Status: "
        f"{response['EndpointStatus']}"
    )

    print(
        f"Model name: "
        f"{model_name}"
    )

    print(
        f"Endpoint config: "
        f"{endpoint_config_name}"
    )

    print("\nContainer image:")
    print(IMAGE_URI)


if __name__ == "__main__":
    main()