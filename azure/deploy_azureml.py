"""Azure ML deployment scaffold; requires Azure identity, workspace, and endpoint config."""
import os

def main():
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise SystemExit("Install azure-ai-ml and azure-identity before using this scaffold.") from exc
    required = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    client = MLClient(DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"])
    print("Connected to Azure ML workspace:", client.workspace_name)
    print("Scaffold only: register the trained model, create an environment and endpoint,")
    print("then deploy with reviewed scoring code and a secured managed identity.")

if __name__ == "__main__":
    main()
