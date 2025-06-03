# Import the create_repo function from the huggingface_hub library
from huggingface_hub import create_repo
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

create_repo("bank-customer-churn",  # Your dataset repo name
            repo_type="dataset",  # Specify this is a dataset
            private=False)  # Set to True if it should be private

api.upload_folder(
    folder_path="mlops/data",
    repo_id="praneeth232/bank-customer-churn",
    repo_type="dataset",
)
