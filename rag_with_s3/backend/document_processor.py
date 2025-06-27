
import os
import boto3
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your-s3-bucket-name") # Replace with your S3 bucket name
s3_client = boto3.client("s3")

def upload_file_to_s3(file_path: str, s3_key: str):
    """Uploads a file to S3."""
    s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    print(f"Uploaded {file_path} to s3://{S3_BUCKET_NAME}/{s3_key}")

def download_file_from_s3(s3_key: str, local_path: str):
    """Downloads a file from S3."""
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
    print(f"Downloaded s3://{S3_BUCKET_NAME}/{s3_key} to {local_path}")

def process_document(file_path: str):
    # Upload raw document to S3
    file_name = os.path.basename(file_path)
    s3_document_key = f"documents/{file_name}"
    upload_file_to_s3(file_path, s3_document_key)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the vectorstore to a temporary local path first
    vectorstore_name = os.path.splitext(file_name)[0]
    temp_vectorstore_path = f"/tmp/{vectorstore_name}"
    vectorstore.save_local(temp_vectorstore_path)

    # Upload FAISS index files to S3
    s3_vectorstore_prefix = f"vectorstores/{vectorstore_name}"
    for root, _, files in os.walk(temp_vectorstore_path):
        for f in files:
            local_file_path = os.path.join(root, f)
            s3_key = os.path.join(s3_vectorstore_prefix, f)
            upload_file_to_s3(local_file_path, s3_key)
    
    # Clean up temporary local files
    os.system(f"rm -rf {temp_vectorstore_path}")

    return vectorstore_name

def load_vectorstore(vectorstore_name: str):
    embeddings = OpenAIEmbeddings()
    
    # Download FAISS index files from S3 to a temporary local path
    temp_vectorstore_path = f"/tmp/{vectorstore_name}"
    os.makedirs(temp_vectorstore_path, exist_ok=True)

    s3_vectorstore_prefix = f"vectorstores/{vectorstore_name}"
    # List objects in the S3 prefix and download them
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_vectorstore_prefix)
    if 'Contents' not in response:
        raise FileNotFoundError(f"Vectorstore \'{vectorstore_name}\' not found in S3 bucket \'{S3_BUCKET_NAME}\'")

    for obj in response['Contents']:
        s3_key = obj['Key']
        local_file_path = os.path.join(temp_vectorstore_path, os.path.basename(s3_key))
        download_file_from_s3(s3_key, local_file_path)

    vectorstore = FAISS.load_local(temp_vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    
    # Clean up temporary local files
    os.system(f"rm -rf {temp_vectorstore_path}")

    return vectorstore


