"""
Example client for interacting with the Academic Notice RAG API.

Demonstrates how to use the API endpoints with both streaming and non-streaming responses.
"""
import requests
import json

# API base URL (adjust if running on different host/port)
BASE_URL = "http://localhost:8000"


def health_check():
    """Check if the API is running."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.json()


def query_notices(query: str, stream: bool = False):
    """
    Query the RAG system.
    
    Args:
        query: The question to ask
        stream: Whether to use streaming response (default: False)
    """
    if stream:
        # Streaming query
        response = requests.post(
            f"{BASE_URL}/v1/query/stream",
            json={"query": query, "stream": True},
            stream=True
        )
        print(f"\nStreaming response for: '{query}'")
        print("-" * 50)
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
        print("\n")
    else:
        # Non-streaming query
        response = requests.post(
            f"{BASE_URL}/v1/query",
            json={"query": query, "stream": False}
        )
        data = response.json()
        print(f"\nResponse for: '{query}'")
        print("-" * 50)
        print(data["response"])
        print("\nSources:")
        for i, source in enumerate(data["sources"], 1):
            print(f"  {i}. {source['content']}")
            print(f"     Metadata: {source['metadata']}")
        print()


def upload_document(file_path: str):
    """
    Upload a document to the RAG system.
    
    Args:
        file_path: Path to the document (image or PDF)
    """
    with open(file_path, "rb") as f:
        files = {"file": (file_path.split("/")[-1], f, "application/octet-stream")}
        response = requests.post(f"{BASE_URL}/v1/documents/upload", files=files)
    
    print(f"\nUpload result for: {file_path}")
    print("-" * 50)
    print(json.dumps(response.json(), indent=2))
    print()


def get_categories():
    """Get available notice categories."""
    response = requests.get(f"{BASE_URL}/v1/categories")
    print("\nAvailable Categories:")
    print("-" * 50)
    print(json.dumps(response.json(), indent=2))
    print()


def main():
    """Run example API calls."""
    print("=" * 60)
    print("Academic Notice RAG API - Example Client")
    print("=" * 60)
    
    # 1. Health check
    health_check()
    
    # 2. Get categories
    get_categories()
    
    # 3. Query examples (non-streaming)
    query_notices("What are the upcoming exam dates?")
    query_notices("Tell me about holiday announcements")
    
    # 4. Query example (streaming)
    query_notices("What events are scheduled this month?", stream=True)
    
    # 5. Upload document example (uncomment and provide a real file path)
    # upload_document("path/to/your/notice.pdf")
    # upload_document("path/to/your/notice.jpg")


if __name__ == "__main__":
    main()
