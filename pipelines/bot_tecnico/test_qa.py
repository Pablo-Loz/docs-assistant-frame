"""
Test script for the Multi-Agent Technical Manual RAG Pipeline.

Run this script to validate the pipeline works without the Open WebUI interface.

Usage:
    python test_qa.py                    # Run default tests
    python test_qa.py "Your question"    # Test specific question
    python test_qa.py -i                 # Interactive mode
    python test_qa.py --products         # List available products
"""

import sys
from pathlib import Path

# Add parent directory to path for package imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

from bot_tecnico import Pipeline


def list_products(pipeline: Pipeline) -> None:
    """List available products discovered from ChromaDB."""
    print("\n" + "=" * 60)
    print("Available Products")
    print("=" * 60)

    if not pipeline._available_products:
        print("No products found. Run ingest.py first.")
        return

    for key in pipeline._available_products:
        product = pipeline._products.get(key)
        desc = product.description if product else ""
        print(f"  - {key}: {desc}")

    print("=" * 60)


def test_pipeline(question: str = None):
    """Test the pipeline with sample questions."""
    print("=" * 60)
    print("Multi-Agent Technical Manual RAG Pipeline - Test")
    print("=" * 60)

    # Initialize pipeline
    print("\n[1] Initializing pipeline...")
    try:
        pipeline = Pipeline()
        pipeline._ensure_initialized()
        print("    Pipeline initialized successfully")
        print(f"    Products: {pipeline._available_products}")
    except FileNotFoundError as e:
        print(f"    Error: {e}")
        print("\n    Please run 'python ingest.py' first to create the vector database.")
        return
    except Exception as e:
        print(f"    Error: {e}")
        return

    # Check document count
    doc_count = pipeline._vector_store.document_count
    print(f"    Connected to ChromaDB ({doc_count} documents)")

    # Test questions covering different scenarios
    if question:
        test_cases = [{"query": question, "scenario": "User provided"}]
    else:
        test_cases = [
            {
                "query": "What are the PCGH technical specifications?",
                "scenario": "Clear product mention (English)",
            },
            {
                "query": "Cuales son las especificaciones del PDWA?",
                "scenario": "Clear product mention (Spanish)",
            },
            {
                "query": "What is the maximum pressure?",
                "scenario": "Ambiguous query (should ask for clarification)",
            },
            {
                "query": "How do I perform maintenance?",
                "scenario": "Generic query (low confidence)",
            },
        ]

    # Run tests
    print("\n[2] Running test queries...")
    print("-" * 60)

    for i, test in enumerate(test_cases, 1):
        q = test["query"]
        scenario = test.get("scenario", "")

        print(f"\n Test {i}: {scenario}")
        print(f" Query: {q}")
        print("-" * 40)

        try:
            # Simulate pipe call (synchronous)
            response = pipeline.pipe(user_message=q, messages=[])
            print(f"\n Response:\n{response}")
        except Exception as e:
            print(f"\n Error: {e}")

        print("\n" + "=" * 60)

    print("\n[3] Test completed!")


def test_clarification_flow():
    """Test the clarification flow specifically."""
    print("=" * 60)
    print("Testing Clarification Flow")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    try:
        pipeline = Pipeline()
        pipeline._ensure_initialized()
        print(f"Products: {pipeline._available_products}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Step 1: Ambiguous query
    print("\n[Step 1] Sending ambiguous query...")
    ambiguous_query = "What is the maximum flow rate?"
    print(f"Query: {ambiguous_query}")

    response1 = pipeline.pipe(user_message=ambiguous_query, messages=[])
    print(f"\nResponse:\n{response1}")

    # Step 2: Simulate user clarification
    print("\n[Step 2] Simulating user response with product selection...")

    # Build conversation history as Open WebUI would
    messages = [
        {"role": "user", "content": ambiguous_query},
        {"role": "assistant", "content": response1},
    ]

    clarification = "PCGH"
    print(f"User clarification: {clarification}")

    response2 = pipeline.pipe(user_message=clarification, messages=messages)
    print(f"\nFinal Response:\n{response2}")

    print("\n" + "=" * 60)


def interactive_mode():
    """Run in interactive mode for continuous testing."""
    print("=" * 60)
    print("Multi-Agent Pipeline - Interactive Mode")
    print("Commands: 'exit', 'products', 'clear'")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    try:
        pipeline = Pipeline()
        pipeline._ensure_initialized()
        doc_count = pipeline._vector_store.document_count
        print(f"Ready ({doc_count} documents, {len(pipeline._available_products)} products)")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("-" * 60)

    # Maintain conversation history
    messages = []

    while True:
        try:
            question = input("\nYour question: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "salir", "q"):
                print("\nGoodbye!")
                break

            if question.lower() == "products":
                list_products(pipeline)
                continue

            if question.lower() == "clear":
                messages = []
                print("Conversation cleared.")
                continue

            print("\nProcessing...")
            response = pipeline.pipe(user_message=question, messages=messages)
            print(f"\nResponse:\n{response}")

            # Update conversation history
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": response})

            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg in ("-i", "--interactive"):
            interactive_mode()
        elif arg == "--products":
            pipeline = Pipeline()
            pipeline._ensure_initialized()
            list_products(pipeline)
        elif arg == "--clarification":
            test_clarification_flow()
        else:
            # Use command line argument as question
            question = " ".join(sys.argv[1:])
            test_pipeline(question)
    else:
        test_pipeline()


if __name__ == "__main__":
    main()
