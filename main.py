import os
# Fix for OpenMP library conflict - must be at the VERY TOP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

from encode import QueryEncoder
from vector_db import VectorDB
from prompt_builder import PromptBuilder

# Try to use the alternative generator first to avoid OpenMP issues
try:
    from alternative_llm_generation import AlternativeLLMGenerator as LLMGenerator
    print("Using alternative LLM generator to avoid OpenMP conflicts...")
except ImportError:
    try:
        from llm_generation import LLMGenerator
        print("Using standard LLM generator...")
    except ImportError:
        print("LLM generator not available, running in retrieval-only mode...")
        LLMGenerator = None

class RAGSystem:
    def __init__(self, documents_path: str, llm_model_path: str = None):
        """Initialize the RAG system."""
        print("Initializing RAG System...")
        
        # Initialize components
        self.query_encoder = QueryEncoder()
        self.vector_db = VectorDB(documents_path)
        self.prompt_builder = PromptBuilder()
        
        # Initialize LLM if available
        self.llm_generation = None
        if LLMGenerator and llm_model_path and os.path.exists(llm_model_path):
            try:
                self.llm_generation = LLMGenerator(llm_model_path)
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
                print("Running in retrieval-only mode.")
        elif LLMGenerator and llm_model_path:
            print(f"Warning: Model file '{llm_model_path}' not found.")
            print("Running in retrieval-only mode.")
        
        print("RAG System initialized successfully! 🚀")
    
    def answer_question(self, question: str, top_k: int = 3) -> dict:
        """Process a question through the RAG pipeline."""
        print(f"\n🔍 Processing question: '{question}'")
        
        try:
            # Step 1: Encode query
            print("1. Encoding query...")
            query_embedding = self.query_encoder.encode(question)
            
            # Step 2: Vector search
            print("2. Performing vector search...")
            distances, indices = self.vector_db.search(query_embedding, top_k)
            
            # Step 3: Document retrieval
            print("3. Retrieving documents...")
            retrieved_docs = self.vector_db.get_documents_by_indices(indices)
            
            # Step 4: Prompt augmentation
            print("4. Building prompt...")
            prompt = self.prompt_builder.build_rag_prompt(question, retrieved_docs)
            
            # Step 5: LLM generation (if available)
            answer = ""
            if self.llm_generation:
                print("5. Generating answer with LLM...")
                answer = self.llm_generation.generate(prompt)
            else:
                answer = "LLM not available. Here are the retrieved documents:"
                for i, doc in enumerate(retrieved_docs):
                    answer += f"\n\nDocument {i+1}: {doc['text'][:100]}..."
            
            # Prepare results
            results = {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "distances": distances.tolist(),
                "indices": indices.tolist(),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
            }
            
            return results
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "retrieved_documents": [],
                "distances": [],
                "indices": [],
                "prompt_preview": ""
            }

def print_results(results: dict):
    """Print formatted RAG results."""
    print("\n" + "="*80)
    print("🤖 RAG SYSTEM - RESULTS")
    print("="*80)
    print(f"❓ QUESTION: {results['question']}")
    print(f"\n💡 ANSWER:")
    print(f"{results['answer']}")
    print(f"\n📄 RETRIEVED DOCUMENTS (Top {len(results['retrieved_documents'])}):")
    for i, doc in enumerate(results['retrieved_documents']):
        print(f"\n[{i+1}] Document ID: {doc['id']}, Distance: {results['distances'][i]:.4f}")
        print(f"    Text: {doc['text'][:100]}...")
    print(f"\n📊 SEARCH METRICS:")
    print(f"   Indices: {results['indices']}")
    print(f"   Distances: {[f'{d:.4f}' for d in results['distances']]}")
    print("="*80)

def interactive_mode(rag_system: RAGSystem, top_k: int = 3):
    """Run continuous interactive question-answering mode."""
    print("\n" + "="*60)
    print("🔄 Interactive RAG System - Ready for Questions!")
    print("="*60)
    print("💡 Type your questions and press Enter.")
    print("⏹️  Type 'quit', 'exit', or 'q' to exit.")
    print("🔄 Type 'clear' or 'cls' to clear the screen.")
    print("ℹ️  Type 'help' for available commands.")
    print("="*60)
    
    question_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n[{question_count + 1}] 🔍 Your question: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the RAG System! Goodbye!")
                break
                
            elif user_input.lower() in ['clear', 'cls']:
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            elif user_input.lower() in ['help', '?']:
                print("\n📋 Available commands:")
                print("  - Type your question to get an AI-powered answer")
                print("  - 'quit', 'exit', 'q': Exit the system")
                print("  - 'clear', 'cls': Clear the screen")
                print("  - 'help', '?': Show this help message")
                continue
                
            elif not user_input:
                print("💡 Please enter a question or command.")
                continue
            
            # Process the question
            question_count += 1
            results = rag_system.answer_question(user_input, top_k)
            print_results(results)
            
            # Show ready for next question
            print(f"\n✅ Ready for question #{question_count + 1}...")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Session interrupted. Type 'quit' to exit or continue asking questions.")
        except EOFError:
            print("\n\n👋 End of input detected. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Please try again or type 'quit' to exit.")

def single_question_mode(rag_system: RAGSystem, question: str, top_k: int = 3):
    """Process a single question and then enter interactive mode."""
    print(f"\n🧪 Processing single question: '{question}'")
    results = rag_system.answer_question(question, top_k)
    print_results(results)
    
    # After processing the single question, enter interactive mode
    print("\n" + "="*60)
    print("🔄 Continuing to interactive mode...")
    print("="*60)
    interactive_mode(rag_system, top_k)

def main():
    parser = argparse.ArgumentParser(
        description='🤖 RAG System for Document Question Answering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data preprocessed_documents.json --model tinyllama.gguf
  %(prog)s --data preprocessed_documents.json --question "What causes squirrels to lose fur?"
  %(prog)s --data preprocessed_documents.json --interactive --top-k 5
        """
    )
    
    parser.add_argument('--data', '-d', 
                       default='preprocessed_documents.json',
                       help='Path to preprocessed documents JSON file (default: preprocessed_documents.json)')
    parser.add_argument('--model', '-m',
                       help='Path to LLM model file (optional)')
    parser.add_argument('--question', '-q', 
                       type=str,
                       help='Single question to process before entering interactive mode')
    parser.add_argument('--top-k', '-k',
                       type=int, default=3,
                       help='Number of documents to retrieve (default: 3)')
    parser.add_argument('--interactive', '-i',
                       action='store_true',
                       help='Start in interactive mode immediately')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file '{args.data}' not found.")
        print("💡 Please run data_preprocess.py first to generate preprocessed_documents.json")
        sys.exit(1)
    
    # Check if model file exists (if provided)
    if args.model and not os.path.exists(args.model):
        print(f"⚠️  Warning: Model file '{args.model}' not found.")
        print("💡 You can download it with:")
        print("   wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        print("💡 Continuing in retrieval-only mode...")
        # Don't exit, just continue without the model
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(args.data, args.model)
        
        # Determine mode
        if args.question:
            # Process single question then continue to interactive mode
            single_question_mode(rag_system, args.question, args.top_k)
        else:
            # Start in interactive mode
            interactive_mode(rag_system, args.top_k)
            
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()