"""
Simple helper script to test direct PDF download functionality
"""
import os
import sys
import base64
import json
import shutil

def create_test_pdf():
    """Create a simple test PDF file"""
    test_pdf_content = (
        b'%PDF-1.4\n'
        b'1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n'
        b'2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n'
        b'3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>/Contents 4 0 R/Parent 2 0 R>>endobj\n'
        b'4 0 obj<</Length 21>>stream\nBT /F1 12 Tf 72 720 Td (Test PDF) Tj ET\nendstream\nendobj\n'
        b'xref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000192 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n264\n%%EOF\n'
    )
    return test_pdf_content

def ensure_history_entry():
    """Ensure there's at least one entry in history for testing"""
    history_path = 'resume_history.json'
    
    # Create or load history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    
    # Add test entry if history is empty
    if not history:
        test_pdf = create_test_pdf()
        test_id = "test123"
        entry = {
            "id": test_id,
            "timestamp": "2023-01-01T00:00:00",
            "display_text": "<p>Test resume</p>",
            "resume": base64.b64encode(test_pdf).decode('utf-8')
        }
        history.append(entry)
        
        # Save history
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        print(f"Added test entry with ID: {test_id}")
        return test_id
    else:
        print(f"Using existing entry with ID: {history[0]['id']}")
        return history[0]['id']

def main():
    # Create test history entry
    test_id = ensure_history_entry()
    
    # Print instructions
    print("\nTest PDF Download")
    print("================")
    print(f"1. Start server: python resumeBuilder.py")
    print(f"2. In browser, navigate to: http://localhost:8000/download/{test_id}")
    print(f"3. PDF should download automatically")
    print("\nIf using curl: curl -v http://localhost:8000/download/{test_id} -o test.pdf")

if __name__ == "__main__":
    main() 