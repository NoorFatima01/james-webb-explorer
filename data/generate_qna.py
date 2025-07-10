import pandas as pd
import requests
import json
import time
from typing import List, Dict
import re

class JWSTChatGenerator:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "gemma3:latest"
        
    def check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model in model_names:
                    print(f"✅ Ollama is running and {self.model} is available")
                    return True
                else:
                    print(f"❌ Model {self.model} not found. Available models: {model_names}")
                    return False
            else:
                print("❌ Ollama is not responding")
                return False
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            return False
    
    def generate_conversations(self, title: str, summary: str, content: str, num_conversations: int = 3) -> List[Dict]:
        """Generate conversation pairs from article data"""
        
        # Combine content fields
        full_content = f"{summary}"
        if content:
            full_content += f"\n\n{content}"
        
        prompt = f"""Based on this James Webb Space Telescope article, create {num_conversations} different natural conversation exchanges between a curious user and a knowledgeable assistant.

Title: {title}
Content: {full_content}

For each conversation, create realistic questions a user might ask and detailed, informative responses. Make the questions varied (what, how, why, tell me about, explain, etc.).

Format EXACTLY like this:
CONVERSATION 1:
User: [natural question]
Assistant: [detailed, informative response based on the article]

CONVERSATION 2:
User: [different type of question]
Assistant: [detailed response]

CONVERSATION 3:
User: [another varied question]
Assistant: [comprehensive response]

Make responses detailed and educational, using information from the article. Don't mention "based on the article" in responses."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
            )
            
            if response.status_code == 200:
                generated_text = response.json()["response"]
                return self.parse_conversations(generated_text)
            else:
                print(f"Error generating conversations: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in generate_conversations: {e}")
            return []
    
    def parse_conversations(self, generated_text: str) -> List[Dict]:
        """Parse the generated text into structured conversation pairs"""
        conversations = []
        
        # Split by conversation markers
        conv_pattern = r'CONVERSATION \d+:'
        conv_sections = re.split(conv_pattern, generated_text)
        
        for section in conv_sections[1:]:  # Skip first empty section
            # Extract user and assistant parts
            user_match = re.search(r'User:\s*(.+?)(?=\nAssistant:|\nUser:|$)', section, re.DOTALL)
            assistant_match = re.search(r'Assistant:\s*(.+?)(?=\nUser:|$)', section, re.DOTALL)
            
            if user_match and assistant_match:
                user_text = user_match.group(1).strip()
                assistant_text = assistant_match.group(1).strip()
                
                if user_text and assistant_text:
                    conversations.append({
                        "messages": [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": assistant_text}
                        ]
                    })
        
        return conversations
    
    def process_csv(self, csv_file: str, output_file: str = "jwst_chat_data.jsonl"):
        """Process the entire CSV file and generate chat data"""
        
        if not self.check_ollama_connection():
            return False
        
        # Read CSV
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} articles from {csv_file}")
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
        
        all_conversations = []
        
        for idx, row in df.iterrows():
            print(f"\n📝 Processing article {idx + 1}/{len(df)}: {row['title'][:50]}...")
            
            # Combine content columns
            content_parts = []
            for col in df.columns:
                if col.startswith('content-') and pd.notna(row[col]):
                    content_parts.append(str(row[col]))
            
            combined_content = " ".join(content_parts)
            
            # Generate conversations
            conversations = self.generate_conversations(
                title=str(row['title']),
                summary=str(row['summary']) if pd.notna(row['summary']) else "",
                content=combined_content
            )
            
            if conversations:
                all_conversations.extend(conversations)
                print(f"  ✅ Generated {len(conversations)} conversations")
            else:
                print(f"  ❌ Failed to generate conversations")
            
            # Small delay to be nice to the local model
            time.sleep(1)
        
        # Save to JSONL format
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in all_conversations:
                    f.write(json.dumps(conv) + '\n')
            
            print(f"\n🎉 Success! Generated {len(all_conversations)} conversations")
            print(f"💾 Saved to: {output_file}")
            
            # Show sample
            if all_conversations:
                print("\n📋 Sample conversation:")
                sample = all_conversations[0]
                print(f"User: {sample['messages'][0]['content']}")
                print(f"Assistant: {sample['messages'][1]['content'][:200]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False

# Usage example
if __name__ == "__main__":
    generator = JWSTChatGenerator()
    
    # Replace with your CSV file path
    csv_file = "jwst_cleaned.csv"
    
    # Generate chat data
    generator.process_csv(csv_file, "jwst_chat_training_data.jsonl")
    
    print("\n🔧 Next steps:")
    print("1. Review the generated conversations in jwst_chat_training_data.jsonl")
    print("2. Use this file for fine-tuning your model")
    print("3. The format is ready for OpenAI fine-tuning or Hugging Face training")