"""Response generation using transformers"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Iterator

class ResponseGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize response generator"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response from prompt"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_streaming(self, prompt: str, max_new_tokens: int = 150) -> Iterator[str]:
        """Generate streaming response"""
        response = self.generate(prompt, max_new_tokens)
        
        # Simulate streaming by yielding words
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
