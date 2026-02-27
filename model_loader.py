"""
Model loader for local HuggingFace transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from info import get_model_path, is_local_model


class ModelLoader:
    """
    Model loader for local HuggingFace models.
    """

    def __init__(self, model_name, gpu_id=0):
        """
        Initialize the model loader.

        Args:
            model_name (str): Model name or shortcut
            gpu_id (int): GPU device ID for local models
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_local_model()

    def _load_local_model(self):
        """Load local HuggingFace model."""
        model_path = get_model_path(self.model_name)

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

            # Set pad token if not exists
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            self.model.eval()
            print(f"Model {self.model_name} loaded successfully")

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate(self, prompt, max_tokens=400, temperature=0.0, top_p=1.0):
        """
        Generate text using the loaded model.

        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter

        Returns:
            str: Generated text
        """
        # Format prompt based on model type
        formatted_prompt = self._format_prompt(prompt)

        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1
            )

        # Decode output
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Extract only the new generated part
        if formatted_prompt in generated_text:
            generated_text = generated_text[len(formatted_prompt):].strip()

        return generated_text

    def _format_prompt(self, prompt):
        """Format prompt based on model type."""
        model_lower = self.model_name.lower()

        if "llama" in model_lower:
            return f"[INST] {prompt} [/INST]"
        elif "mistral" in model_lower:
            return f"<s>[INST] {prompt} [/INST]"
        elif "qwen" in model_lower or "deepseek" in model_lower:
            return f"User: {prompt}\nAssistant:"
        else:
            # Default format
            return prompt


def test_model_loader():
    """Test function for the model loader."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    try:
        loader = ModelLoader(model_name)
        prompt = "What is 2+2?"
        response = loader.generate(prompt, max_tokens=50)
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_model_loader()
