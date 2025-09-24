import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Optional
import logging
import os
from app.config.settings import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Take model settings from settings module
        self.model_name = settings.llm_model_name
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.top_p = settings.top_p

        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Take llm settings from settings module
        self.llm_config = settings.get_llm_settings()

        logger.info(f"LLM Service initialized - Model: {self.model_name}, Device: {self.device}")

    async def initialize(self):
        # Load model and tokenizer
        try:
            logger.info(f"Loading LLM model: {self.model_name}")
            logger.info(f"Settings - Max Tokens: {self.max_tokens}, Temperature: {self.temperature}, Top_p: {self.top_p}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None)
            
            # Create pipeline
            self.pipeline = pipeline(
                'text-generation', 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=-1, # Use CPU
                return_full_text=False)
            
            logger.info("LLM Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise
        
    async def generate_story(self, prompt: str, max_length: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        if not self.pipeline:
            await self.initialize()
        
        try:
            # Get the parameters from settings (use override if available)
            max_length = max_length or self.max_tokens
            temperature = temperature or self.temperature
            top_p = top_p or self.top_p

            logger.info(f"Generate Story - Length: {max_length}, Temp: {temperature}, Top-p: {top_p}")

            # Basic prompt format
            formatted_prompt = f"Story: {prompt}\n\nOnce upon a time"
            response = self.pipeline(
                formatted_prompt, 
                max_new_tokens=max_length, 
                temperature=temperature, 
                top_p=top_p, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1, # Recurrence prevention
                no_repeat_ngram_size=3 # N-gram repetition prevention
            )

            # Clean up the text
            generated_text = response[0]['generated_text'].strip()
            cleaned_text = self._clean_text(generated_text)

            logger.info(f"Generated story - Length: {len(cleaned_text)} characters")

            return cleaned_text

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return self._get_fallback_story(prompt)
    
    def _clean_text(self, text: str) -> str:
        # Remove extra spaces 
        text = ' '.join(text.split())

        # Minimum length check
        min_length = getattr(settings, 'min_story_length', 100)
        if len(text) < min_length:
            return self._get_fallback_story("general")
        
        # Maximum length check
        max_length = getattr(settings, 'max_story_length', 2000)
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.info(f"Text {max_length} shortened to characters")
        
        return text
    
    def _get_fallback_story(self, prompt: str) -> str:
        # Dynamic fallback stories by Prompt
        fallback_stories = {
            "adventure": self._create_adventure_story(),
            "friendship": self._create_friendship_story(),
            "animal": self._create_animal_story(),
            "fantasy": self._create_fantasy_story(),
            "default": self._create_default_story()
        }
        
        # Use whatever type is in the prompt
        prompt_lower = prompt.lower()
        for genre, story in fallback_stories.items():
            if genre in prompt_lower:
                logger.info(f"Fallback story used: {genre}")
                return story
        
        logger.info("Default fallback story used")
        return fallback_stories["default"]
    
    def _create_adventure_story(self) -> str:
        return """
                # The Adventure of Brave Cat Mırmır
                Once upon a time, there lived a very brave cat named Mırmır. Every day, Mırmır sought new adventures in the garden.
                One morning, Mırmır saw something sparkling in the farthest corner of the garden. Curious, he approached and realized it was a beautiful, colorful butterfly.
                "Hello, little cat," said the butterfly gently. "I'm looking for the way to the flower gardens. Will you help me?"
                Mırmır readily agreed. "Of course! We'll search together."
                The two explored the garden. They smelled the flowers, examined the trees, and finally discovered a magnificent garden filled with fragrant flowers.
                The butterfly was overjoyed. "Thank you, Mırmır. Thanks to your friendship, I found my home."
                Mırmır learned something valuable that day: Helping others always brings good results.
                """

    def _create_friendship_story(self) -> str:
        return """
                # A Tale of Two Friends
                Deep in the forest, Honey Bear and Crispy Rabbit were best friends. They played together every day and had adventures.
                One day, Crispy was very upset. "I have to pick carrots to prepare for winter, but it's too hard on my own," he said.
                Honey immediately wanted to help his friend. "Don't worry, Crispy, I'll help you!"
                They worked together. Honey was digging in the ground with her strong claws, and Crispy was picking carrots.
                When the work was over, Crispy was overjoyed. "You're a wonderful friend, Honey. I don't know how to thank you."
                "Friends help each other," Honey said with a smile. "You would have done the same for me."
                That day, the two friends realized that true friendship was about sharing and helping each other.
                """
    
    def _create_animal_story(self) -> str:
        return """
                # The Adventure of Little Bird Cik
                There lived a little bird named Cik. Cik was just learning to fly and was sometimes afraid.
                One day, his mother said, "Cik, it's time for him to fly on his own."
                Cik climbed up to the edge of the branch, trembling. The ground seemed so high down. "I can't do it, Mom, I'm so scared!"
                Then came the old owl Huu. "Little bird, you don't have to be afraid. Fly with the courage within you."
                Huu taught Cik to breathe. "Now close your eyes and spread your wings."
                Cik mustered his courage and spread his wings. Suddenly, he found himself in the air! He was flying!
                "I did it! I can fly!" he cried with joy.
                The mother bird and Huu the owl were so proud. That day, thanks to his courage, Cik had learned to overcome his fears.
                """
    
    def _create_fantasy_story(self) -> str:
        return """
                # The Secret of the Enchanted Garden
                Ela was a little girl living with her grandmother. There was a very special tree in their garden.
                One day, she found a shiny stone under the tree. The moment she touched the stone, the flowers began to speak!
                "Hello, Ela," said the rose. "You can understand our language because your heart is pure."
                The flowers told Ela that the garden was magical. "But our spell is breaking. Help us."
                Ela asked curiously, "How can I help?"
                "Water us with love and speak to us every day. Love is the most powerful magic."
                From that day on, Ela spoke to the flowers every morning, watering them with love. The garden grew more beautiful with each passing day.
                Her grandmother was amazed. "How did this garden become so beautiful?"
                Ela smiled. She had learned the power of love and would always live this secret with her heart.
                """

    def _create_default_story(self) -> str:
        return """
                # The Story of Little Bear Honey
                Deep in the forest lived a sweet bear named Honey. Honey was a very curious and friendly bear.
                One day, Honey came across a little rabbit lost in the forest. The rabbit was very scared and crying.
                "Don't worry, little friend," Honey said calmly. "I'll help you. Together, we'll find your family."
                Bal patiently accompanied the rabbit. He asked all his animal friends in the forest for help.
                Filthy the squirrel looked out from the high trees. Huu the owl searched from the sky. Sip the ant checked the underground passages.
                They finally found the rabbit's family. Everyone was very happy and grateful to Honey.
                "You're such a kind-hearted friend, Honey," said the rabbit's mother.
                That day, Honey realized that helping others made her very happy, too.
                """

    def update_settings(self, **kwargs):
        # Update settings in Runtime
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"LLM setting updated: {key} = {value}")

# Global LLM service instance
llm_service = LLMService()