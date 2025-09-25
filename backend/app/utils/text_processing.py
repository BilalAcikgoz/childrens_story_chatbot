import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter
import unicodedata

class TextProcessor:
    # Comprehensive text processing utilities for story generation
    
    def __init__(self):
        self.turkish_chars = "çğıöşüÇĞIİÖŞÜ"
        self.punctuation = string.punctuation + "…""''"
        
    def clean_text(self, text: str) -> str:
        # Clean and normalize text while preserving Turkish characters
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean excessive punctuation
        text = re.sub(r'([.!?]){4,}', r'\1\1\1', text)  # Max 3 consecutive punctuation
        text = re.sub(r'([,;:]){2,}', r'\1', text)  # Max 1 consecutive comma/semicolon
        
        # Remove excessive dashes
        text = re.sub(r'[-]{3,}', '--', text)
        
        # Clean quotation marks
        text = re.sub(r'["""''`]+', '"', text)
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        # Normalize text for consistency
        if not text:
            return ""
        
        # Normalize Unicode characters (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Fix common Turkish character issues
        replacements = {
            'i̇': 'i',  # Turkish dotted i
            'I': 'İ',   # Uppercase i should be İ in Turkish
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def count_words(self, text: str) -> int:
        # Count words in text
        if not text:
            return 0
        
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s' + self.turkish_chars + ']', ' ', text)
        words = cleaned.split()
        
        return len(words)
    
    def count_sentences(self, text: str) -> int:
        # Count sentences in text
        if not text:
            return 0
        
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings and very short fragments
        valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        return len(valid_sentences)
    
    def calculate_reading_time(self, text: str, wpm: int = 200) -> int:
        # Calculate estimated reading time in minutes
        word_count = self.count_words(text)
        return max(1, round(word_count / wpm))
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        # Extract important keywords from text
        if not text:
            return []
        
        # Clean text and convert to lowercase
        cleaned = re.sub(r'[^\w\s' + self.turkish_chars + ']', ' ', text.lower())
        words = cleaned.split()
        
        # Common Turkish stop words
        stop_words = {
            'bir', 'bu', 'da', 'de', 've', 'ki', 'mi', 'mu', 'mü', 'ile', 'için',
            'olan', 'oldu', 'olur', 'var', 'yok', 've', 'veya', 'ama', 'fakat',
            'çok', 'daha', 'en', 'gibi', 'kadar', 'sonra', 'önce', 'şimdi',
            'hiç', 'her', 'bazı', 'o', 'bu', 'şu', 'ben', 'sen', 'biz', 'siz'
        }
        
        # Filter out stop words and short words
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Return most common words
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        # Calculate similarity between two texts using word overlap
        if not text1 or not text2:
            return 0.0
        
        # Get keywords from both texts
        keywords1 = set(self.extract_keywords(text1, 20))
        keywords2 = set(self.extract_keywords(text2, 20))
        
        # Calculate Jaccard similarity
        if not keywords1 and not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def format_story(self, story: str, max_line_length: int = 80) -> str:
        # Format story for better readability
        if not story:
            return ""
        
        # Clean the story first
        story = self.clean_text(story)
        
        # Split into sentences
        sentences = re.split(r'([.!?]+)', story)
        formatted_sentences = []
        
        current_line = ""
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed line length
            if len(current_line + " " + sentence) > max_line_length and current_line:
                formatted_sentences.append(current_line.strip())
                current_line = sentence
            else:
                current_line += " " + sentence if current_line else sentence
        
        # Add remaining text
        if current_line:
            formatted_sentences.append(current_line.strip())
        
        return "\n\n".join(formatted_sentences)
    
    def extract_title_from_story(self, story: str) -> str:
        # Extract or generate title from story content
        if not story:
            return "Untitled Story"
        
        lines = story.split('\n')
        
        # Look for existing title patterns
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if not line:
                continue
            
            # Check for markdown title
            if line.startswith('#'):
                title = re.sub(r'#+\s*', '', line).strip()
                if 5 <= len(title) <= 100:
                    return title
            
            # Check for title-like line (short, capitalized)
            if len(line) <= 100 and line[0].isupper() and not line.endswith('.'):
                # Check if it looks like a title (mostly uppercase words)
                words = line.split()
                if len(words) <= 10 and sum(1 for w in words if w[0].isupper()) >= len(words) * 0.7:
                    return line
        
        # Generate title from first sentence
        first_sentence = self.get_first_sentence(story)
        if first_sentence:
            # Take first few words
            words = first_sentence.split()[:6]  # Max 6 words
            title = " ".join(words)
            if len(title) > 50:
                title = title[:47] + "..."
            return title.title()
        
        return "Untitled Story"
    
    def get_first_sentence(self, text: str) -> str:
        # Extract first sentence from text
        if not text:
            return ""
        
        # Clean text first
        text = self.clean_text(text)
        
        # Find first sentence
        match = re.search(r'^[^.!?]*[.!?]', text)
        if match:
            return match.group().strip()
        
        # If no sentence ending found, take first 100 characters
        if len(text) > 100:
            return text[:97] + "..."
        
        return text
    
    def detect_repetition(self, text: str) -> Dict[str, float]:
        # Detect various types of repetition in text
        if not text:
            return {"word_repetition": 0.0, "phrase_repetition": 0.0}
        
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return {"word_repetition": 0.0, "phrase_repetition": 0.0}
        
        # Word repetition
        unique_words = len(set(words))
        word_repetition = 1 - (unique_words / word_count)
        
        # Phrase repetition (2-3 word phrases)
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(" ".join(words[i:i+2]))
        
        phrase_repetition = 0.0
        if phrases:
            unique_phrases = len(set(phrases))
            phrase_repetition = 1 - (unique_phrases / len(phrases))
        
        return {
            "word_repetition": round(word_repetition, 3),
            "phrase_repetition": round(phrase_repetition, 3)
        }
    
    def validate_text_quality(self, text: str) -> Dict[str, any]:
        # Comprehensive text quality analysis
        if not text:
            return {
                "is_valid": False,
                "issues": ["Text is empty"],
                "metrics": {}
            }
        
        issues = []
        metrics = {
            "word_count": self.count_words(text),
            "sentence_count": self.count_sentences(text),
            "avg_sentence_length": 0,
            "reading_time": self.calculate_reading_time(text)
        }
        
        # Calculate average sentence length
        if metrics["sentence_count"] > 0:
            metrics["avg_sentence_length"] = metrics["word_count"] / metrics["sentence_count"]
        
        # Check repetition
        repetition_metrics = self.detect_repetition(text)
        metrics.update(repetition_metrics)
        
        # Quality checks
        if metrics["word_count"] < 50:
            issues.append("Text too short")
        
        if metrics["sentence_count"] < 3:
            issues.append("Too few sentences")
        
        if repetition_metrics["word_repetition"] > 0.5:
            issues.append("High word repetition")
        
        if repetition_metrics["phrase_repetition"] > 0.3:
            issues.append("High phrase repetition")
        
        if metrics["avg_sentence_length"] > 30:
            issues.append("Sentences too long")
        elif metrics["avg_sentence_length"] < 5:
            issues.append("Sentences too short")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "metrics": metrics
        }
    
    def prepare_for_embedding(self, text: str) -> str:
        # Prepare text for embedding generation (RAG)
        if not text:
            return ""
        
        # Clean and normalize
        text = self.clean_text(text)
        text = self.normalize_text(text)
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s' + self.turkish_chars + '.!?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        # Split text into overlapping chunks for processing
        if not text or chunk_size <= 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence ending within last 100 characters
                last_part = text[end-100:end+100]
                sentence_end = -1
                
                for punct in ['. ', '! ', '? ']:
                    pos = last_part.find(punct, 50)  # Start looking from middle
                    if pos != -1:
                        sentence_end = pos + 1
                        break
                
                if sentence_end != -1:
                    end = end - 100 + sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        return chunks

# Global text processor instance
text_processor = TextProcessor()

# Convenience functions
def clean_text(text: str) -> str:
    return text_processor.clean_text(text)

def count_words(text: str) -> int:
    return text_processor.count_words(text)

def count_sentences(text: str) -> int:
    return text_processor.count_sentences(text)

def format_story(story: str) -> str:
    return text_processor.format_story(story)

def extract_title(story: str) -> str:
    return text_processor.extract_title_from_story(story)

def validate_quality(text: str) -> Dict[str, any]:
    return text_processor.validate_text_quality(text)