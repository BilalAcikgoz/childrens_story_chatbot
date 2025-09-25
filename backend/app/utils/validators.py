from typing import List, Optional, Tuple
import re

# Story genres and their descriptions
GENRE_EXAMPLES = {
    "adventure": "Exciting journeys and discoveries with brave characters",
    "fantasy": "Magical worlds with imaginative creatures and enchanted elements",
    "friendship": "Stories about friendship, cooperation and helping others", 
    "educational": "Fun stories that teach values and knowledge in entertaining ways",
    "animal": "Cute stories featuring animal characters and their adventures",
    "family": "Heartwarming stories about family values and loving relationships",
    "nature": "Stories about nature, environment and ecological awareness",
    "science": "Simple scientific concepts and discoveries made accessible for children",
    "mystery": "Age-appropriate mysteries and puzzle-solving adventures",
    "humor": "Funny, lighthearted stories that make children laugh and smile"
}

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    # Validate user story prompt
    if not prompt or not prompt.strip():
        return False, "Story topic cannot be empty"
    
    cleaned_prompt = prompt.strip()
    
    if len(cleaned_prompt) < 10:
        return False, "Story topic must be at least 10 characters long"
    
    if len(cleaned_prompt) > 500:
        return False, "Story topic cannot exceed 500 characters"
    
    # Check for inappropriate content (basic filtering)
    inappropriate_words = [
        'violence', 'blood', 'death', 'kill', 'murder', 'weapon', 'gun', 'knife',
        'hate', 'scary', 'horror', 'nightmare', 'monster', 'devil', 'hell',
        'drug', 'alcohol', 'smoke', 'cigarette', 'adult', 'sex', 'war', 'fight'
    ]
    
    prompt_lower = cleaned_prompt.lower()
    for word in inappropriate_words:
        if word in prompt_lower:
            return False, "Inappropriate content detected. Please choose a child-friendly topic."
    
    # Check for excessive special characters
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\s\.,!?;:\-çğıöşüÇĞIİÖŞÜ]', cleaned_prompt))
    if special_char_count > len(cleaned_prompt) * 0.1:  # More than 10% special chars
        return False, "Too many special characters in the prompt"
    
    return True, ""

def validate_length(length: str) -> bool:
    # Validate story length parameter
    valid_lengths = ["short", "medium", "long"]
    return length in valid_lengths

def validate_age_group(age_group: str) -> bool:
    # Validate age group parameter
    valid_age_groups = ["3-5", "6-10", "11-15"]
    return age_group in valid_age_groups

def validate_genre(genre: str) -> bool:
    # Validate story genre parameter
    return genre in GENRE_EXAMPLES.keys()

def validate_characters(characters: List[str]) -> Tuple[bool, str]:
    # Validate character names list
    if not characters:
        return True, ""  # Empty list is valid
    
    if len(characters) > 5:
        return False, "Maximum 5 characters allowed per story"
    
    for i, char in enumerate(characters):
        if not char or not char.strip():
            return False, f"Character name at position {i+1} cannot be empty"
        
        cleaned_char = char.strip()
        
        if len(cleaned_char) > 30:
            return False, f"Character name '{cleaned_char}' exceeds 30 character limit"
        
        if len(cleaned_char) < 2:
            return False, f"Character name '{cleaned_char}' must be at least 2 characters"
        
        # Only letters, spaces, hyphens, and apostrophes allowed
        if not re.match(r"^[a-zA-ZğüşıöçĞÜŞİÖÇ\s\-']+$", cleaned_char):
            return False, f"Character name '{cleaned_char}' contains invalid characters. Only letters, spaces, hyphens and apostrophes allowed"
        
        # Check for inappropriate names
        inappropriate_names = ['devil', 'satan', 'hell', 'damn', 'hate', 'kill', 'murder']
        if any(bad_word in cleaned_char.lower() for bad_word in inappropriate_names):
            return False, f"Character name '{cleaned_char}' contains inappropriate content"
    
    # Check for duplicate character names
    cleaned_names = [char.strip().lower() for char in characters if char.strip()]
    if len(cleaned_names) != len(set(cleaned_names)):
        return False, "Duplicate character names are not allowed"
    
    return True, ""

def sanitize_input(text: str) -> str:
    # Sanitize and clean user input text
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove HTML tags (basic protection)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove potentially dangerous characters but keep Turkish characters and basic punctuation
    text = re.sub(r'[^\w\s\.,!?;:\-\'\"çğıöşüÇĞIİÖŞÜ]', '', text)
    
    # Limit consecutive punctuation
    text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)  # Max 3 consecutive punctuation
    
    return text.strip()

def validate_story_request(
    prompt: str,
    length: str = "medium",
    genre: str = "adventure", 
    age_group: str = "6-10",
    characters: Optional[List[str]] = None,
    use_rag: bool = True
) -> Tuple[bool, str]:
    # Comprehensive validation for story generation request
    # Validate prompt
    prompt_valid, prompt_error = validate_prompt(prompt)
    if not prompt_valid:
        return False, f"Prompt validation failed: {prompt_error}"
    
    # Validate length
    if not validate_length(length):
        return False, f"Invalid length '{length}'. Must be: short, medium, or long"
    
    # Validate genre
    if not validate_genre(genre):
        available_genres = ", ".join(GENRE_EXAMPLES.keys())
        return False, f"Invalid genre '{genre}'. Available genres: {available_genres}"
    
    # Validate age group
    if not validate_age_group(age_group):
        return False, f"Invalid age group '{age_group}'. Must be: 3-5, 6-10, or 11-15"
    
    # Validate characters
    if characters:
        chars_valid, chars_error = validate_characters(characters)
        if not chars_valid:
            return False, f"Character validation failed: {chars_error}"
    
    return True, ""

def is_safe_content(text: str) -> bool:
    # Check if content is safe for children
    if not text:
        return True
    
    text_lower = text.lower()
    
    # List of unsafe keywords
    unsafe_keywords = [
        # Violence
        'kill', 'murder', 'death', 'die', 'blood', 'violence', 'fight', 'war',
        'weapon', 'gun', 'knife', 'sword', 'bomb', 'explosion', 'attack',
        
        # Fear/Horror
        'scary', 'horror', 'nightmare', 'monster', 'ghost', 'zombie',
        'demon', 'devil', 'satan', 'hell', 'evil', 'haunted',
        
        # Inappropriate
        'drug', 'alcohol', 'drunk', 'smoke', 'cigarette', 'beer', 'wine',
        'sex', 'adult', 'mature', 'nude', 'naked',
        
        # Negative emotions (extreme)
        'hate', 'revenge', 'angry', 'mad', 'furious', 'rage', 'suicide'
    ]
    
    # Check for unsafe keywords
    for keyword in unsafe_keywords:
        if keyword in text_lower:
            return False
    
    return True

def validate_story_output(
    story_content: str, 
    min_length: int = 100, 
    max_length: int = 2000
) -> Tuple[bool, str]:
    # Validate generated story content
    if not story_content or not story_content.strip():
        return False, "Story content is empty"
    
    cleaned_content = story_content.strip()
    
    if len(cleaned_content) < min_length:
        return False, f"Story too short (minimum {min_length} characters required)"
    
    if len(cleaned_content) > max_length:
        return False, f"Story too long (maximum {max_length} characters allowed)"
    
    # Check if content is safe for children
    if not is_safe_content(cleaned_content):
        return False, "Story contains inappropriate content for children"
    
    # Check for minimum story structure (should have some sentences)
    sentences = re.split(r'[.!?]+', cleaned_content)
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    if len(valid_sentences) < 3:
        return False, "Story should contain at least 3 meaningful sentences"
    
    # Check for excessive repetition
    words = cleaned_content.lower().split()
    if len(words) > 0:
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        if repetition_ratio > 0.5:  # More than 50% repetition
            return False, "Story contains too much repetition"
    
    return True, ""

def clean_story_title(title: str) -> str:
    # Clean and format story title
    if not title:
        return "Untitled Story"
    
    # Remove markdown formatting
    title = re.sub(r'#+\s*', '', title)
    title = re.sub(r'\*+', '', title)
    
    # Clean and capitalize
    title = sanitize_input(title)
    title = title.title()
    
    # Ensure reasonable length
    if len(title) > 100:
        title = title[:97] + "..."
    elif len(title) < 3:
        title = "New Story"
    
    return title.strip()

def get_available_options() -> dict:
    # Get all available story generation options
    return {
        "lengths": ["short", "medium", "long"],
        "age_groups": ["3-5", "6-10", "11-15"],
        "genres": list(GENRE_EXAMPLES.keys()),
        "genre_descriptions": GENRE_EXAMPLES
    }

def validate_model_output(output: str) -> Tuple[bool, str, str]:
    # Validate and extract story from model output
    if not output or not output.strip():
        return False, "", ""
    
    lines = output.strip().split('\n')
    title = ""
    story_lines = []
    
    # Try to extract title and story
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if this looks like a title (starts with # or is the first significant line)
        if line.startswith('#') or (not title and len(line) < 100 and i < 3):
            if not title:  # Take the first title-like line
                title = clean_story_title(line)
        else:
            story_lines.append(line)
    
    # Join story lines
    story_content = ' '.join(story_lines)
    
    # Validate the extracted content
    is_valid, error = validate_story_output(story_content)
    
    if not title:
        title = "New Story"
    
    return is_valid, story_content, title

def validate_fine_tune_data(data: dict) -> Tuple[bool, str]:
    # Validate fine-tuning data format
    required_fields = ['prompt', 'completion']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        if not data[field] or not data[field].strip():
            return False, f"Field '{field}' cannot be empty"
    
    # Validate prompt
    prompt_valid, prompt_error = validate_prompt(data['prompt'])
    if not prompt_valid:
        return False, f"Invalid prompt: {prompt_error}"
    
    # Validate completion (story)
    completion_valid, completion_error = validate_story_output(data['completion'])
    if not completion_valid:
        return False, f"Invalid completion: {completion_error}"
    
    return True, ""

def validate_rag_query(query: str, max_results: int = 10) -> Tuple[bool, str]:
    # Validate RAG query parameters
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    if len(query.strip()) < 3:
        return False, "Query must be at least 3 characters long"
    
    if len(query.strip()) > 1000:
        return False, "Query cannot exceed 1000 characters"
    
    if not (1 <= max_results <= 50):
        return False, "Max results must be between 1 and 50"
    
    return True, ""

def validate_api_response(response_data: dict) -> bool:
    # Validate API response structure
    required_fields = ['success']
    
    for field in required_fields:
        if field not in response_data:
            return False
    
    if response_data['success']:
        # Success response should have story data
        return 'story' in response_data and isinstance(response_data['story'], dict)
    else:
        # Error response should have error message
        return 'error' in response_data and isinstance(response_data['error'], str)