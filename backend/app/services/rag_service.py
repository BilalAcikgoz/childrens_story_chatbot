import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from app.config.settings import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.model_name = settings.embedding_model_name
        self.csv_file_path = "data/story.csv"
        self.batch_size = 1000  # Batch processing for large files
        
    async def initialize(self):
        # Inıtialize RAG system
        try:
            logger.info("Initialize to RAG Service...")
            
            # Create chromaDB client
            chroma_path = settings.chroma_persist_directory
            os.makedirs(chroma_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=chroma_path)
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Create or connect Collection
            try:
                self.collection = self.client.get_collection(
                    name=settings.chroma_collection_name
                )
                logger.info(f"Available collection was found: {self.collection.count()} story")
            except:
                self.collection = self.client.create_collection(
                    name=settings.chroma_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("New collection was created")
                await self._load_sample_stories()
            
            logger.info("RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"RAG service initialization error: {str(e)}")
            # Can work without RAG
            self.client = None
            self.collection = None
    
    async def _load_stories_from_csv(self):
        try:
            csv_path = Path(self.csv_file_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {self.csv_file_path}")
                return
            
            logger.info(f"CSV file is loading: {self.csv_file_path}")
            
            # Read CSV file in chunks
            chunk_size = self.batch_size
            total_processed = 0
            
            # Firstly check the file dimension
            df_sample = pd.read_csv(csv_path, nrows=5)
            logger.info(f"CSV columns: {list(df_sample.columns)}")
            
            if 'text' not in df_sample.columns:
                logger.error("Column 'text' not found in CSV file!")
                return
            
            # Process in chunks
            for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
                try:
                    await self._process_chunk(chunk_df, total_processed)
                    total_processed += len(chunk_df)
                    
                    if total_processed % (chunk_size * 10) == 0:  # Log in every 10 batches
                        logger.info(f"Number of stories processed: {total_processed}")
                        
                    # Set limits on large files for memory management 
                    if total_processed >= 50000:  # İlk 50K hikaye
                        logger.info("The first 50,000 stories have been uploaded, more are coming...")
                        break
                        
                except Exception as e:
                    logger.error(f"Chunk process error (row {total_processed}): {str(e)}")
                    continue
            
            logger.info(f"Sum {total_processed} story added to ChromaDB")
            
        except Exception as e:
            logger.error(f"CSV load error: {str(e)}")
    
    async def _process_chunk(self, chunk_df: pd.DataFrame, start_index: int):
        # Process chunk and add to ChromaDB
        try:
            # Clean Nan values
            chunk_df = chunk_df.dropna(subset=['text'])
            
            # Filter too short texts
            chunk_df = chunk_df[chunk_df['text'].str.len() >= 50]
            
            if len(chunk_df) == 0:
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in chunk_df.iterrows():
                text = str(row['text']).strip()
                
                # Check text length
                if len(text) < 50 or len(text) > 2000:
                    continue
                
                # Clean text
                cleaned_text = self._clean_text(text)
                if not cleaned_text:
                    continue
                
                # Predict story features
                estimated_metadata = self._estimate_story_features(cleaned_text)
                
                documents.append(cleaned_text)
                metadatas.append(estimated_metadata)
                ids.append(f"story_csv_{start_index + len(documents)}")
            
            # Add to ChromaDB as a batch
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.debug(f"Chunk added: {len(documents)} story")
                
        except Exception as e:
            logger.error(f"Chunk processing error: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        # Clean text
        try:
            # Clean excessive whitespace
            text = ' '.join(text.split())
            
            # Clean HTML tags
            import re
            text = re.sub(r'<[^>]+>', '', text)
            
            # Clean special characters except basic punctuation
            text = re.sub(r'[^\w\s\.,!?;:\-çğıöşüÇĞIİÖŞÜ]', '', text)
            
            return text.strip()
        except:
            return ""
    
    def _estimate_story_features(self, text: str) -> Dict[str, str]:
        # Guess the features from the story text
        text_lower = text.lower()
        
        # Predict length
        word_count = len(text.split())
        if word_count < 200:
            length = "short"
        elif word_count < 500:
            length = "medium"
        else:
            length = "long"
        
        # Age group estimate
        simple_words = ['mother', 'father', 'home', 'game', 'friend', 'love']
        complex_words = ['struggle', 'success', 'difficulty', 'target', 'future']
        
        simple_count = sum(1 for word in simple_words if word in text_lower)
        complex_count = sum(1 for word in complex_words if word in text_lower)
        
        if simple_count > complex_count:
            age_group = "3-8"
        elif complex_count > simple_count:
            age_group = "11-15"
        else:
            age_group = "6-10"
        
        # Type prediction (keywords)
        genre_keywords = {
            'adventure': ['journey', 'discovery', 'courage', 'adventure', 'danger'],
            'friendship': ['friend', 'allies', 'help', 'unity', 'share'],
            'fantasy': ['magic', 'magic', 'dragon', 'fairy', 'magical'],
            'animal': ['cat', 'dog', 'bird', 'forest', 'animal'],
            'family': ['mother', 'father', 'sibling', 'family', 'home'],
            'educator': ['learn', 'lesson', 'knowledge', 'value', 'advice']
        }
        
        genre_scores = {}
        for genre, keywords in genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            genre_scores[genre] = score
        
        # Choose the genre with the highest score
        genre = max(genre_scores, key=genre_scores.get) if any(genre_scores.values()) else "genel"
        
        return {
            "genre": genre,
            "length": length,
            "age_group": age_group,
            "source": "csv_dataset"
        }
    
    async def search_similar_stories(
        self,
        query: str,
        genre: Optional[str] = None,
        age_group: Optional[str] = None,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                return []
            
            # Create where conditions
            where_conditions = {}
            if genre:
                where_conditions["genre"] = genre
            if age_group:
                where_conditions["age_group"] = age_group
            
            # Search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 10),  # Max 10 results
                where=where_conditions if where_conditions else None
            )
            
            # Format results
            similar_stories = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    
                    similar_stories.append({
                        "content": doc[:300] + "..." if len(doc) > 300 else doc,  # Preview
                        "full_content": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance
                    })
            
            # Sort by Similarity
            similar_stories.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_stories
            
        except Exception as e:
            logger.error(f"Search similar story error: {str(e)}")
            return []
    
    async def add_story(
        self,
        content: str,
        title: str,
        genre: str,
        age_group: str,
        length: str,
        story_id: str
    ):
        # Add user story to RAG database
        try:
            if not self.collection:
                return
            
            metadata = {
                "title": title,
                "genre": genre,
                "age_group": age_group,
                "length": length,
                "source": "user_generated"
            }
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"user_story_{story_id}"]
            )
            
            logger.info(f"User story added to RAG: {story_id}")
            
        except Exception as e:
            logger.error(f"Error adding story RAG: {str(e)}")
    
    def create_context_from_stories(self, similar_stories: List[Dict]) -> str:
        # Create context text from similar stories
        if not similar_stories:
            return ""
        
        context_parts = []
        for story in similar_stories[:2]:  # Take top 2 stories
            metadata = story["metadata"]
            content = story["content"]
            
            context_parts.append(
                f"Similar story example ({metadata.get('genre', 'general')} - {metadata.get('age_group', 'general')}): "
                f"{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def get_collection_stats(self) -> Dict:
        try:
            if not self.collection:
                return {"status": "Collection is none"}
            
            count = self.collection.count()
            return {
                "total_stories": count,
                "status": "Active",
                "collection_name": settings.chroma_collection_name
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"status": "Hata", "error": str(e)}

# Global RAG service instance
rag_service = RAGService()