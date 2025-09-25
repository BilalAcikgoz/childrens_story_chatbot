import time
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.story import Story
from app.models.user import User
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service
from app.utils.prompts import create_story_prompt

logger = logging.getLogger(__name__)

class StoryService:
    
    async def generate_story(
        self,
        db: Session,
        user_session: str,
        prompt: str,
        length: str = "medium",
        genre: str = "adventure",
        characters: List[str] = None,
        age_group: str = "6-10",
        use_rag: bool = True
    ) -> Dict[str, Any]:
        # Main story generator function
        
        start_time = time.time()
        
        try:
            # Find or create user
            user = self._get_or_create_user(db, user_session)
            
            # Create RAG context 
            context = ""
            if use_rag:
                context = await self._get_rag_context(prompt, genre, age_group)
            
            # Create prompt 
            full_prompt = create_story_prompt(
                topic=prompt,
                length=length,
                characters=characters or [],
                age_group=age_group,
                genre=genre,
                context=context
            )
            
            logger.info(f"The story is being produced: {prompt[:50]}...")
            
            # Create story
            story_content = await llm_service.generate_story(
                prompt=full_prompt,
                max_length=self._get_max_tokens(length),
                temperature=0.7
            )
            
            # Create title
            title = self._generate_title(story_content, prompt)
            
            # Save to database
            story = Story(
                user_id=user_session,
                title=title,
                content=story_content,
                prompt=prompt,
                length=length,
                genre=genre,
                characters=characters or [],
                age_group=age_group,
                generation_time=time.time() - start_time
            )
            
            db.add(story)
            db.commit()
            db.refresh(story)
            
            # Add to RAG
            if use_rag and rag_service.collection:
                await rag_service.add_story(
                    content=story_content,
                    title=title,
                    genre=genre,
                    age_group=age_group,
                    length=length,
                    story_id=str(story.id)
                )
            
            logger.info(f"Story created successfully: ID={story.id}, time={story.generation_time:.2f}s")
            
            return {
                "success": True,
                "story": {
                    "id": story.id,
                    "title": title,
                    "content": story_content,
                    "genre": genre,
                    "length": length,
                    "characters": characters or [],
                    "age_group": age_group,
                    "generation_time": story.generation_time,
                    "created_at": story.created_at.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Story generation error: {str(e)}")
            return {
                "success": False,
                "error": "An error occurred while creating the story. Please try again.",
                "details": str(e)
            }
    
    def _get_max_tokens(self, length: str) -> int:
        token_limits = {
            "short": 300,
            "medium": 500,
            "long": 800
        }
        return token_limits.get(length, 500)
    
    async def _get_rag_context(
        self, 
        prompt: str, 
        genre: str, 
        age_group: str
    ) -> str:
        # Take context from RAG system
        try:
            if not rag_service.collection:
                return ""
                
            similar_stories = await rag_service.search_similar_stories(
                query=prompt,
                genre=genre,
                age_group=age_group,
                n_results=2
            )
            return rag_service.create_context_from_stories(similar_stories)
        except Exception as e:
            logger.error(f"RAG context error: {str(e)}")
            return ""
    
    def _generate_title(self, story_content: str, original_prompt: str) -> str:
        try:
            # Try using the first line as a title
            lines = story_content.strip().split('\n')
            first_line = lines[0].strip()
            
            # If the first line starts with #, use it as a header
            if first_line.startswith('#'):
                title = first_line.replace('#', '').strip()
                if 3 < len(title) < 100:
                    return title
            
            # Create simple title from prompt
            words = original_prompt.strip().split()[:4]  # İlk 4 kelime
            title = " ".join(words).title()
            
            # If it's too short, add a "Story"
            if len(title) < 20 and "story" not in title.lower():
                title = f"{title} Story"
            
            return title[:100] if len(title) < 100 else title[:97] + "..."
            
        except Exception as e:
            logger.error(f"Error creating title: {str(e)}")
            return "New Story"
    
    def _get_or_create_user(self, db: Session, session_id: str) -> User:
        try:
            user = db.query(User).filter(User.session_id == session_id).first()
            if not user:
                user = User(session_id=session_id)
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"New user created: {session_id}")
            return user
        except Exception as e:
            logger.error(f"User operation error: {str(e)}")
            raise
    
    def get_user_stories(
        self,
        db: Session,
        user_session: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        try:
            stories = db.query(Story)\
                .filter(Story.user_id == user_session)\
                .order_by(Story.created_at.desc())\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            return [
                {
                    "id": story.id,
                    "title": story.title,
                    "content": story.content,
                    "prompt": story.prompt,
                    "genre": story.genre,
                    "length": story.length,
                    "characters": story.characters or [],
                    "age_group": story.age_group,
                    "created_at": story.created_at.isoformat(),
                    "generation_time": round(story.generation_time, 2)
                }
                for story in stories
            ]
        except Exception as e:
            logger.error(f"Error fetching user stories: {str(e)}")
            return []
    
    def get_story_by_id(self, db: Session, story_id: int) -> Optional[Dict]:
        try:
            story = db.query(Story).filter(Story.id == story_id).first()
            if not story:
                return None
            
            return {
                "id": story.id,
                "title": story.title,
                "content": story.content,
                "prompt": story.prompt,
                "genre": story.genre,
                "length": story.length,
                "characters": story.characters or [],
                "age_group": story.age_group,
                "created_at": story.created_at.isoformat(),
                "generation_time": round(story.generation_time, 2)
            }
        except Exception as e:
            logger.error(f"Error fetching stories: {str(e)}")
            return None
    
    def delete_story(self, db: Session, story_id: int, user_session: str) -> bool:
        try:
            story = db.query(Story)\
                .filter(Story.id == story_id, Story.user_id == user_session)\
                .first()
            
            if story:
                db.delete(story)
                db.commit()
                logger.info(f"Story deleted: ID={story_id}")
                return True
            else:
                logger.warning(f"No stories found to delete: ID={story_id}")
                return False
                
        except Exception as e:
            logger.error(f"Story deletion error: {str(e)}")
            return False
    
    def get_story_stats(self, db: Session, user_session: str) -> Dict:
        try:
            total_stories = db.query(Story).filter(Story.user_id == user_session).count()
            
            # Distribution by species
            genre_stats = db.query(Story.genre, db.func.count(Story.id))\
                .filter(Story.user_id == user_session)\
                .group_by(Story.genre)\
                .all()
            
            # Distribution by length
            length_stats = db.query(Story.length, db.func.count(Story.id))\
                .filter(Story.user_id == user_session)\
                .group_by(Story.length)\
                .all()
            
            return {
                "total_stories": total_stories,
                "genre_distribution": {genre: count for genre, count in genre_stats},
                "length_distribution": {length: count for length, count in length_stats}
            }
        except Exception as e:
            logger.error(f"Error fetching statistics: {str(e)}")
            return {
                "total_stories": 0,
                "genre_distribution": {},
                "length_distribution": {}
            }

# Global service instance
story_service = StoryService()