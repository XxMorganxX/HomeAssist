"""
Main orchestrator for managing provider lifecycle and pipeline execution.
"""

import asyncio
import random
import subprocess
from typing import Dict, Any, Optional, AsyncIterator
from pathlib import Path

try:
    # Try relative imports first (when used as package)
    from .interfaces import (
        TranscriptionInterface,
        ResponseInterface,
        TextToSpeechInterface,
        ContextInterface,
        WakeWordInterface
    )
    from .factory import ProviderFactory
    from .models.data_models import TranscriptionResult, ResponseChunk, AudioOutput
    pass
except ImportError:
    # Fall back to absolute imports (when run as module)
    from assistant_framework.interfaces import (
        TranscriptionInterface,
        ResponseInterface,
        TextToSpeechInterface,
        ContextInterface,
        WakeWordInterface
    )
    from assistant_framework.factory import ProviderFactory
    from assistant_framework.models.data_models import TranscriptionResult, ResponseChunk, AudioOutput
    pass


class AssistantOrchestrator:
    """Main orchestrator class that manages providers and pipeline execution."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config: Configuration dictionary containing provider settings
        """
        self.config = config
        self.providers = {}
        self.is_initialized = False
        
        # Provider instances
        self.transcription: Optional[TranscriptionInterface] = None
        self.response: Optional[ResponseInterface] = None
        self.tts: Optional[TextToSpeechInterface] = None
        self.context: Optional[ContextInterface] = None
        self.wakeword: Optional[WakeWordInterface] = None
    
    async def initialize(self) -> bool:
        """
        Initialize all providers based on configuration.
        
        Returns:
            bool: True if all providers initialized successfully
        """
        try:
            # Create all providers using factory
            self.providers = ProviderFactory.create_all_providers(self.config)
            
            # Assign provider instances
            self.transcription = self.providers.get('transcription')
            self.response = self.providers.get('response')
            self.tts = self.providers.get('tts')
            self.context = self.providers.get('context')
            self.wakeword = self.providers.get('wakeword')
            
            # Initialize each provider
            initialization_tasks = []
            
            if self.transcription:
                initialization_tasks.append(
                    ('transcription', self.transcription.initialize())
                )
            
            if self.response:
                initialization_tasks.append(
                    ('response', self.response.initialize())
                )
            
            if self.tts:
                initialization_tasks.append(
                    ('tts', self.tts.initialize())
                )
            
            if self.wakeword:
                initialization_tasks.append(
                    ('wakeword', self.wakeword.initialize())
                )
            
            # Run initializations concurrently
            results = await asyncio.gather(
                *[task[1] for task in initialization_tasks],
                return_exceptions=True
            )
            
            # Check results
            failed_providers = []
            for i, (provider_name, _) in enumerate(initialization_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    print(f"❌ {provider_name} provider initialization failed: {result}")
                    failed_providers.append(provider_name)
                elif not result:
                    print(f"❌ {provider_name} provider initialization returned False")
                    failed_providers.append(provider_name)
                else:
                    print(f"✅ {provider_name} provider initialized successfully")
            
            if failed_providers:
                print(f"❌ Failed to initialize providers: {failed_providers}")
                await self.cleanup()
                return False
            
            self.is_initialized = True
            print("🚀 Assistant orchestrator initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            await self.cleanup()
            return False
    
    async def run_transcription_only(self) -> AsyncIterator[TranscriptionResult]:
        """
        Run transcription component only.
        
        Yields:
            TranscriptionResult: Transcription results as they become available
        """
        if not self.transcription:
            raise RuntimeError("Transcription provider not available")
        
        async for result in self.transcription.start_streaming():
            yield result
    
    async def run_wakeword_only(self) -> AsyncIterator[dict]:
        """
        Run wake word detection only.
        
        Yields:
            dict: Wake word event dictionaries
        """
        if not self.wakeword:
            raise RuntimeError("Wake word provider not available")

        async for event in self.wakeword.start_detection():
            yield {
                'model_name': event.model_name,
                'score': event.score,
                'timestamp': event.timestamp,
                'metadata': event.metadata,
            }
    
    async def run_response_only(self, 
                               message: str, 
                               use_context: bool = True) -> AsyncIterator[ResponseChunk]:
        """
        Run response component only.
        
        Args:
            message: Input message to respond to
            use_context: Whether to use conversation context
            
        Yields:
            ResponseChunk: Response chunks as they become available
        """
        if not self.response:
            raise RuntimeError("Response provider not available")
        
        # Get context if available and requested
        context_history = None
        tool_context = None
        if use_context and self.context:
            # Use a compact, recent-biased window for the responder
            get_recent_for_response = getattr(self.context, 'get_recent_for_response', None)
            if callable(get_recent_for_response):
                context_history = get_recent_for_response()
            else:
                # Fallback: recent slice of full history
                get_full = getattr(self.context, 'get_full_history', None)
                full_history = get_full() if callable(get_full) else self.context.get_history()
                context_history = full_history[-8:]
            # Short window for tool selection
            get_tool_ctx = getattr(self.context, 'get_tool_context', None)
            tool_context = get_tool_ctx() if callable(get_tool_ctx) else None
        
        async for chunk in self.response.stream_response(message, context_history, tool_context):
            yield chunk
    
    async def run_tts_only(self, 
                          text: str,
                          voice: Optional[str] = None,
                          speed: Optional[float] = None,
                          pitch: Optional[float] = None,
                          save_path: Optional[Path] = None,
                          play_audio: bool = True) -> AudioOutput:
        """
        Run TTS component only.
        
        Args:
            text: Text to synthesize
            voice: Optional voice override
            speed: Optional speed override  
            pitch: Optional pitch override
            save_path: Optional path to save audio file
            play_audio: Whether to play the audio
            
        Returns:
            AudioOutput: Synthesized audio
        """
        if not self.tts:
            raise RuntimeError("TTS provider not available")
        
        # Synthesize audio
        audio = await self.tts.synthesize(text, voice, speed, pitch)
        
        # Save if requested
        if save_path:
            await self.tts.save_audio(audio, save_path)
        
        # Play if requested
        if play_audio:
            self.tts.play_audio(audio)
        
        return audio
    
    async def run_wakeword_then_pipeline(self,
                                         greeting_dir: Optional[Path] = None,
                                         auto_save_audio: bool = True,
                                         speech_audio_dir: Optional[Path] = None,
                                         play_responses: bool = True) -> None:
        """
        Wait for a wake word, play a greeting (optional), then run the full pipeline.
        """
        if not self.wakeword:
            raise RuntimeError("Wake word provider not available")
        if not all([self.transcription, self.response, self.tts]):
            missing = []
            if not self.transcription:
                missing.append('transcription')
            if not self.response:
                missing.append('response')
            if not self.tts:
                missing.append('tts')
            raise RuntimeError(f"Missing required providers: {missing}")

        print("👂 Listening for wake word...")
        # Wait for first event
        async for _ in self.run_wakeword_only():
            print("🔔 Wakeword detected!")
            break

        # Stop wakeword detection to release microphone
        try:
            await self.wakeword.stop_detection()
        except Exception:
            pass

        # Optional greeting
        if greeting_dir:
            try:
                files = []
                if greeting_dir.exists():
                    files.extend(list(greeting_dir.glob("*.mov")))
                    files.extend(list(greeting_dir.glob("*.mp3")))
                if files:
                    chosen = random.choice(files)
                    print(f"🎵 Playing greeting: {chosen.name}")
                    # On macOS, afplay blocks until completion
                    subprocess.run(["afplay", str(chosen)], check=False)
                else:
                    print("⚠️  No greeting audio files found")
            except Exception as e:
                print(f"⚠️  Failed to play greeting: {e}")

        # Start main voice pipeline
        await self.run_full_pipeline(
            auto_save_audio=auto_save_audio,
            speech_audio_dir=speech_audio_dir,
            play_responses=play_responses,
        )

    async def run_full_pipeline(self, 
                               auto_save_audio: bool = True,
                               speech_audio_dir: Optional[Path] = None,
                               play_responses: bool = True) -> None:
        """
        Run the full pipeline: transcription → response → TTS.
        
        Args:
            auto_save_audio: Whether to automatically save TTS audio
            speech_audio_dir: Directory to save audio files
            play_responses: Whether to play TTS responses
        """
        if not all([self.transcription, self.response, self.tts]):
            missing = []
            if not self.transcription:
                missing.append('transcription')
            if not self.response:
                missing.append('response')
            if not self.tts:
                missing.append('tts')
            raise RuntimeError(f"Missing required providers: {missing}")
        
        print("🎤 Starting full pipeline - speak to begin...")
        
        try:
            # Start transcription stream
            async for transcription_result in self.transcription.start_streaming():
                # Only process final transcriptions
                if not transcription_result.is_final:
                    print(f"[PARTIAL] {transcription_result.text}")
                    continue
                
                print(f"🎙️  [FINAL] {transcription_result.text}")
                
                # Add user message to context
                if self.context:
                    self.context.add_message("user", transcription_result.text)
                
                # Get response
                print("🤔 Generating response...")
                full_response = ""
                
                async for response_chunk in self.run_response_only(
                    transcription_result.text, 
                    use_context=True
                ):
                    # Accumulate response text
                    full_response += response_chunk.content
                    
                    # Print streaming response
                    if response_chunk.content:
                        print(response_chunk.content, end="", flush=True)
                    
                    # Handle completion
                    if response_chunk.is_complete:
                        print()  # New line
                        break
                
                # Add assistant response to context
                if self.context:
                    self.context.add_message("assistant", full_response)
                    # Auto-trim if needed
                    self.context.auto_trim_if_needed()
                
                # Generate TTS
                if full_response.strip():
                    print("🔊 Generating speech...")
                    
                    # Determine save path if auto-saving
                    save_path = None
                    if auto_save_audio:
                        if not speech_audio_dir:
                            speech_audio_dir = Path("speech_audio")
                        speech_audio_dir.mkdir(exist_ok=True)
                        
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        save_path = speech_audio_dir / f"response_{timestamp}.mp3"
                    
                    # Synthesize and play
                    await self.run_tts_only(
                        full_response,
                        save_path=save_path,
                        play_audio=play_responses
                    )
                
                print("=" * 50)
                print("🎤 Ready for next input...")
                
        except KeyboardInterrupt:
            print("\n🛑 Pipeline stopped by user")
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
        finally:
            await self.stop_transcription()
    
    async def process_single_message(self, 
                                   message: str,
                                   use_context: bool = True,
                                   generate_tts: bool = True,
                                   play_audio: bool = True,
                                   save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process a single message through the pipeline.
        
        Args:
            message: Input message
            use_context: Whether to use conversation context
            generate_tts: Whether to generate TTS
            play_audio: Whether to play audio
            save_path: Optional path to save audio
            
        Returns:
            Dictionary containing response and audio (if generated)
        """
        result = {'message': message, 'response': '', 'audio': None}
        
        # Add user message to context
        if use_context and self.context:
            self.context.add_message("user", message)
        
        # Get response
        full_response = ""
        async for response_chunk in self.run_response_only(message, use_context):
            full_response += response_chunk.content
            if response_chunk.is_complete:
                break
        
        result['response'] = full_response
        
        # Add assistant response to context
        if use_context and self.context:
            self.context.add_message("assistant", full_response)
            self.context.auto_trim_if_needed()
        
        # Generate TTS if requested
        if generate_tts and full_response.strip():
            audio = await self.run_tts_only(
                full_response,
                save_path=save_path,
                play_audio=play_audio
            )
            result['audio'] = audio
        
        return result
    
    async def stop_transcription(self) -> None:
        """Stop transcription if active."""
        if self.transcription and self.transcription.is_active:
            await self.transcription.stop_streaming()
    
    async def cleanup(self) -> None:
        """Clean up all provider resources."""
        cleanup_tasks = []
        
        if self.transcription:
            cleanup_tasks.append(self.transcription.cleanup())
        
        if self.response:
            cleanup_tasks.append(self.response.cleanup())
        
        if self.tts:
            cleanup_tasks.append(self.tts.cleanup())
        
        if hasattr(self, 'wakeword') and self.wakeword:
            cleanup_tasks.append(self.wakeword.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.is_initialized = False
        print("🧹 Orchestrator cleanup complete")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the orchestrator and providers.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            'initialized': self.is_initialized,
            'providers': {}
        }
        
        # Check each provider
        for provider_type, provider in self.providers.items():
            if provider:
                provider_status = {
                    'available': True,
                    'type': provider.__class__.__name__,
                    'capabilities': provider.capabilities if hasattr(provider, 'capabilities') else {}
                }
                
                # Add specific status for transcription
                if hasattr(provider, 'is_active'):
                    provider_status['active'] = provider.is_active
                
                status['providers'][provider_type] = provider_status
            else:
                status['providers'][provider_type] = {'available': False}
        
        # Add context summary if available
        if self.context:
            status['context'] = self.context.get_summary()
        
        return status
    
    def get_available_providers(self) -> Dict[str, list]:
        """Get list of all available providers."""
        return ProviderFactory.get_available_providers()


async def create_orchestrator(config: Dict[str, Any]) -> AssistantOrchestrator:
    """
    Convenience function to create and initialize an orchestrator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized AssistantOrchestrator instance
    """
    orchestrator = AssistantOrchestrator(config)
    
    if not await orchestrator.initialize():
        raise RuntimeError("Failed to initialize orchestrator")
    
    return orchestrator