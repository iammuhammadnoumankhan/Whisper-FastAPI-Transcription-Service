from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated
import os
import uuid
import logging
import uvicorn
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
import tempfile
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi.staticfiles import StaticFiles


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("whisper-api")

# Create temp directory for uploads
UPLOAD_DIR = Path(tempfile.gettempdir()) / "whisper_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="High-performance speech-to-text API using faster-whisper",
    version="1.0.0",
)



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


# Store transcription tasks
transcription_tasks = {}

# Model cache to avoid reloading
model_cache = {}

class TranscriptionOptions(BaseModel):
    model_size: str = Field(
        "large-v3", 
        description="Model size to use for transcription"
    )
    device: str = Field(
        "cuda", 
        description="Device to use for computation (cuda, cpu)"
    )
    compute_type: str = Field(
        "float16", 
        description="Compute type for model (float16, int8_float16, int8)"
    )
    language: Optional[str] = Field(
        None, 
        description="Language code for transcription (e.g., 'en', 'fr')"
    )
    batch_size: Optional[int] = Field(
        16, 
        description="Batch size for transcription when using batched mode"
    )
    beam_size: int = Field(
        5, 
        description="Beam size for transcription"
    )
    word_timestamps: bool = Field(
        False, 
        description="Whether to include timestamps for each word"
    )
    vad_filter: bool = Field(
        True, 
        description="Whether to apply voice activity detection"
    )
    vad_parameters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Parameters for VAD filtering"
    )
    condition_on_previous_text: bool = Field(
        True, 
        description="Whether to condition on previous text"
    )
    use_batched_mode: bool = Field(
        True, 
        description="Whether to use batched inference for faster processing"
    )

class TranscriptionTask(BaseModel):
    id: str
    status: str
    created_at: str
    file_name: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Dict[str, Any]]] = None

class TranscriptionResult(BaseModel):
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float

def get_or_load_model(options: TranscriptionOptions):
    """Get or load a model based on transcription options"""
    key = f"{options.model_size}_{options.device}_{options.compute_type}"
    
    if key not in model_cache:
        logger.info(f"Loading model: {options.model_size} on {options.device} with {options.compute_type}")
        model = WhisperModel(
            options.model_size,
            device=options.device,
            compute_type=options.compute_type,
            download_root=os.environ.get("MODEL_DIR", None)
        )
        model_cache[key] = model
    
    return model_cache[key]

async def process_transcription(task_id: str, file_path: str, options: TranscriptionOptions):
    """Process transcription in background"""
    try:
        transcription_tasks[task_id]["status"] = "processing"
        
        # Get or load the model
        model = get_or_load_model(options)
        
        # Create batched pipeline if requested
        if options.use_batched_mode:
            pipeline = BatchedInferencePipeline(
                model=model,
            )
        else:
            pipeline = model
        
        # Prepare transcription kwargs
        kwargs = {
            "beam_size": options.beam_size,
            "word_timestamps": options.word_timestamps,
            "vad_filter": options.vad_filter,
            "condition_on_previous_text": options.condition_on_previous_text,
        }
        
        # Add language if specified
        if options.language:
            kwargs["language"] = options.language
            
        # Add VAD parameters if specified
        if options.vad_parameters:
            kwargs["vad_parameters"] = options.vad_parameters
            
        # Add batch size if using batched mode
        if options.use_batched_mode:
            kwargs["batch_size"] = options.batch_size
        
        # Run transcription
        start_time = datetime.now()
        segments, info = pipeline.transcribe(file_path, **kwargs)
        
        # Convert generator to list to complete transcription
        segments_list = list(segments)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert segments to JSON-compatible format
        segments_data = []
        for i, segment in enumerate(segments_list):
            segment_data = {
                "id": i,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }
            
            # Add word timestamps if requested
            if options.word_timestamps and hasattr(segment, 'words'):
                segment_data["words"] = [
                    {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
                    for word in segment.words
                ]
                
            segments_data.append(segment_data)
        
        # Store result
        result = {
            "segments": segments_data,
            "language": info.language,
            "language_probability": info.language_probability,
            "processing_time": processing_time,
            "audio_duration": segments_list[-1].end if segments_list else 0,
        }
        
        transcription_tasks[task_id]["status"] = "completed"
        transcription_tasks[task_id]["result"] = result
        
        # Clean up
        logger.info(f"Transcription completed for task {task_id}")
        
    except Exception as e:
        logger.exception(f"Error processing transcription: {str(e)}")
        transcription_tasks[task_id]["status"] = "failed"
        transcription_tasks[task_id]["error"] = str(e)
    finally:
        # Remove temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

# Function to parse form fields into TranscriptionOptions
def parse_form_options(
    model_size: str = Form("large-v3"),
    device: str = Form("cuda"),
    compute_type: str = Form("float16"),
    language: Optional[str] = Form(None),
    batch_size: int = Form(16),
    beam_size: int = Form(5),
    word_timestamps: str = Form("false"),
    vad_filter: str = Form("true"),
    condition_on_previous_text: str = Form("true"),
    use_batched_mode: str = Form("true")
) -> TranscriptionOptions:
    """Parse form fields into TranscriptionOptions"""
    # Convert string booleans to actual booleans
    word_timestamps_bool = word_timestamps.lower() == "true"
    vad_filter_bool = vad_filter.lower() == "true"
    condition_on_previous_text_bool = condition_on_previous_text.lower() == "true"
    use_batched_mode_bool = use_batched_mode.lower() == "true"
    
    return TranscriptionOptions(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
        beam_size=beam_size,
        word_timestamps=word_timestamps_bool,
        vad_filter=vad_filter_bool,
        condition_on_previous_text=condition_on_previous_text_bool,
        use_batched_mode=use_batched_mode_bool
    )

@app.post("/api/transcribe", response_model=TranscriptionTask)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: TranscriptionOptions = Depends(parse_form_options)
):
    """
    Transcribe audio file using faster-whisper
    """
    # Log received options
    logger.info(f"Received transcription request with options: {options}")
    
    # Generate a unique ID for this task
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    
    # Create a new task
    task = {
        "id": task_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "file_name": file.filename,
    }
    
    transcription_tasks[task_id] = task
    
    # Process in background
    background_tasks.add_task(
        process_transcription,
        task_id=task_id,
        file_path=str(file_path),
        options=options
    )
    
    return JSONResponse(status_code=202, content=task)

@app.get("/api/tasks/{task_id}", response_model=TranscriptionTask)
async def get_task(task_id: str):
    """
    Get the status of a transcription task
    """
    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return transcription_tasks[task_id]

@app.get("/api/tasks", response_model=List[TranscriptionTask])
async def list_tasks(limit: int = 10, status: Optional[str] = None):
    """
    List transcription tasks
    """
    tasks = list(transcription_tasks.values())
    
    if status:
        tasks = [task for task in tasks if task["status"] == status]
    
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    return tasks[:limit]

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a transcription task
    """
    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del transcription_tasks[task_id]
    
    return {"status": "deleted", "task_id": task_id}

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Get host from environment or use default
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Start server
    uvicorn.run("app:app", host=host, port=port, reload=True)