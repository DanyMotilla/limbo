from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Response, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import trimesh
import tempfile
import os
import logging
import traceback
import gc
import psutil
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_fill_holes
from skimage import measure
import asyncio
import contextlib
import time
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
import pymeshlab
import shutil
from contextlib import asynccontextmanager
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for chunking and processing
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 16
MAX_WORKERS = 1
MEMORY_THRESHOLD = 0.6
MAX_RESOLUTION = 96

# Added strict memory and size limits
MAX_VERTICES = 1000000  # Maximum vertices in input mesh
MAX_RESOLUTION = 100    # Maximum grid resolution
MAX_MEMORY_PERCENT = 50 # Maximum memory usage allowed (%)

def check_memory_usage():
    """Check if memory usage is too high"""
    memory = psutil.Process(os.getpid()).memory_percent()
    if memory > MAX_MEMORY_PERCENT:
        raise ValueError(f"Memory usage too high ({memory:.1f}%). Try reducing resolution.")
    return memory

# Job storage
class JobStatus(BaseModel):
    id: str
    status: str = "pending"
    progress: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    result_file: Optional[str] = None
    temp_dir: Optional[str] = None

jobs: Dict[str, JobStatus] = {}

# Create a directory for storing temporary files
TEMP_ROOT = os.path.join(os.path.dirname(__file__), "temp")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    try:
        # Create temp directory on startup
        os.makedirs(TEMP_ROOT, exist_ok=True)
        logger.info(f"Created temp directory at {TEMP_ROOT}")
        yield
    finally:
        # Clean up temp directory on shutdown
        if os.path.exists(TEMP_ROOT):
            logger.info(f"Cleaning up temp directory at {TEMP_ROOT}")
            shutil.rmtree(TEMP_ROOT)

app = FastAPI(lifespan=lifespan)

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

if os.getenv("ENVIRONMENT") != "production":
    ALLOWED_ORIGINS.extend([
        "http://localhost:*",
        "http://127.0.0.1:*"
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"],
)

# Thread pool for background jobs
job_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def create_error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

def get_memory_usage() -> float:
    """Get current memory usage percentage"""
    return psutil.Process().memory_percent()

def chunk_coordinates(resolution: int) -> List[Tuple[slice, slice, slice]]:
    """Generate optimal chunks based on resolution and available memory"""
    chunk_size = min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, resolution // 4))
    chunks = []
    
    for x in range(0, resolution, chunk_size):
        for y in range(0, resolution, chunk_size):
            for z in range(0, resolution, chunk_size):
                x_end = min(x + chunk_size, resolution)
                y_end = min(y + chunk_size, resolution)
                z_end = min(z + chunk_size, resolution)
                chunks.append((
                    slice(x, x_end),
                    slice(y, y_end),
                    slice(z, z_end)
                ))
    return chunks

def process_chunk(chunk: Tuple[slice, slice, slice], points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Process a single chunk of points for containment"""
    chunk_points = points[chunk]
    chunk_inside = mesh.contains(chunk_points.reshape(-1, 3))
    return chunk_inside.reshape(chunk_points.shape[:-1])

def generate_gyroid_field(chunk: Tuple[slice, slice, slice], period: float, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Generate gyroid field for a specific chunk"""
    x_chunk, y_chunk, z_chunk = chunk
    return (np.sin(X[x_chunk, y_chunk, z_chunk] * 2 * np.pi * period) * 
            np.cos(Y[x_chunk, y_chunk, z_chunk] * 2 * np.pi * period) +
            np.sin(Y[x_chunk, y_chunk, z_chunk] * 2 * np.pi * period) * 
            np.cos(Z[x_chunk, y_chunk, z_chunk] * 2 * np.pi * period) +
            np.sin(Z[x_chunk, y_chunk, z_chunk] * 2 * np.pi * period) * 
            np.cos(X[x_chunk, y_chunk, z_chunk] * 2 * np.pi * period))

def process_gyroid_job(
    job_id: str,
    input_file: str,
    output_file: str,
    mode: str,
    period: float,
    thickness: float,
    resolution: int
):
    """Background job processor"""
    try:
        job = jobs[job_id]
        job.status = "processing"
        
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            job.status = "failed"
            job.error = "Input file not found"
            return
            
        # Check if output directory exists
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            logger.error(f"Output directory not found: {output_dir}")
            job.status = "failed"
            job.error = "Output directory not found"
            return
        
        logger.info(f"Processing job {job_id} with input={input_file}, output={output_file}")
        success = generate_gyroid_mesh_sync(
            input_file,
            output_file,
            mode,
            period,
            thickness,
            resolution,
            job
        )
        
        if not success:
            job.status = "failed"
            job.error = "Failed to generate gyroid mesh"
            return
            
        # Verify the output file exists
        if not os.path.exists(output_file):
            logger.error(f"Output file not found after generation: {output_file}")
            job.status = "failed"
            job.error = "Output file not found after generation"
            return
            
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        job.status = "failed"
        job.error = str(e)

def cleanup_temp_dir(job: JobStatus):
    """Clean up temporary directory for a job"""
    try:
        if job.temp_dir and os.path.exists(job.temp_dir):
            logger.info(f"Cleaning up temp dir: {job.temp_dir}")
            shutil.rmtree(job.temp_dir)
    except Exception as e:
        logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.post("/generate_gyroid")
async def generate_gyroid(
    background_tasks: BackgroundTasks,
    stl_file: UploadFile = File(...),
    mode: str = Form(default="surface"),
    period: float = Form(default=3.0),
    thickness: float = Form(default=0.5),
    resolution: int = Form(default=60)
):
    try:
        # Validate inputs
        if resolution < 20 or resolution > MAX_RESOLUTION:
            logger.error(f"Invalid resolution: {resolution}. Must be between 20 and {MAX_RESOLUTION}")
            raise HTTPException(status_code=400, detail=f"Resolution must be between 20 and {MAX_RESOLUTION}")
        
        if period <= 0:
            logger.error(f"Invalid period: {period}. Must be positive")
            raise HTTPException(status_code=400, detail="Period must be positive")
        
        if thickness <= 0 or thickness >= 1:
            logger.error(f"Invalid thickness: {thickness}. Must be between 0 and 1")
            raise HTTPException(status_code=400, detail="Thickness must be between 0 and 1")
        
        if mode not in ["surface", "volume"]:
            logger.error(f"Invalid mode: {mode}. Must be 'surface' or 'volume'")
            raise HTTPException(status_code=400, detail="Mode must be 'surface' or 'volume'")
        
        logger.info(f"Received request - mode: {mode}, period: {period}, thickness: {thickness}, resolution: {resolution}")
        
        # Create job-specific temp directory
        job_id = str(uuid.uuid4())
        temp_dir = os.path.join(TEMP_ROOT, job_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        if not os.path.exists(temp_dir):
            logger.error(f"Failed to create temp dir: {temp_dir}")
            raise HTTPException(status_code=500, detail="Failed to create temporary directory")
        
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input.stl")
        output_path = os.path.join(temp_dir, "output.stl")
        
        try:
            contents = await stl_file.read()
            with open(input_path, "wb") as f:
                f.write(contents)
                
            # Verify the file exists and is readable
            if not os.path.exists(input_path):
                raise ValueError(f"Failed to save file to {input_path}")
                
            # Try to read it with trimesh to validate it's a valid STL
            try:
                mesh = trimesh.load(input_path)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError("Not a valid STL mesh")
            except Exception as e:
                raise ValueError(f"Invalid STL file: {str(e)}")
                
        except Exception as e:
            # Clean up temp dir on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.error(f"Error handling uploaded file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid STL file: {str(e)}")
        
        # Create job status
        job = JobStatus(
            id=job_id,
            status="pending",
            progress=0.0,
            created_at=datetime.now().isoformat(),
            temp_dir=temp_dir,
            result_file=None
        )
        jobs[job_id] = job
        
        # Start processing in background
        background_tasks.add_task(
            process_gyroid_job,
            job_id=job_id,
            input_file=input_path,
            output_file=output_path,
            mode=mode,
            period=period,
            thickness=thickness,
            resolution=resolution
        )
        
        return {"job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_gyroid: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Log job status
    logger.info(f"Job {job_id} status: {job.status}, progress: {job.progress}")
    if job.error:
        logger.error(f"Job {job_id} error: {job.error}")
    
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "result_file": job.result_file if job.status == "completed" else None
    }

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed, status: {job.status}")
    
    if not job.result_file:
        raise HTTPException(status_code=404, detail="No result file available")
    
    if not os.path.exists(job.result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        response = FileResponse(
            job.result_file,
            media_type="model/stl",
            filename="output.stl"
        )
        
        # Only clean up after successful file send
        background_tasks.add_task(cleanup_temp_dir, job)
        response.background = background_tasks
        
        return response
        
    except Exception as e:
        logger.error(f"Error sending result file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def simplify_mesh(mesh, target_faces=10000):
    """Simplify a mesh using pymeshlab"""
    try:
        # Create a new MeshSet
        ms = pymeshlab.MeshSet()
        
        # Save the mesh to a temporary file
        temp_input = tempfile.mktemp(suffix=".stl")
        mesh.export(temp_input)
        
        # Load the mesh into pymeshlab
        ms.load_new_mesh(temp_input)
        
        # Simplify the mesh
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            planarquadric=True
        )
        
        # Save the simplified mesh
        temp_output = tempfile.mktemp(suffix=".stl")
        ms.save_current_mesh(temp_output)
        
        # Load the simplified mesh back into trimesh
        simplified_mesh = trimesh.load_mesh(temp_output)
        
        # Cleanup temporary files
        os.unlink(temp_input)
        os.unlink(temp_output)
        
        return simplified_mesh
    except Exception as e:
        logger.warning(f"Pymeshlab simplification failed: {str(e)}")
        return mesh

def generate_gyroid_mesh_sync(
    input_path: str,
    output_path: str,
    mode: str,
    period: float,
    thickness: float,
    resolution: int,
    job: JobStatus
) -> bool:
    try:
        logger.info(f"Starting mesh generation with resolution {resolution}")
        logger.info(f"Parameters: mode={mode}, period={period}, thickness={thickness}")

        # Load the input mesh
        mesh = trimesh.load(input_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Not a valid STL mesh")

        # Check mesh size
        if len(mesh.vertices) > MAX_VERTICES:
            raise ValueError(f"Input mesh too large ({len(mesh.vertices)} vertices). Maximum allowed: {MAX_VERTICES}")

        # Check resolution
        if resolution > MAX_RESOLUTION:
            raise ValueError(f"Resolution too high ({resolution}). Maximum allowed: {MAX_RESOLUTION}")

        # Log mesh info
        logger.info(f"Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        logger.info(f"Mesh bounds: {mesh.bounds}")
        memory = check_memory_usage()
        logger.info(f"Current memory usage: {memory:.1f}%")

        # Get mesh bounds and scale before simplification
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        scale = 1.0 / max(extents)

        # Simplify the input mesh first to reduce processing time
        logger.info(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        target_faces = min(5000, len(mesh.faces))  # More aggressive simplification
        if len(mesh.faces) > target_faces:
            mesh = simplify_mesh(mesh, target_faces)
            logger.info(f"Simplified mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Scale the mesh to unit cube for containment check first (do this before grid creation)
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_translation(-bounds[0])
        mesh_scaled.apply_scale(scale)
        del mesh  # Free original mesh memory

        # Create regular grid with reduced resolution for containment check
        containment_resolution = min(60, resolution)  # Use lower resolution for containment
        x = np.linspace(0, 1, containment_resolution)
        y = np.linspace(0, 1, containment_resolution)
        z = np.linspace(0, 1, containment_resolution)
        
        # More efficient chunking for containment check
        chunk_size = min(4000, containment_resolution * containment_resolution)
        total_points = containment_resolution * containment_resolution * containment_resolution
        inside = np.zeros(total_points, dtype=bool)
        
        total_chunks = total_points // chunk_size + (1 if total_points % chunk_size else 0)
        logger.info(f"Processing containment in {total_chunks} chunks at resolution {containment_resolution}")
        
        # Process containment check with larger chunks but lower resolution
        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)
            chunk_points = np.zeros((end_idx - i, 3))
            flat_indices = np.arange(i, end_idx)
            chunk_points[:, 0] = x[flat_indices // (containment_resolution * containment_resolution)]
            chunk_points[:, 1] = y[(flat_indices // containment_resolution) % containment_resolution]
            chunk_points[:, 2] = z[flat_indices % containment_resolution]
            
            inside[i:end_idx] = mesh_scaled.contains(chunk_points)
            job.progress = 0.4 + 0.2 * (i / total_points)
            check_memory_usage()
            del chunk_points
            gc.collect()
        
        del mesh_scaled
        
        # Reshape and resize to target resolution if needed
        inside = inside.reshape((containment_resolution, containment_resolution, containment_resolution))
        if containment_resolution != resolution:
            # Use nearest neighbor interpolation for boolean array
            from scipy.ndimage import zoom
            zoom_factor = resolution / containment_resolution
            inside = zoom(inside.astype(float), zoom_factor, order=0).astype(bool)
            logger.info(f"Upscaled containment mask to target resolution {resolution}")
        
        # Create final resolution grid for gyroid generation
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        z = np.linspace(0, 1, resolution)
        
        # Improved chunking for containment check
        chunk_size = min(2000, resolution * resolution)  # Smaller chunks for better memory management
        total_points = resolution * resolution * resolution
        gyroid = np.zeros(total_points, dtype=float)
        
        total_chunks = total_points // chunk_size + (1 if total_points % chunk_size else 0)
        logger.info(f"Processing gyroid in {total_chunks} chunks")
        
        # Generate points in smaller chunks to avoid memory issues
        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)
            # Generate only the points we need for this chunk
            chunk_points = np.zeros((end_idx - i, 3))
            flat_indices = np.arange(i, end_idx)
            chunk_points[:, 0] = x[flat_indices // (resolution * resolution)]
            chunk_points[:, 1] = y[(flat_indices // resolution) % resolution]
            chunk_points[:, 2] = z[flat_indices % resolution]
            
            # Process gyroid for this chunk
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Regular meshgrid for gyroid
            if mode == "surface":
                gyroid[i:end_idx] = (np.sin(X[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) * 
                                    np.cos(Y[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) +
                                    np.sin(Y[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) * 
                                    np.cos(Z[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) +
                                    np.sin(Z[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) * 
                                    np.cos(X[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)))
            else:  # volume
                gyroid[i:end_idx] = (np.sin(X[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) * 
                                    np.cos(Y[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) +
                                    np.sin(Y[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) * 
                                    np.cos(Z[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) +
                                    np.sin(Z[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) * 
                                    np.cos(X[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution] * 2 * np.pi / (period * scale)) - 
                                    thickness)
            
            # Only keep gyroid inside the mesh
            gyroid[i:end_idx] = np.where(inside[flat_indices // (resolution * resolution), (flat_indices // resolution) % resolution, flat_indices % resolution], gyroid[i:end_idx], 1.0 if mode == "volume" else 0.0)
            
            # Update progress and check memory
            job.progress = 0.7 + 0.2 * (i / total_points)
            check_memory_usage()
            
            # Explicitly clear memory
            del chunk_points
            gc.collect()
        
        gyroid = gyroid.reshape((resolution, resolution, resolution))
        
        # Generate mesh using marching cubes
        verts, faces = measure.marching_cubes(gyroid, level=0.0)[:2]
        
        # Scale vertices back to original size
        verts = verts / (resolution - 1)  # Normalize to [0,1]
        verts = verts * extents + bounds[0]  # Scale to original size
        
        # Create mesh
        result_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Check output mesh size
        if len(result_mesh.vertices) > MAX_VERTICES:
            raise ValueError(f"Output mesh too large ({len(result_mesh.vertices)} vertices). Try reducing resolution.")
        
        logger.info(f"Output mesh: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")
        memory = check_memory_usage()
        logger.info(f"Final memory usage: {memory:.1f}%")
        
        job.progress = 0.9

        # Save the mesh
        try:
            result_mesh.export(output_path)
            if not os.path.exists(output_path):
                raise ValueError("Failed to save output mesh")
            
            # Update job status
            job.status = "completed"
            job.progress = 1.0
            job.result_file = output_path
            
            logger.info(f"Successfully generated mesh at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save mesh: {str(e)}")
            job.status = "failed"
            job.error = f"Failed to save mesh: {str(e)}"
            return False

    except Exception as e:
        logger.error(f"Error in generate_gyroid_mesh_sync: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        job.status = "failed"
        job.error = str(e)
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600,
        workers=1
    )
