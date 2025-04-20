from fastapi import FastAPI, UploadFile, Form, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import trimesh
import tempfile
import os
import logging
import traceback
import gc
import psutil
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_fill_holes
from skimage import measure
import asyncio
import contextlib
import time
import uuid
from pydantic import BaseModel
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Job storage
class JobStatus(BaseModel):
    id: str
    status: str  # pending, processing, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    result_file: Optional[str] = None
    error: Optional[str] = None

active_jobs: Dict[str, JobStatus] = {}

app = FastAPI()

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Main development port
    "http://127.0.0.1:5173"  # Alternative localhost
]

# Development fallback - allow any localhost port if in development
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
job_pool = ThreadPoolExecutor(max_workers=1)

def create_error_response(status_code: int, detail: str) -> JSONResponse:
    """Create a properly formatted error response with CORS headers"""
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

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
        job = active_jobs[job_id]
        job.status = "processing"
        job.started_at = datetime.now().isoformat()
        
        # Generate the mesh
        success = generate_gyroid_mesh_sync(
            input_file,
            output_file,
            mode,
            period,
            thickness,
            resolution
        )
        
        if not success:
            job.status = "failed"
            job.error = "Failed to generate gyroid mesh"
            return
        
        job.status = "completed"
        job.result_file = output_file
        job.completed_at = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        
    finally:
        # Cleanup input file
        try:
            os.unlink(input_file)
        except:
            pass

@app.post("/generate_gyroid")
async def start_gyroid_generation(
    stl_file: UploadFile,
    mode: str = Form(...),
    period: float = Form(...),
    thickness: float = Form(...),
    resolution: int = Form(...)
):
    try:
        # Validate resolution
        if resolution > 150:
            return create_error_response(
                status.HTTP_400_BAD_REQUEST,
                "Resolution cannot exceed 150 due to memory constraints"
            )
        
        # Create job ID and temporary files
        job_id = str(uuid.uuid4())
        input_file = tempfile.mktemp(suffix=".stl")
        output_file = tempfile.mktemp(suffix=".stl")
        
        # Save uploaded file
        content = await stl_file.read()
        with open(input_file, "wb") as f:
            f.write(content)
            
        # Create job status
        active_jobs[job_id] = JobStatus(
            id=job_id,
            status="pending",
            created_at=datetime.now().isoformat()
        )
        
        # Start background job
        job_pool.submit(
            process_gyroid_job,
            job_id,
            input_file,
            output_file,
            mode,
            period,
            thickness,
            resolution
        )
        
        return {"job_id": job_id}
        
    except Exception as e:
        logger.error(f"Error starting job: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to start gyroid generation: {str(e)}"
        )

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job"""
    try:
        job = active_jobs.get(job_id)
        if not job:
            return create_error_response(
                status.HTTP_404_NOT_FOUND,
                "Job not found"
            )
            
        response = job.dict()
        
        # Clean up completed jobs older than 1 hour
        if job.status in ["completed", "failed"]:
            if job.completed_at:
                completed_time = datetime.fromisoformat(job.completed_at)
                if (datetime.now() - completed_time).total_seconds() > 3600:
                    del active_jobs[job_id]
                    if job.result_file and os.path.exists(job.result_file):
                        os.unlink(job.result_file)
                    
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            str(e)
        )

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the result file for a completed job"""
    try:
        job = active_jobs.get(job_id)
        if not job:
            return create_error_response(
                status.HTTP_404_NOT_FOUND,
                "Job not found"
            )
            
        if job.status != "completed":
            return create_error_response(
                status.HTTP_400_BAD_REQUEST,
                f"Job is not completed (status: {job.status})"
            )
            
        if not job.result_file or not os.path.exists(job.result_file):
            return create_error_response(
                status.HTTP_404_NOT_FOUND,
                "Result file not found"
            )
            
        with open(job.result_file, "rb") as f:
            content = f.read()
            
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="gyroid.stl"',
                "Content-Length": str(len(content))
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting job result: {str(e)}")
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            str(e)
        )

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage - RSS: {mem_info.rss / 1024 / 1024:.2f}MB, VMS: {mem_info.vms / 1024 / 1024:.2f}MB")

def chunk_size_for_resolution(resolution):
    """Calculate appropriate chunk size based on resolution"""
    if resolution <= 64:
        return resolution
    elif resolution <= 128:
        return resolution // 2
    else:
        return resolution // 4

def generate_gyroid_mesh_sync(
    input_path: str,
    output_path: str,
    mode: str,
    period: float,
    thickness: float,
    resolution: int
) -> bool:
    """Synchronous version of gyroid mesh generation"""
    try:
        log_memory_usage()
        
        # Load input mesh
        mesh = trimesh.load_mesh(input_path)
        if mesh is None:
            logger.error("Failed to load input mesh")
            return False
            
        # Scale mesh to fit in unit cube
        extents = mesh.extents
        scale = 1.0 / max(extents)
        translation = -mesh.bounds[0]
        mesh.apply_translation(translation)
        mesh.apply_scale(scale)
        
        # Create grid points for containment check
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        z = np.linspace(0, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # Scale points to mesh space
        points = points * 2 - 1  # Scale to [-1, 1]
        
        # Check which points are inside the mesh
        inside = mesh.contains(points)
        mask = inside.reshape(resolution, resolution, resolution)
        
        # Generate gyroid field
        gyroid = np.sin(X * 2 * np.pi * period) * np.cos(Y * 2 * np.pi * period) + \
                np.sin(Y * 2 * np.pi * period) * np.cos(Z * 2 * np.pi * period) + \
                np.sin(Z * 2 * np.pi * period) * np.cos(X * 2 * np.pi * period)
        
        # Generate final mesh
        if mode == "surface":  # Surface Lattice (Shell Only)
            # First create a thin shell of the original mesh
            shell_distance = 0.05  # Small offset from surface
            shell = np.zeros_like(mask, dtype=float)
            shell[1:-1, 1:-1, 1:-1] = mask[1:-1, 1:-1, 1:-1] & ~mask[:-2, 1:-1, 1:-1] | \
                                     mask[1:-1, 1:-1, 1:-1] & ~mask[2:, 1:-1, 1:-1] | \
                                     mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, :-2, 1:-1] | \
                                     mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, 2:, 1:-1] | \
                                     mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, 1:-1, :-2] | \
                                     mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, 1:-1, 2:]
            
            # Apply gyroid pattern only to shell region
            gyroid = gyroid * shell
            
            # Generate surface mesh
            verts, faces = measure.marching_cubes(gyroid, level=thickness)[:2]
            
        else:  # mode == "volume" - Volume Lattice (Full Interior)
            # Apply mask to constrain to mesh volume
            gyroid = gyroid * mask
            
            # Direct marching cubes for volume lattice
            verts, faces = measure.marching_cubes(gyroid, level=thickness)[:2]
        
        # Scale vertices back to original size
        verts = verts / resolution
        verts = verts * 2 - 1  # Scale to [-1,1]
        verts = verts / scale  # Scale back to original size
        verts = verts - translation  # Move back to original position
        
        # Create output mesh
        output_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Save result
        output_mesh.export(output_path)
        
        log_memory_usage()
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating mesh: {str(e)}")
        logger.error(traceback.format_exc())
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
