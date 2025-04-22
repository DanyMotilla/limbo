# LIMBO - Lattice Interior Mesh Builder & Optimizer

A powerful web application for generating optimized gyroid lattice structures in 3D models, perfect for prosthetics and lightweight structural components. Future versions will introduce complete automation of prosthetic design, making customized medical solutions more accessible than ever.

> Made with love by a crow and a moth who dream of changing the world, one line of code at a time.

## Features

- **Dual Lattice Modes**:
  - Surface Lattice: Creates patterns that follow the object's surface
  - Volume Lattice: Fills the entire interior with a gyroid pattern

- **Customizable Parameters**:
  - Pattern Scale: Control cell size and density (0.5 to 10.0)
  - Pattern Thickness: Adjust wall thickness (0.05 to 0.5)
  - Resolution: Balance between quality and processing speed (20 to 100)

- **Smart Processing**:
  - Automatic mesh simplification for large models
  - Memory-efficient chunked processing
  - Real-time progress tracking
  - Robust error handling

## Technical Details

### Surface Mode Implementation

- Uses signed distance field approach for continuous surfaces
- Adaptive surface thickness for better detail
- Smart dilation/erosion for hollow shell generation
- Resolution automatically adjusted for surface quality

### Memory Optimization

- Chunked processing for both containment and gyroid generation
- Efficient numpy array operations
- Automatic garbage collection of intermediate results

## Tech Stack

### Frontend
- React 18.2.0
- Three.js with @react-three/fiber and @react-three/drei
- Replicad for CAD operations
- Vite as build tool

### Backend
- FastAPI
- Python 3.8+
- NumPy, SciPy for mesh processing
- Trimesh for STL handling

## Getting Started

### Prerequisites
- Node.js 16+
- Python 3.8+
- pip (Python package manager)
- At least 4GB RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LIMBO.git
cd LIMBO
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
cd api
pip install -r requirements.txt
```

### Running the Application

1. Start the frontend development server:
```bash
npm run start
```
The application will be available at http://localhost:5173

2. Start the backend server (in a new terminal):
```bash
cd api
python3 main.py
```
The API server will start on http://localhost:8000

## API Documentation

### Endpoints

#### POST /generate_gyroid
Generate a gyroid lattice structure from an STL file.

**Parameters:**
- `stl_file`: STL file (multipart/form-data)
  - Maximum vertices: 1,000,000
- `mode`: String ("surface" or "volume")
- `period`: Float (0.5 to 10.0) - Controls cell size
- `thickness`: Float (0.05 to 0.5) - Controls wall thickness
- `resolution`: Integer (20 to 100) - Controls mesh quality

**Response:**
```json
{
    "job_id": "uuid-string"
}
```

**Resource Limits:**
- Maximum input mesh size: 1,000,000 vertices
- Maximum output mesh size: 1,000,000 vertices
- Maximum memory usage: 50% of system RAM
- Maximum resolution: 100

#### GET /job/{job_id}
Check the status of a gyroid generation job.

**Response:**
```json
{
    "id": "uuid-string",
    "status": "pending|processing|completed|failed",
    "progress": 0.0 to 1.0,
    "result_file": "path/to/file.stl",
    "error": null
}
```

The `progress` field provides real-time feedback on the generation process:
- 0.0-0.4: Initial setup and mesh loading
- 0.4-0.6: Containment analysis
- 0.6-0.9: Gyroid pattern generation
- 0.9-1.0: Final mesh processing and export

#### GET /job/{job_id}/result
Download the generated STL file.

## Usage Tips

1. **Surface Lattice Mode**:
   - Best for creating textured surfaces
   - Use lower thickness values for delicate patterns
   - Higher resolution gives smoother surface detail
   - Recommended for decorative purposes

2. **Volume Lattice Mode**:
   - Ideal for lightweight internal structures
   - Adjust pattern scale to control cell density
   - Balance between strength and weight reduction
   - Perfect for functional parts

3. **Performance Optimization**:
   - Start with lower resolution (40-60) for quick previews
   - For large models, the system will automatically simplify the mesh
   - Monitor memory usage when processing complex models
   - Consider splitting very large models into smaller parts

## Configuration

- **Backend Settings**:
  - Default port: 8000
  - Maximum workers: 4
  - Memory monitoring enabled
  - CORS enabled for development

- **Frontend Settings**:
  - Development port: 5173
  - API retry attempts: 3
  - Progress polling interval: 1s

## Error Handling

The application includes comprehensive error handling for:
- Invalid input files
- Memory exhaustion
- Network issues
- Processing failures
- Invalid surface generation

## Known Limitations

- Maximum recommended input mesh size: 1 million faces
- Surface mode requires minimum resolution of 80 for quality results
- Memory usage scales with resolution³

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
