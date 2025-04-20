# LIMBO - Lattice Interior Mesh Builder & Optimizer

A powerful web application for generating optimized gyroid lattice structures in 3D models, perfect for prosthetics and lightweight structural components. Future versions will introduce complete automation of prosthetic design, making customized medical solutions more accessible than ever.

> Made with love by a crow and a moth who dream of changing the world, one line of code at a time.

## Features

- **Dual Lattice Modes**:
  - Surface Lattice: Creates patterns that follow the object's surface
  - Volume Lattice: Fills the entire interior with a gyroid pattern

- **Customizable Parameters**:
  - Pattern Scale: Control cell size and density
  - Pattern Thickness: Adjust wall thickness
  - Resolution: Balance between quality and processing speed

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
- `mode`: String ("surface" or "volume")
- `period`: Float (0.5 to 10.0) - Controls cell size
- `thickness`: Float (0.05 to 0.5) - Controls wall thickness
- `resolution`: Integer (40 to 150) - Controls mesh quality

**Response:**
```json
{
    "job_id": "uuid-string"
}
```

#### GET /job/{job_id}
Check the status of a gyroid generation job.

**Response:**
```json
{
    "id": "uuid-string",
    "status": "pending|processing|completed|failed",
    "progress": 0.0,
    "result_file": "path/to/file.stl",
    "error": null
}
```

#### GET /job/{job_id}/result
Download the generated STL file.

## Usage Tips

1. **Surface Lattice Mode**:
   - Best for creating textured surfaces
   - Use lower thickness values for delicate patterns
   - Higher resolution gives smoother surface detail

2. **Volume Lattice Mode**:
   - Ideal for lightweight internal structures
   - Adjust pattern scale to control cell density
   - Balance between strength and weight reduction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
