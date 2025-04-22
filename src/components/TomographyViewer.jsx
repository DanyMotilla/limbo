import React, { useState, useEffect, useRef } from 'react';
import { Box, Grid, Slider, Button, Typography } from '@mui/material';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const TomographyViewer = () => {
  const [fileId, setFileId] = useState(null);
  const [dimensions, setDimensions] = useState([0, 0, 0]);
  const [axialSlice, setAxialSlice] = useState(0);
  const [coronalSlice, setCoronalSlice] = useState(0);
  const [mesh, setMesh] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [axialImage, setAxialImage] = useState(null);
  const [coronalImage, setCoronalImage] = useState(null);
  const modelRef = useRef(null);

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload_tomography', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setFileId(data.file_id);
      setDimensions(data.dimensions);
      setAxialSlice(Math.floor(data.dimensions[0] / 2));
      setCoronalSlice(Math.floor(data.dimensions[1] / 2));
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  // Load slice images
  const loadSlice = async (axis, index) => {
    if (!fileId) return null;
    try {
      const response = await fetch(
        `http://localhost:8000/tomography_slice/${fileId}?axis=${axis}&slice_index=${index}`
      );
      const data = await response.json();
      return `data:image/png;base64,${data.image}`;
    } catch (error) {
      console.error('Error loading slice:', error);
      return null;
    }
  };

  // Update slices when indices change
  useEffect(() => {
    if (fileId) {
      loadSlice(0, axialSlice).then(setAxialImage);
      loadSlice(1, coronalSlice).then(setCoronalImage);
    }
  }, [fileId, axialSlice, coronalSlice]);

  // Process tomography to 3D mesh
  const processTomography = async () => {
    try {
      setProcessing(true);
      const response = await fetch(`http://localhost:8000/process_tomography/${fileId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to process tomography data');
      }
      
      // Get the blob data and create URL
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      // Load the STL file into Three.js
      const loader = new STLLoader();
      const geometry = await loader.loadAsync(url);
      
      // Create mesh with a nice material
      const material = new THREE.MeshPhongMaterial({
        color: 0x00ff00,
        specular: 0x111111,
        shininess: 200,
      });
      
      // Update the mesh state
      setMesh({ geometry, material });
      
      // Clean up the URL
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Error processing tomography:', error);
    } finally {
      setProcessing(false);
    }
  };

  // Set up Three.js scene
  useEffect(() => {
    if (!modelRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, modelRef.current.clientWidth / modelRef.current.clientHeight, 0.1, 1000);
    camera.position.z = 2;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(modelRef.current.clientWidth, modelRef.current.clientHeight);
    modelRef.current.innerHTML = '';
    modelRef.current.appendChild(renderer.domElement);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
    light1.position.set(1, 1, 1);
    scene.add(light1);

    const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
    light2.position.set(-1, -1, -1);
    scene.add(light2);

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };

    // Add mesh to scene if it exists
    if (mesh) {
      const threeMesh = new THREE.Mesh(mesh.geometry, mesh.material);
      // Center the mesh
      const box = new THREE.Box3().setFromObject(threeMesh);
      const center = box.getCenter(new THREE.Vector3());
      threeMesh.position.sub(center);
      scene.add(threeMesh);
      
      // Adjust camera to fit mesh
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      camera.position.z = maxDim * 2;
    }

    animate();

    // Cleanup
    return () => {
      renderer.dispose();
      if (mesh) {
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
    };
  }, [mesh, modelRef]);

  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      overflow: 'hidden',
      p: 2
    }}>
      {/* Controls */}
      <Box sx={{ mb: 2 }}>
        <input
          accept=".nii,.nii.gz"
          style={{ display: 'none' }}
          id="tomography-file"
          type="file"
          onChange={handleFileUpload}
        />
        <label htmlFor="tomography-file">
          <Button variant="contained" component="span" sx={{ mr: 2 }}>
            Upload Tomography Data
          </Button>
        </label>
        <Button
          variant="outlined"
          onClick={async () => {
            try {
              const response = await fetch('http://localhost:8000/sample_data');
              const data = await response.json();
              setFileId(data.file_id);
              setDimensions(data.dimensions);
              setAxialSlice(Math.floor(data.dimensions[0] / 2));
              setCoronalSlice(Math.floor(data.dimensions[1] / 2));
            } catch (error) {
              console.error('Error loading sample data:', error);
            }
          }}
          sx={{ mr: 2 }}
        >
          Load Sample Data
        </Button>
        {fileId && (
          <Button
            variant="contained"
            onClick={processTomography}
            disabled={processing}
            color="secondary"
          >
            {processing ? 'Processing...' : 'Generate 3D Model'}
          </Button>
        )}
      </Box>

      {/* Content */}
      <Grid container spacing={2} sx={{ flex: 1, minHeight: 0 }}>
        {/* Left side - Slices */}
        <Grid item xs={6} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Axial View */}
          <Box sx={{ flex: 1, minHeight: 0, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Axial View (Slice {axialSlice})
            </Typography>
            <Box sx={{ 
              height: 'calc(50% - 30px)', 
              border: '1px solid #ccc',
              borderRadius: 1,
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'background.paper'
            }}>
              {axialImage ? (
                <img
                  src={axialImage}
                  alt="Axial slice"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <Typography color="text.secondary">
                  {fileId ? 'Loading...' : 'Upload a file to view slices'}
                </Typography>
              )}
            </Box>
            <Box sx={{ px: 1, mt: 1 }}>
              <Slider
                value={axialSlice}
                onChange={(_, value) => setAxialSlice(value)}
                min={0}
                max={dimensions[0] - 1}
                disabled={!fileId}
                valueLabelDisplay="auto"
                size="small"
              />
            </Box>
          </Box>

          {/* Coronal View */}
          <Box sx={{ flex: 1, minHeight: 0 }}>
            <Typography variant="subtitle1" gutterBottom>
              Coronal View (Slice {coronalSlice})
            </Typography>
            <Box sx={{ 
              height: 'calc(50% - 30px)',
              border: '1px solid #ccc',
              borderRadius: 1,
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'background.paper'
            }}>
              {coronalImage ? (
                <img
                  src={coronalImage}
                  alt="Coronal slice"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <Typography color="text.secondary">
                  {fileId ? 'Loading...' : 'Upload a file to view slices'}
                </Typography>
              )}
            </Box>
            <Box sx={{ px: 1, mt: 1 }}>
              <Slider
                value={coronalSlice}
                onChange={(_, value) => setCoronalSlice(value)}
                min={0}
                max={dimensions[1] - 1}
                disabled={!fileId}
                valueLabelDisplay="auto"
                size="small"
              />
            </Box>
          </Box>
        </Grid>

        {/* Right side - 3D View */}
        <Grid item xs={6} sx={{ height: '100%' }}>
          <Typography variant="subtitle1" gutterBottom>
            3D Model
          </Typography>
          <Box sx={{ 
            height: 'calc(100% - 24px)',
            border: '1px solid #ccc',
            borderRadius: 1,
            overflow: 'hidden',
            bgcolor: 'background.paper'
          }} ref={modelRef}>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TomographyViewer;
