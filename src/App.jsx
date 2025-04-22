import React, { useState, useEffect } from "react";
import FileSaver from "file-saver";
import { wrap } from "comlink";

import ThreeContext from "./ThreeContext.jsx";
import ReplicadMesh from "./ReplicadMesh.jsx";
import TabPanel from "./components/TabPanel.jsx";
import { generateGyroid } from "./services/gyroidService.js";
import { Tabs, Tab, Box } from '@mui/material';
import TomographyViewer from './components/TomographyViewer';

import cadWorker from "./worker.js?worker";
const cad = wrap(new cadWorker());

export default function ReplicadApp() {
  const [size, setSize] = useState(5);
  const [leftActiveTab, setLeftActiveTab] = useState(0);
  const [rightActiveTab, setRightActiveTab] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  // Gyroid parameters
  const [gyroidMode, setGyroidMode] = useState('volume');
  const [period, setPeriod] = useState(1.5);
  const [thickness, setThickness] = useState(0.2);
  const [resolution, setResolution] = useState(60);

  // Reset error when parameters change
  useEffect(() => {
    setError(null);
  }, [gyroidMode, period, thickness, resolution]);

  const downloadModel = async () => {
    const blob = await cad.createBlob(size);
    FileSaver.saveAs(blob, "thing.stl");
  };

  const generateGyroidModel = async () => {
    try {
      setIsProcessing(true);
      setProgress(0);
      setError(null);
      
      // Get the current model as STL
      const stlBlob = await cad.createBlob(size);
      
      // Generate gyroid with progress callback
      const gyroidBlob = await generateGyroid(stlBlob, {
        mode: gyroidMode,
        period,
        thickness,
        resolution,
        onProgress: (progress) => {
          setProgress(progress);
          console.log('Progress:', Math.round(progress * 100) + '%');
        }
      });

      // Save the file
      FileSaver.saveAs(gyroidBlob, "gyroid_model.stl");
      setProgress(1); // Show 100% completion
    } catch (error) {
      console.error('Error generating gyroid:', error);
      setError(error.message || 'Failed to generate gyroid model');
    } finally {
      setIsProcessing(false);
    }
  };

  const [mesh, setMesh] = useState(null);

  useEffect(() => {
    cad.createMesh(size).then((m) => setMesh(m));
  }, [size]);

  // Parameters Panel Content
  const ParametersPanel = (
    <div style={{
      padding: '1rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      height: '100%',
      overflow: 'auto'
    }}>
      <div>
        <h3 style={{ marginBottom: '1rem' }}>Base Shape Parameters</h3>
        <label htmlFor="thicknessInput" style={{ display: 'block', marginBottom: '0.5rem' }}>
          Thickness
        </label>
        <input
          id="thicknessInput"
          type="number"
          step="1"
          min="1"
          max="10"
          value={size}
          onChange={(v) => {
            const val = parseInt(v.target.value);
            if (val > 0 && val <= 10) setSize(val);
          }}
          style={{
            width: '100%',
            padding: '0.5rem',
            borderRadius: '4px',
            border: '1px solid #ccc'
          }}
        />
      </div>

      <div style={{ marginTop: '1rem' }}>
        <h3 style={{ marginBottom: '1rem' }}>Gyroid Parameters</h3>
        
        {error && (
          <div style={{
            padding: '0.75rem',
            marginBottom: '1rem',
            backgroundColor: '#f8d7da',
            color: '#721c24',
            borderRadius: '4px',
            border: '1px solid #f5c6cb'
          }}>
            {error}
          </div>
        )}
        
        <div style={{ marginBottom: '2rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold'
          }}>
            Pattern Type
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: '#666', 
            marginBottom: '0.5rem' 
          }}>
            Choose how the gyroid pattern is applied to your model
          </p>
          <select 
            value={gyroidMode}
            onChange={(e) => setGyroidMode(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem',
              borderRadius: '4px',
              border: '1px solid #ccc'
            }}
          >
            <option value="surface">Surface Lattice - Pattern follows the object's surface</option>
            <option value="volume">Volume Lattice - Pattern fills the entire interior</option>
          </select>
        </div>

        {gyroidMode === 'surface' && (
          <div style={{
            padding: '0.75rem',
            marginBottom: '1rem',
            backgroundColor: '#cce5ff',
            color: '#004085',
            borderRadius: '4px',
            border: '1px solid #b8daff',
            fontSize: '0.9em'
          }}>
            Surface mode requires higher resolution for good quality. Resolution will be automatically adjusted to minimum 80.
          </div>
        )}
        
        <div style={{ marginBottom: '2rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold'
          }}>
            Pattern Scale: {period}
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: '#666', 
            marginBottom: '0.5rem' 
          }}>
            Controls the size and number of gyroid cells. Higher values create more cells in a denser pattern, lower values create fewer, larger cells.
          </p>
          <input
            type="range"
            min="0.5"
            max="10.0"
            step="0.1"
            value={period}
            onChange={(e) => setPeriod(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8em',
            color: '#666'
          }}>
            <span>Few large cells</span>
            <span>Many small cells</span>
          </div>
        </div>

        <div style={{ marginBottom: '2rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold'  
          }}>
            Pattern Thickness: {thickness}
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: '#666', 
            marginBottom: '0.5rem' 
          }}>
            {gyroidMode === 'surface' 
              ? "Controls the thickness of the lattice walls on the surface. Lower values create more delicate patterns, higher values make them more robust."
              : "Controls the thickness of the internal lattice structure. Lower values create more void space, higher values create a denser structure."}
          </p>
          <input
            type="range"
            min="0.05"
            max="0.5"
            step="0.05"
            value={thickness}
            onChange={(e) => setThickness(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8em',
            color: '#666'
          }}>
            <span>Thin walls</span>
            <span>Thick walls</span>
          </div>
        </div>

        <div style={{ marginBottom: '2rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold'
          }}>
            Resolution: {resolution}
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: '#666', 
            marginBottom: '0.5rem' 
          }}>
            Controls the quality of the generated pattern. Higher values create smoother, more detailed patterns but require more processing time.
          </p>
          <input
            type="range"
            min="40"
            max="150"
            step="5"
            value={resolution}
            onChange={(e) => setResolution(parseInt(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8em',
            color: '#666'
          }}>
            <span>Faster, rougher</span>
            <span>Slower, smoother</span>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: 'auto', padding: '1rem 0' }}>
        {isProcessing && (
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ marginBottom: '0.5rem', fontSize: '0.9em', color: '#666' }}>
              Generating gyroid pattern: {Math.round(progress * 100)}%
            </div>
            <div style={{ 
              width: '100%', 
              height: '4px', 
              backgroundColor: '#e9ecef',
              borderRadius: '2px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${progress * 100}%`,
                height: '100%',
                backgroundColor: '#007bff',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        )}
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            onClick={downloadModel}
            style={{
              flex: 1,
              padding: '0.5rem 1rem',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Download Base STL
          </button>
          <button 
            onClick={generateGyroidModel}
            disabled={isProcessing}
            style={{
              flex: 1,
              padding: '0.5rem 1rem',
              backgroundColor: isProcessing ? '#ccc' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isProcessing ? 'not-allowed' : 'pointer'
            }}
          >
            {isProcessing ? `Generating (${Math.round(progress * 100)}%)` : 'Generate Gyroid'}
          </button>
        </div>
      </div>
    </div>
  );

  // Node System Panel Content
  const NodeSystemPanel = (
    <div style={{
      padding: '1rem',
      backgroundColor: '#f8f9fa',
      height: '100%',
      overflow: 'auto'
    }}>
      <p style={{ margin: 0 }}>Node system interface will be implemented here...</p>
    </div>
  );

  // Tomography View
  const TomographyView = (
    <div style={{ 
      height: '100%',
      position: 'absolute',
      width: '100%',
      top: 0,
      left: 0
    }}>
      <TomographyViewer />
    </div>
  );

  // Parametric Shape View
  const ParametricView = (
    <div style={{ 
      height: '100%',
      position: 'absolute',
      width: '100%',
      top: 0,
      left: 0
    }}>
      {mesh ? (
        <ThreeContext>
          <ReplicadMesh edges={mesh.edges} faces={mesh.faces} />
        </ThreeContext>
      ) : (
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          fontSize: "2em"
        }}>
          Loading...
        </div>
      )}
    </div>
  );

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <main style={{
          display: 'flex',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          overflow: 'hidden'
        }}>
          {/* Left Side */}
          <div style={{ 
            flex: 1, 
            borderRight: '1px solid #ccc',
            height: '100%',
            overflow: 'hidden'
          }}>
            <TabPanel
              tabs={[
                { label: 'Parameters', content: ParametersPanel },
                { label: 'Nodes', content: NodeSystemPanel }
              ]}
              activeTab={leftActiveTab}
              onTabChange={setLeftActiveTab}
            />
          </div>

          {/* Right Side */}
          <div style={{ 
            flex: 1,
            height: '100%',
            overflow: 'hidden'
          }}>
            <TabPanel
              tabs={[
                { label: 'Parametric Shape', content: ParametricView },
                { label: '3D Tomography View', content: TomographyView }
              ]}
              activeTab={rightActiveTab}
              onTabChange={setRightActiveTab}
            />
          </div>
        </main>
      </Box>
    </div>
  );
}
