import React, { useState, useEffect, useCallback } from "react";
import FileSaver from "file-saver";
import { wrap } from "comlink";
import { ThemeProvider, Box, Tabs, Tab, Slider } from '@mui/material';
import { theme } from './theme';

import ThreeContext from "./ThreeContext.jsx";
import ReplicadMesh from "./ReplicadMesh.jsx";
import TabPanel from "./components/TabPanel.jsx";
import { generateGyroid } from "./services/gyroidService.js";
import TomographyViewer from './components/TomographyViewer';

import cadWorker from "./worker.js?worker";
const cad = wrap(new cadWorker());

export default function ReplicadApp() {
  const [size, setSize] = useState(5);
  const [mainTab, setMainTab] = useState(0);
  const [prostheticTab, setProstheticTab] = useState(0);
  const [tomographyTab, setTomographyTab] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [isDragging, setIsDragging] = useState(false);

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

  // Handle sidebar resize
  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
    const startX = e.clientX;
    const startWidth = sidebarWidth;
    
    const handleMove = (e) => {
      const delta = e.clientX - startX;
      const newWidth = Math.max(250, Math.min(startWidth + delta, window.innerWidth * 0.5));
      setSidebarWidth(newWidth);
      document.body.style.cursor = 'col-resize';
    };
    
    const handleUp = () => {
      setIsDragging(false);
      document.body.style.cursor = '';
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
    };
    
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleUp);
  }, [sidebarWidth]);

  // Remove the separate move and up handlers since they're now defined inside handleMouseDown
  const handleMouseMove = null;
  const handleMouseUp = null;

  // Remove the cleanup effect since we don't have separate handlers anymore
  useEffect(() => {
    return () => {
      if (isDragging) {
        document.body.style.cursor = '';
      }
    };
  }, [isDragging]);

  // Parameters Panel Content
  const ParametersPanel = (
    <Box sx={{
      p: 3,
      display: 'flex',
      flexDirection: 'column',
      gap: 3,
      height: '100%',
      overflow: 'auto',
      bgcolor: 'background.default'
    }}>
      <Box>
        <h3 style={{ marginBottom: '1rem', color: theme.palette.text.primary }}>Base Shape Parameters</h3>
        <label htmlFor="thicknessInput" style={{ 
          display: 'block', 
          marginBottom: '0.5rem',
          color: theme.palette.text.primary 
        }}>
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
            padding: '0.75rem',
            borderRadius: '4px',
            border: `1px solid ${theme.palette.divider}`,
            backgroundColor: theme.palette.background.paper,
            color: theme.palette.text.primary
          }}
        />
      </Box>

      <Box>
        <h3 style={{ marginBottom: '1rem', color: theme.palette.text.primary }}>Gyroid Parameters</h3>
        
        {error && (
          <Box sx={{
            p: 2,
            mb: 2,
            bgcolor: '#f8d7da',
            color: '#721c24',
            borderRadius: 1,
            border: '1px solid #f5c6cb'
          }}>
            {error}
          </Box>
        )}
        
        <Box sx={{ mb: 4 }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold',
            color: theme.palette.text.primary
          }}>
            Pattern Type
          </label>
          <select
            value={gyroidMode}
            onChange={(e) => setGyroidMode(e.target.value)}
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: '4px',
              border: `1px solid ${theme.palette.divider}`,
              backgroundColor: theme.palette.background.paper,
              color: theme.palette.text.primary
            }}
          >
            <option value="surface">Surface Lattice - Pattern follows the object's surface</option>
            <option value="volume">Volume Lattice - Pattern fills the entire interior</option>
          </select>
        </Box>

        {gyroidMode === 'surface' && (
          <Box sx={{
            p: 2,
            mb: 2,
            bgcolor: '#cce5ff',
            color: '#004085',
            borderRadius: 1,
            border: '1px solid #b8daff',
            fontSize: '0.9em'
          }}>
            Surface mode requires higher resolution for good quality. Resolution will be automatically adjusted to minimum 80.
          </Box>
        )}
        
        <Box sx={{ mb: 4 }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold',
            color: theme.palette.text.primary
          }}>
            Pattern Scale: {period}
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: theme.palette.text.secondary,
            marginBottom: '0.5rem' 
          }}>
            Controls the size and number of gyroid cells. Higher values create more cells in a denser pattern, lower values create fewer, larger cells.
          </p>
          <Box sx={{
            '& .MuiSlider-root': {
              color: 'primary.main',
            }
          }}>
            <Slider
              value={period}
              onChange={(_, value) => setPeriod(value)}
              min={0.5}
              max={10.0}
              step={0.1}
              valueLabelDisplay="auto"
            />
          </Box>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8em',
            color: 'text.secondary'
          }}>
            <span>Few large cells</span>
            <span>Many small cells</span>
          </Box>
        </Box>

        <Box sx={{ mb: 4 }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold',
            color: theme.palette.text.primary
          }}>
            Pattern Thickness: {thickness}
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: theme.palette.text.secondary,
            marginBottom: '0.5rem' 
          }}>
            {gyroidMode === 'surface' 
              ? "Controls the thickness of the lattice walls on the surface. Lower values create more delicate patterns, higher values make them more robust."
              : "Controls the thickness of the internal lattice structure. Lower values create more void space, higher values create a denser structure."}
          </p>
          <Box sx={{
            '& .MuiSlider-root': {
              color: 'primary.main',
            }
          }}>
            <Slider
              value={thickness}
              onChange={(_, value) => setThickness(value)}
              min={0.05}
              max={0.5}
              step={0.05}
              valueLabelDisplay="auto"
            />
          </Box>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8em',
            color: 'text.secondary'
          }}>
            <span>Thin walls</span>
            <span>Thick walls</span>
          </Box>
        </Box>

        <Box sx={{ mb: 4 }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem',
            fontWeight: 'bold',
            color: theme.palette.text.primary
          }}>
            Resolution: {resolution}
          </label>
          <p style={{ 
            fontSize: '0.9em', 
            color: theme.palette.text.secondary,
            marginBottom: '0.5rem' 
          }}>
            Controls the quality of the generated pattern. Higher values create smoother, more detailed patterns but require more processing time.
          </p>
          <Box sx={{
            '& .MuiSlider-root': {
              color: 'primary.main',
            }
          }}>
            <Slider
              value={resolution}
              onChange={(_, value) => setResolution(value)}
              min={40}
              max={150}
              step={5}
              valueLabelDisplay="auto"
            />
          </Box>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8em',
            color: 'text.secondary'
          }}>
            <span>Faster, rougher</span>
            <span>Slower, smoother</span>
          </Box>
        </Box>
        
        {isProcessing && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ mb: 1, color: theme.palette.text.secondary }}>
              Generating gyroid pattern: {Math.round(progress * 100)}%
            </Box>
            <Box sx={{ 
              width: '100%', 
              height: '4px', 
              bgcolor: 'background.paper',
              borderRadius: '2px',
              overflow: 'hidden'
            }}>
              <Box sx={{
                width: `${progress * 100}%`,
                height: '100%',
                bgcolor: 'primary.main',
                transition: 'width 0.3s ease'
              }} />
            </Box>
          </Box>
        )}
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <button 
            onClick={downloadModel}
            style={{
              flex: 1,
              padding: '0.75rem 1rem',
              backgroundColor: theme.palette.primary.main,
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            Download Base STL
          </button>
          <button 
            onClick={generateGyroidModel}
            disabled={isProcessing}
            style={{
              flex: 1,
              padding: '0.75rem 1rem',
              backgroundColor: isProcessing ? theme.palette.action.disabled : theme.palette.secondary.main,
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              fontWeight: 500
            }}
          >
            {isProcessing ? `Generating (${Math.round(progress * 100)}%)` : 'Generate Gyroid'}
          </button>
        </Box>
      </Box>
    </Box>
  );

  // Node System Panel Content
  const NodeSystemPanel = (
    <Box sx={{
      p: 3,
      bgcolor: 'background.default',
      height: '100%',
      overflow: 'hidden',
      position: 'relative'
    }}>
      <Box sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'grid',
        bgcolor: theme.palette.background.dark,
        backgroundImage: `radial-gradient(#586e75 1px, transparent 1px)`,
        backgroundSize: '20px 20px',
        overflow: 'auto',
        '&::-webkit-scrollbar': {
          width: '8px',
          height: '8px',
        },
        '&::-webkit-scrollbar-track': {
          background: theme.palette.background.dark,
        },
        '&::-webkit-scrollbar-thumb': {
          background: theme.palette.secondary.main,
          borderRadius: '4px',
        },
        '&::-webkit-scrollbar-thumb:hover': {
          background: theme.palette.secondary.dark,
        },
      }}>
        {/* Node system content will be added here */}
      </Box>
    </Box>
  );

  // Parametric Shape View
  const ParametricView = (
    <Box sx={{ 
      height: '100%',
      position: 'absolute',
      width: '100%',
      top: 0,
      left: 0,
      bgcolor: 'background.paper'
    }}>
      {mesh ? (
        <ThreeContext>
          <ReplicadMesh edges={mesh.edges} faces={mesh.faces} />
        </ThreeContext>
      ) : (
        <Box sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          fontSize: "2em",
          color: "text.secondary"
        }}>
          Loading...
        </Box>
      )}
    </Box>
  );

  // Tomography View
  const TomographyView = (
    <Box sx={{ 
      height: '100%',
      position: 'absolute',
      width: '100%',
      top: 0,
      left: 0,
      bgcolor: 'background.paper'
    }}>
      <TomographyViewer />
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ 
        height: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        bgcolor: 'background.default',
        userSelect: isDragging ? 'none' : 'auto' // Prevent text selection while dragging
      }}>
        {/* Main Navigation */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={mainTab} 
            onChange={(_, value) => setMainTab(value)}
            sx={{
              '& .MuiTab-root': {
                fontSize: '1rem',
                textTransform: 'none',
                fontWeight: 500,
                py: 2
              }
            }}
          >
            <Tab label="Prosthetic Design" />
            <Tab label="Tomography Analysis" />
          </Tabs>
        </Box>

        {/* Prosthetic Design View */}
        {mainTab === 0 && (
          <Box sx={{ 
            flex: 1,
            display: 'flex',
            overflow: 'hidden',
            position: 'relative'
          }}>
            {/* Left Side - Design Tools */}
            <Box sx={{ 
              width: prostheticTab === 1 ? `${sidebarWidth}px` : '300px',
              minWidth: '250px',
              maxWidth: '50%',
              borderRight: 1,
              borderColor: 'divider',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              transition: isDragging ? 'none' : 'width 0.2s ease',
              position: 'relative',
              zIndex: 1
            }}>
              <TabPanel
                tabs={[
                  { label: 'Parameters', content: ParametersPanel },
                  { label: 'Node System', content: NodeSystemPanel }
                ]}
                activeTab={prostheticTab}
                onTabChange={setProstheticTab}
              />
            </Box>

            {/* Resize Handle */}
            {prostheticTab === 1 && (
              <Box
                sx={{
                  position: 'absolute',
                  left: `${sidebarWidth - 4}px`,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  zIndex: 2,
                  height: '80px', 
                  display: 'flex',
                  alignItems: 'center',
                  transition: isDragging ? 'none' : 'left 0.2s ease',
                  pointerEvents: 'none', // Make the container pass through events
                }}
              >
                <Box
                  sx={{
                    width: '8px', 
                    height: '100%',
                    bgcolor: isDragging ? 'primary.main' : 'divider',
                    borderRadius: '8px', 
                    cursor: 'col-resize',
                    position: 'relative',
                    pointerEvents: 'auto', // Re-enable pointer events for the handle
                    '&:hover': {
                      bgcolor: 'primary.main',
                    },
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      left: '-8px',
                      top: 0,
                      width: '24px', 
                      height: '100%',
                      cursor: 'col-resize',
                    }
                  }}
                  onMouseDown={handleMouseDown}
                >
                  <Box
                    sx={{
                      position: 'absolute',
                      left: '50%',
                      top: '50%',
                      transform: 'translate(-50%, -50%)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '12px', 
                      alignItems: 'center',
                      pointerEvents: 'none', // Prevent dots from interfering
                    }}
                  >
                    {[0, 1, 2].map((i) => (
                      <Box
                        key={i}
                        sx={{
                          width: '4px', 
                          height: '4px', 
                          borderRadius: '50%',
                          bgcolor: isDragging ? 'primary.light' : 'text.disabled',
                          transition: 'background-color 0.2s ease',
                        }}
                      />
                    ))}
                  </Box>
                </Box>
              </Box>
            )}

            {/* Right Side - 3D Preview */}
            <Box sx={{ 
              flex: 1,
              overflow: 'hidden',
              position: 'relative'
            }}>
              {ParametricView}
            </Box>
          </Box>
        )}

        {/* Tomography Analysis View */}
        {mainTab === 1 && (
          <Box sx={{ 
            flex: 1,
            display: 'flex',
            overflow: 'hidden'
          }}>
            {/* Left Side - Tomography Views */}
            <Box sx={{ 
              flex: 1,
              borderRight: 1,
              borderColor: 'divider',
              overflow: 'hidden'
            }}>
              <TomographyViewer />
            </Box>

            {/* Right Side - Data Extraction */}
            <Box sx={{ 
              width: '300px',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <TabPanel
                tabs={[
                  { 
                    label: 'Data Extraction', 
                    content: (
                      <Box sx={{ p: 3 }}>
                        <h3 style={{ 
                          marginBottom: '1rem',
                          color: theme.palette.text.primary 
                        }}>
                          Extract Data from 3D Model
                        </h3>
                        {/* Add data extraction tools here */}
                        <p style={{ color: theme.palette.text.secondary }}>
                          Data extraction tools will be implemented here...
                        </p>
                      </Box>
                    )
                  }
                ]}
                activeTab={0}
                onTabChange={() => {}}
              />
            </Box>
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
}
