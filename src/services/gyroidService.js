export async function generateGyroid(stlBlob, params) {
    const MAX_RETRIES = 3;
    const POLL_INTERVAL = 1000; // Poll every second for more responsive UI
    const MAX_POLL_TIME = 600000; // 10 minutes
    const API_BASE_URL = 'http://localhost:8000';

    try {
        console.log('Starting gyroid generation with params: ', params);
        
        // Validate and adjust parameters
        const adjustedParams = {
            ...params,
            resolution: params.mode === 'surface' ? Math.max(80, params.resolution) : params.resolution
        };
        
        // Create form data
        const formData = new FormData();
        formData.append('stl_file', stlBlob, 'input.stl');
        formData.append('mode', adjustedParams.mode);
        formData.append('period', adjustedParams.period);
        formData.append('thickness', adjustedParams.thickness);
        formData.append('resolution', adjustedParams.resolution);

        // Start the job
        console.log('Starting job with adjusted params:', adjustedParams);
        const response = await fetch(`${API_BASE_URL}/generate_gyroid`, {
            method: 'POST',
            body: formData,
            mode: 'cors',
            credentials: 'include',  
            headers: {
                'Accept': 'application/json',
            }
        });

        if (!response.ok) {
            let errorMessage = 'Failed to start gyroid generation';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
            } catch (e) {
                console.error('Error parsing error response:', e);
            }
            throw new Error(errorMessage);
        }

        // Get job ID
        const { job_id } = await response.json();
        console.log('Job started with ID:', job_id);

        let retryCount = 0;
        let lastProgress = -1;
        let lastUpdate = Date.now();

        // Poll for job completion
        const startTime = Date.now();
        while (true) {
            if (Date.now() - startTime > MAX_POLL_TIME) {
                throw new Error('Job timed out after 10 minutes');
            }

            try {
                // Get job status
                const statusResponse = await fetch(`${API_BASE_URL}/job/${job_id}`, {
                    mode: 'cors',
                    credentials: 'include',  
                    headers: {
                        'Accept': 'application/json',
                    }
                });

                if (!statusResponse.ok) {
                    throw new Error('Failed to get job status');
                }

                const jobStatus = await statusResponse.json();
                
                // Check for progress updates
                if (jobStatus.progress !== lastProgress) {
                    lastProgress = jobStatus.progress;
                    lastUpdate = Date.now();
                    console.log('Progress:', Math.round(jobStatus.progress * 100) + '%');
                    // Call progress callback if provided
                    if (params.onProgress) {
                        params.onProgress(jobStatus.progress);
                    }
                } else if (Date.now() - lastUpdate > 30000) { // No progress for 30 seconds
                    throw new Error('Job appears to be stuck - no progress for 30 seconds');
                }

                if (jobStatus.status === 'completed') {
                    // Get the result
                    const resultResponse = await fetch(`${API_BASE_URL}/job/${job_id}/result`, {
                        mode: 'cors',
                        credentials: 'include',  
                        headers: {
                            'Accept': 'application/octet-stream',
                        }
                    });

                    if (!resultResponse.ok) {
                        const errorData = await resultResponse.json();
                        throw new Error(errorData.detail || 'Failed to get job result');
                    }

                    console.log('Successfully retrieved result');
                    return await resultResponse.blob();
                }

                if (jobStatus.status === 'failed') {
                    throw new Error(jobStatus.error || 'Job failed');
                }

                // Wait before polling again
                await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
                
            } catch (error) {
                retryCount++;
                console.warn(`Attempt ${retryCount} failed:`, error);
                
                if (retryCount >= MAX_RETRIES) {
                    throw new Error(`Failed after ${MAX_RETRIES} retries: ${error.message}`);
                }
                
                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL * 2));
            }
        }
    } catch (error) {
        console.error('Error in generateGyroid:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        throw error;
    }
}
