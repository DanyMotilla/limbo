export async function generateGyroid(stlBlob, params) {
    const MAX_RETRIES = 3;
    const POLL_INTERVAL = 2000; // Poll every 2 seconds
    const MAX_POLL_TIME = 600000; // 10 minutes

    try {
        console.log('Starting gyroid generation with params: ', params);
        
        // Create form data
        const formData = new FormData();
        formData.append('stl_file', stlBlob, 'input.stl'); // Set filename explicitly
        formData.append('mode', params.mode);
        formData.append('period', params.period);
        formData.append('thickness', params.thickness);
        formData.append('resolution', params.resolution);
        formData.append('fileSize', stlBlob.size);

        // Start the job
        console.log('Starting job...');
        const response = await fetch('http://localhost:8000/generate_gyroid', {
            method: 'POST',
            body: formData,
            mode: 'cors',
            credentials: 'omit'
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

        // Poll for job completion
        const startTime = Date.now();
        while (true) {
            if (Date.now() - startTime > MAX_POLL_TIME) {
                throw new Error('Job timed out after 10 minutes');
            }

            // Get job status
            const statusResponse = await fetch(`http://localhost:8000/job/${job_id}`, {
                mode: 'cors',
                credentials: 'omit'
            });

            if (!statusResponse.ok) {
                throw new Error('Failed to get job status');
            }

            const jobStatus = await statusResponse.json();
            console.log('Job status:', jobStatus);

            if (jobStatus.status === 'completed') {
                // Get the result
                const resultResponse = await fetch(`http://localhost:8000/job/${job_id}/result`, {
                    mode: 'cors',
                    credentials: 'omit'
                });

                if (!resultResponse.ok) {
                    throw new Error('Failed to get job result');
                }

                console.log('Successfully retrieved result');
                return await resultResponse.blob();
            }

            if (jobStatus.status === 'failed') {
                throw new Error(jobStatus.error || 'Job failed');
            }

            // Wait before polling again
            await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
        }
    } catch (error) {
        console.error('Error:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        throw error;
    }
}
