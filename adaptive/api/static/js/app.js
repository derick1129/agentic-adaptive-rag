// Main Application Logic for Adaptive Agentic RAG Dashboard

document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI elements
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const ingestionForm = document.getElementById('ingestion-form');
    const ingestionResult = document.getElementById('ingestion-result');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadResult = document.getElementById('upload-result');
    const queryForm = document.getElementById('query-form');
    const queryResult = document.getElementById('query-result');
    const generationOutput = document.getElementById('generation-output');
    const currentTraceId = document.getElementById('current-trace-id');
    const sessionForm = document.getElementById('session-form');
    const sessionResult = document.getElementById('session-result');

    // Restore saved session on load
    function loadSession() {
        try {
            const saved = JSON.parse(localStorage.getItem('ragSession') || 'null');
            if (saved) {
                api.setSession(saved);
                document.getElementById('tenant-id').value = api.session.tenantId;
                document.getElementById('subject-id').value = api.session.subjectId;
                document.getElementById('session-acl').value = api.session.acl.join(', ');
                updateSessionDisplay();
            }
        } catch (e) {
            // ignore malformed saved session
        }
    }

    function updateSessionDisplay() {
        const acl = api.session.acl.length > 0 ? api.session.acl.join(', ') : '(none)';
        sessionResult.className = 'result-area success';
        sessionResult.innerHTML = `
            <p>Connected as <strong>tenant=</strong>${api.session.tenantId},
            <strong>subject=</strong>${api.session.subjectId},
            <strong>ACL=</strong>${acl}</p>
        `;
    }

    // Handle session (connect) form submission
    sessionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const session = {
            tenantId: document.getElementById('tenant-id').value,
            subjectId: document.getElementById('subject-id').value,
            acl: document.getElementById('session-acl').value,
        };
        api.setSession(session);
        localStorage.setItem('ragSession', JSON.stringify(api.session));
        updateSessionDisplay();
    });

    // Poll ingestion job status until terminal state
    async function pollJobStatus(jobId, resultElement, maxAttempts = 20) {
        let attempts = 0;
        const terminal = (status) => {
            const s = String(status).toUpperCase();
            return s === 'ACTIVE' || s === 'FAILED' || s === 'COMPLETED' || s === 'DONE';
        };
        const tick = async () => {
            attempts += 1;
            try {
                const job = await api.getJobStatus(jobId);
                if (!terminal(job.status) && attempts < maxAttempts) {
                    setTimeout(tick, 3000);
                } else {
                    showResult(resultElement, terminal(job.status) && job.status.toLowerCase() !== 'failed' ? 'success' : 'error',
                        `Job <strong>${jobId}</strong> status: <strong>${job.status}</strong>` +
                        (job.document_id ? `<br>Document ID: ${job.document_id}` : '') +
                        (job.error_message ? `<br>Error: ${job.error_message}` : '')
                    );
                }
                return job;
            } catch (error) {
                showResult(resultElement, 'error', `Job status check failed: ${error.message}`);
                return null;
            }
        };
        return tick();
    }

    // Handle file upload
    uploadBtn.addEventListener('click', async () => {
        const fileInput = document.getElementById('file-upload');
        const file = fileInput.files[0];
        if (!file) {
            showResult(uploadResult, 'error', 'Please select a file to upload');
            return;
        }

        let metadata = {};
        const metadataInput = document.getElementById('metadata').value.trim();
        if (metadataInput) {
            try {
                metadata = JSON.parse(metadataInput);
            } catch (parseError) {
                showResult(uploadResult, 'error', `Invalid JSON in metadata: ${parseError.message}`);
                return;
            }
        }

        showResult(uploadResult, 'loading', `Uploading ${file.name}...`);

        try {
            const response = await api.uploadDocument(file, JSON.stringify(metadata));
            showResult(uploadResult, 'success',
                `Upload successful!<br><strong>Job ID:</strong> ${response.job_id}<br><strong>Status:</strong> ${response.status}`
            );
            fileInput.value = '';
            document.getElementById('metadata').value = '{}';
            pollJobStatus(response.job_id, uploadResult);
        } catch (error) {
            showResult(uploadResult, 'error', `Upload failed: ${error.message}`);
        }
    });

    // Check system health on load
    async function checkSystemHealth() {
        statusIndicator.className = 'status-indicator';
        statusText.textContent = 'Checking system status...';
        
        try {
            const health = await api.healthCheck();
            statusIndicator.className = 'status-indicator healthy';
            statusText.textContent = 'System is healthy';
        } catch (error) {
            statusIndicator.className = 'status-indicator unhealthy';
            statusText.textContent = `System unavailable: ${error.message}`;
        }
    }

    // Handle ingestion form submission
    ingestionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get form data
        const filename = document.getElementById('filename').value.trim();
        const content = document.getElementById('content').value.trim();
        const sourceType = document.getElementById('source-type').value;
        const metadataInput = document.getElementById('metadata').value.trim();
        
        let metadata = {};
        if (metadataInput) {
            try {
                metadata = JSON.parse(metadataInput);
            } catch (parseError) {
                showResult(ingestionResult, 'error', `Invalid JSON in metadata: ${parseError.message}`);
                return;
            }
        }
        
        // Validate required fields
        if (!filename || !content) {
            showResult(ingestionResult, 'error', 'Filename and content are required');
            return;
        }
        
        // Show loading state
        showResult(ingestionResult, 'loading', 'Submitting document for ingestion...');
        
        try {
            const response = await api.ingestDocument({
                filename,
                content,
                source_type: sourceType,
                metadata
            });
            
            showResult(ingestionResult, 'success', 
                `Ingestion successful!<br><strong>Job ID:</strong> ${response.job_id}<br><strong>Status:</strong> ${response.status}<br><strong>Message:</strong> ${response.message}`
            );
            pollJobStatus(response.job_id, ingestionResult);
            
            // Clear form after successful submission
            ingestionForm.reset();
            document.getElementById('metadata').value = '{}';
            
        } catch (error) {
            showResult(ingestionResult, 'error', `Ingestion failed: ${error.message}`);
        }
    });

    // Handle query form submission
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get form data
        const queryText = document.getElementById('query-text').value.trim();
        const queryLimit = parseInt(document.getElementById('query-limit').value) || 10;
        const documentIdsInput = document.getElementById('document-ids').value.trim();
        
        let documentIds = null;
        if (documentIdsInput) {
            documentIds = documentIdsInput.split(',').map(id => id.trim()).filter(id => id);
            if (documentIds.length === 0) {
                documentIds = null;
            }
        }
        
        // Validate required fields
        if (!queryText) {
            showResult(queryResult, 'error', 'Search query is required');
            return;
        }
        
        // Show loading state
        showResult(queryResult, 'loading', 'Searching documents...');
        
        try {
            const response = await api.queryDocuments({
                query: queryText,
                limit: queryLimit,
                document_ids: documentIds
            });
            
            // Update trace ID
            currentTraceId.textContent = response.trace_id || 'N/A';
            
            // Display results
            if (response.results && response.results.length > 0) {
                let resultsHTML = `<h3>Search Results (${response.results.length} found)</h3>`;
                response.results.forEach((result, index) => {
                    resultsHTML += `
                        <div class="result-item chunk">
                            <div class="result-header">
                                <span class="result-id">Result #${index + 1}</span>
                                <span class="result-score">Score: ${result.score.toFixed(4)}</span>
                            </div>
                            <div class="result-text">${result.text}</div>
                            ${result.metadata && Object.keys(result.metadata).length > 0 ? 
                                `<div class="result-metadata"><strong>Metadata:</strong> ${JSON.stringify(result.metadata)}</div>` : ''}
                        </div>
                    `;
                });
                
                showResult(queryResult, 'success', resultsHTML);
            } else if (response.cache_status === 'hit') {
                showResult(queryResult, 'success', '<h3>Answer Served from Semantic Cache</h3><p>⚡ Sub-millisecond response served directly from Redis cache.</p>');
            } else {
                showResult(queryResult, 'success', '<h3>No Results Found</h3><p>No documents matched your search query.</p>');
            }
            
            // Update generation output / answer
            let genHTML = '<h3>Generated Answer</h3>';
            if (response.answer) {
                genHTML += `
                    <div class="result-item" style="border-left-color: #2ecc71; background-color: #f8fcf9; padding: 12px; margin-bottom: 15px;">
                        <div style="font-weight: 600; margin-bottom: 6px; color: #27ae60;">
                            Synthesized Response ${response.cache_status === 'hit' ? '<span style="font-size: 0.8em; background: #e8f8f5; color: #16a085; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">⚡ Semantic Cache Hit</span>' : ''}
                        </div>
                        <div class="result-text" style="white-space: pre-wrap; font-size: 1.05em; line-height: 1.5;">${response.answer}</div>
                    </div>
                `;
            }
            if (response.results && response.results.length > 0) {
                const contextText = response.results.map((r, i) => `[${i + 1}] ${r.text}`).join('\n\n');
                genHTML += `
                    <h4 style="margin-top: 12px; color: #555;">Retrieved Context Passages</h4>
                    <div class="result-item" style="border-left-color: #9b59b6;">
                        <div class="result-text" style="white-space: pre-wrap;">${contextText}</div>
                    </div>
                `;
            } else if (!response.answer) {
                genHTML += '<p>No relevant documents found for generation context.</p>';
            }
            generationOutput.innerHTML = genHTML;
            
        } catch (error) {
            showResult(queryResult, 'error', `Search failed: ${error.message}`);
        }
    });

    // Helper function to show results in UI areas
    function showResult(element, type, message) {
        element.className = `result-area ${type}`;
        element.innerHTML = message;
        
        // Auto-clear success/error messages after delay (optional)
        if (type === 'success' || type === 'error') {
            setTimeout(() => {
                if (element.className.includes('success') || element.className.includes('error')) {
                    element.className = 'result-area';
                    element.innerHTML = '';
                }
            }, 5000);
        }
    }

    // Initial health check
    loadSession();
    checkSystemHealth();
    
    // Periodic health checks (every 30 seconds)
    setInterval(checkSystemHealth, 30000);
});
