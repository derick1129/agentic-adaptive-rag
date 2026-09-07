// API Communication Layer for Adaptive Agentic RAG Dashboard

class RAGAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.session = {
            tenantId: 'default',
            subjectId: 'anonymous',
            acl: [],
        };
    }

    setSession(session) {
        this.session = {
            tenantId: (session.tenantId || 'default').trim() || 'default',
            subjectId: (session.subjectId || 'anonymous').trim() || 'anonymous',
            acl: Array.isArray(session.acl)
                ? session.acl.map(a => String(a).trim()).filter(a => a)
                : String(session.acl || '')
                    .split(',')
                    .map(a => a.trim())
                    .filter(a => a),
        };
    }

    _sessionHeaders() {
        const headers = {
            'X-Tenant-Id': this.session.tenantId,
            'X-Subject-Id': this.session.subjectId,
        };
        if (this.session.acl.length > 0) {
            headers['X-ACL'] = this.session.acl.join(',');
        }
        return headers;
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Unable to connect to server: ${error.message}`);
        }
    }

    async _readError(response) {
        try {
            const data = await response.json();
            if (data && data.detail) {
                return String(data.detail);
            }
            return `HTTP ${response.status}: ${response.statusText}`;
        } catch (e) {
            try {
                const text = await response.text();
                const trimmed = (text || '').trim();
                return trimmed
                    ? `HTTP ${response.status}: ${trimmed.slice(0, 300)}`
                    : `HTTP ${response.status}: ${response.statusText}`;
            } catch (e2) {
                return `HTTP ${response.status}: ${response.statusText}`;
            }
        }
    }

    async ingestDocument(formData) {
        try {
            const response = await fetch(`${this.baseUrl}/v1/ingestion`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._sessionHeaders(),
                },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                const errorData = await this._readError(response);
                throw new Error(errorData);
            }

            return await response.json();
        } catch (error) {
            throw error;
        }
    }

    async uploadDocument(file, metadata) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', metadata || '{}');

        try {
            const response = await fetch(`${this.baseUrl}/v1/ingestion/upload`, {
                method: 'POST',
                headers: this._sessionHeaders(),
                body: formData,
            });

            if (!response.ok) {
                const errorData = await this._readError(response);
                throw new Error(errorData);
            }

            return await response.json();
        } catch (error) {
            throw error;
        }
    }

    async getJobStatus(jobId) {
        try {
            const response = await fetch(`${this.baseUrl}/v1/ingestion/${encodeURIComponent(jobId)}`, {
                headers: this._sessionHeaders(),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            throw error;
        }
    }

    async queryDocuments(formData) {
        try {
            const response = await fetch(`${this.baseUrl}/v1/queries`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._sessionHeaders(),
                },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                const errorData = await this._readError(response);
                throw new Error(errorData);
            }

            return await response.json();
        } catch (error) {
            throw error;
        }
    }
}

// Initialize API instance
const api = new RAGAPI();

// Export for use in other modules
window.RAGAPI = RAGAPI;
window.api = api;
