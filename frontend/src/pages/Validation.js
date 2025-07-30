import React, { useState } from 'react';
import {
  Typography,
  Box,
  Button,
  Paper,
  Alert,
  CircularProgress
} from '@mui/material';
import { Verified, PlayArrow } from '@mui/icons-material';
import api from '../services/api';

function Validation() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runValidation = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post('/validation');
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Validation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        <Verified sx={{ mr: 1, verticalAlign: 'middle' }} />
        Validation Analysis
      </Typography>
      
      <Typography variant="body1" color="textSecondary" paragraph>
        Validate the trading strategies and models to ensure accuracy and reliability.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Button
          variant="contained"
          onClick={runValidation}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
          size="large"
        >
          {loading ? 'Running Validation...' : 'Run Validation'}
        </Button>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Paper sx={{ p: 3 }}>
          <Alert severity="success" sx={{ mb: 2 }}>
            {result.message}
          </Alert>
          
          {result.results && (
            <Box component="pre" sx={{ 
              whiteSpace: 'pre-wrap', 
              fontFamily: 'monospace',
              backgroundColor: '#f5f5f5',
              p: 2,
              borderRadius: 1,
              overflow: 'auto'
            }}>
              {typeof result.results === 'string' 
                ? result.results 
                : JSON.stringify(result.results, null, 2)}
            </Box>
          )}
        </Paper>
      )}

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Validation Process
        </Typography>
        <Typography variant="body2" paragraph>
          The validation process includes:
        </Typography>
        <ul>
          <li>Strategy performance verification</li>
          <li>Data integrity checks</li>
          <li>Model accuracy assessment</li>
          <li>Risk parameter validation</li>
          <li>Output consistency verification</li>
        </ul>
      </Paper>
    </Box>
  );
}

export default Validation;
