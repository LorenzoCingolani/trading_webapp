import React, { useState } from 'react';
import {
  Typography,
  Box,
  Button,
  Paper,
  Alert,
  CircularProgress
} from '@mui/material';
import { PieChart, PlayArrow } from '@mui/icons-material';
import api from '../services/api';

function PDM() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runPDM = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post('/pdm');
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'PDM analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        <PieChart sx={{ mr: 1, verticalAlign: 'middle' }} />
        PDM Analysis
      </Typography>
      
      <Typography variant="body1" color="textSecondary" paragraph>
        Portfolio Diversification Model (PDM) analysis for optimal asset allocation and risk management.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Button
          variant="contained"
          onClick={runPDM}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
          size="large"
        >
          {loading ? 'Running PDM Analysis...' : 'Run PDM Analysis'}
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
          About PDM
        </Typography>
        <Typography variant="body2" paragraph>
          Portfolio Diversification Model (PDM) helps in:
        </Typography>
        <ul>
          <li>Optimal asset allocation across instruments</li>
          <li>Risk diversification strategies</li>
          <li>Correlation analysis between assets</li>
          <li>Portfolio optimization techniques</li>
          <li>Risk-adjusted return calculations</li>
        </ul>
        <Typography variant="body2">
          The analysis generates portfolio weights and risk metrics to guide investment decisions.
        </Typography>
      </Paper>
    </Box>
  );
}

export default PDM;
