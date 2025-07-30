import React, { useState } from 'react';
import {
  Typography,
  Box,
  Button,
  Paper,
  Alert,
  CircularProgress
} from '@mui/material';
import { TrendingUp, PlayArrow } from '@mui/icons-material';
import api from '../services/api';

function SharpeRatio() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runSharpeAnalysis = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post('/sharpe');
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Sharpe ratio analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
        Sharpe Ratio Analysis
      </Typography>
      
      <Typography variant="body1" color="textSecondary" paragraph>
        Calculate Sharpe ratios to evaluate risk-adjusted performance of trading strategies.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Button
          variant="contained"
          onClick={runSharpeAnalysis}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
          size="large"
        >
          {loading ? 'Calculating Sharpe Ratios...' : 'Calculate Sharpe Ratios'}
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
          Understanding Sharpe Ratio
        </Typography>
        <Typography variant="body2" paragraph>
          The Sharpe ratio measures the performance of an investment compared to a risk-free asset, 
          after adjusting for its risk. It is calculated as:
        </Typography>
        <Box sx={{ 
          p: 2, 
          bgcolor: '#f5f5f5', 
          borderRadius: 1, 
          fontFamily: 'monospace',
          textAlign: 'center',
          mb: 2
        }}>
          Sharpe Ratio = (Return of Portfolio - Risk-free Rate) / Standard Deviation of Portfolio
        </Box>
        <Typography variant="body2" paragraph>
          <strong>Interpretation:</strong>
        </Typography>
        <ul>
          <li><strong>&gt; 1.0:</strong> Excellent risk-adjusted performance</li>
          <li><strong>0.5 - 1.0:</strong> Good risk-adjusted performance</li>
          <li><strong>0 - 0.5:</strong> Adequate risk-adjusted performance</li>
          <li><strong>&lt; 0:</strong> Poor performance, higher risk than return</li>
        </ul>
      </Paper>
    </Box>
  );
}

export default SharpeRatio;
