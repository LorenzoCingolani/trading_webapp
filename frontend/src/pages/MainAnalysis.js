import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  OutlinedInput,
  Paper,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { ExpandMore, PlayArrow } from '@mui/icons-material';
import api from '../services/api';

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

function MainAnalysis() {
  const [instruments, setInstruments] = useState([]);
  const [selectedInstruments, setSelectedInstruments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInstruments = async () => {
      try {
        const response = await api.get('/instruments');
        setInstruments(response.data);
      } catch (err) {
        setError('Failed to fetch instruments');
      }
    };

    fetchInstruments();
  }, []);

  const handleInstrumentChange = (event) => {
    const value = event.target.value;
    setSelectedInstruments(typeof value === 'string' ? value.split(',') : value);
  };

  const runAnalysis = async () => {
    if (selectedInstruments.length === 0) {
      setError('Please select at least one instrument');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post('/analysis/main', {
        instruments: selectedInstruments
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Main Analysis
      </Typography>
      
      <Typography variant="body1" color="textSecondary" paragraph>
        Run comprehensive trading strategy analysis including EWMA, Breakout, Carry, and Stochastic models on selected instruments.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Select Instruments</InputLabel>
          <Select
            multiple
            value={selectedInstruments}
            onChange={handleInstrumentChange}
            input={<OutlinedInput label="Select Instruments" />}
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value) => (
                  <Chip key={value} label={value} />
                ))}
              </Box>
            )}
            MenuProps={MenuProps}
          >
            {instruments.map((instrument) => (
              <MenuItem key={instrument.name} value={instrument.name}>
                {instrument.name} ({instrument.currency})
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button
          variant="contained"
          onClick={runAnalysis}
          disabled={loading || selectedInstruments.length === 0}
          startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
          fullWidth
        >
          {loading ? 'Running Analysis...' : 'Run Main Analysis'}
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
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Analysis Results</Typography>
              </AccordionSummary>
              <AccordionDetails>
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
              </AccordionDetails>
            </Accordion>
          )}
        </Paper>
      )}

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Analysis Information
        </Typography>
        <Typography variant="body2" paragraph>
          The main analysis runs the following models:
        </Typography>
        <ul>
          <li><strong>EWMA (Exponential Weighted Moving Average):</strong> Parameters [2, 4, 8, 16]</li>
          <li><strong>Breakout Models:</strong> Various volatility and lookback parameters</li>
          <li><strong>Carry Strategy:</strong> Interest rate differential analysis</li>
          <li><strong>Stochastic Models:</strong> Momentum-based trading signals</li>
        </ul>
        <Typography variant="body2">
          Results are saved to the DATA/output_instruments directory and plots to DATA/output_plots.
        </Typography>
      </Paper>
    </Box>
  );
}

export default MainAnalysis;
