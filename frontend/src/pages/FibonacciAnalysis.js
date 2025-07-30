import React, { useState } from 'react';
import {
  Typography,
  Box,
  Button,
  TextField,
  Paper,
  Alert,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { Calculate, TrendingUp } from '@mui/icons-material';
import api from '../services/api';

function FibonacciAnalysis() {
  const [high, setHigh] = useState(150);
  const [low, setLow] = useState(60);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runAnalysis = async () => {
    if (!high || !low || high <= low) {
      setError('Please enter valid high and low values (high must be greater than low)');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post('/analysis/fibonacci', {
        high: parseFloat(high),
        low: parseFloat(low)
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
        Fibonacci Retracement Analysis
      </Typography>
      
      <Typography variant="body1" color="textSecondary" paragraph>
        Calculate Fibonacci retracement levels and sub-levels based on high and low price values.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="High Price"
              type="number"
              value={high}
              onChange={(e) => setHigh(e.target.value)}
              inputProps={{ step: 0.01 }}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Low Price"
              type="number"
              value={low}
              onChange={(e) => setLow(e.target.value)}
              inputProps={{ step: 0.01 }}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <Button
              variant="contained"
              onClick={runAnalysis}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <Calculate />}
              fullWidth
            >
              {loading ? 'Calculating...' : 'Calculate Levels'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Grid container spacing={3}>
          {/* Main Levels */}
          <Grid item xs={12} lg={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Main Fibonacci Levels
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Level</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">Units to Buy</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.main_levels.map((level, index) => (
                        <TableRow key={index}>
                          <TableCell>{index}</TableCell>
                          <TableCell align="right">
                            {level[Object.keys(level)[0]]?.toFixed(4)}
                          </TableCell>
                          <TableCell align="right">
                            {level.units_to_buy || 0}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Price Levels Breakdown */}
          <Grid item xs={12} lg={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Price Levels Breakdown
                </Typography>
                
                {/* High Levels */}
                <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                  High Levels (Extension)
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  {result.price_levels.high.map((price, index) => (
                    <Box
                      key={index}
                      sx={{
                        px: 1,
                        py: 0.5,
                        bgcolor: 'success.light',
                        color: 'success.contrastText',
                        borderRadius: 1,
                        fontSize: '0.875rem'
                      }}
                    >
                      {price.toFixed(2)}
                    </Box>
                  ))}
                </Box>

                {/* Standard Levels */}
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Standard Levels (Retracement)
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  {result.price_levels.standard.map((price, index) => (
                    <Box
                      key={index}
                      sx={{
                        px: 1,
                        py: 0.5,
                        bgcolor: 'primary.light',
                        color: 'primary.contrastText',
                        borderRadius: 1,
                        fontSize: '0.875rem'
                      }}
                    >
                      {price.toFixed(2)}
                    </Box>
                  ))}
                </Box>

                {/* Low Levels */}
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Low Levels (Support)
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {result.price_levels.low.map((price, index) => (
                    <Box
                      key={index}
                      sx={{
                        px: 1,
                        py: 0.5,
                        bgcolor: 'warning.light',
                        color: 'warning.contrastText',
                        borderRadius: 1,
                        fontSize: '0.875rem'
                      }}
                    >
                      {price.toFixed(2)}
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Sub Levels */}
          {Object.keys(result.sub_levels).length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Sub-Levels
                  </Typography>
                  <Box sx={{ overflow: 'auto' }}>
                    <pre style={{ 
                      fontSize: '0.875rem',
                      backgroundColor: '#f5f5f5',
                      padding: '16px',
                      borderRadius: '4px',
                      margin: 0
                    }}>
                      {JSON.stringify(result.sub_levels, null, 2)}
                    </pre>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          About Fibonacci Retracement
        </Typography>
        <Typography variant="body2" paragraph>
          Fibonacci retracement levels are horizontal lines that indicate areas of support or resistance 
          at the key Fibonacci levels before the price continues in the original direction.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Key Levels:</strong>
        </Typography>
        <ul>
          <li><strong>23.6%:</strong> Shallow retracement, strong trend</li>
          <li><strong>38.2%:</strong> Common retracement level</li>
          <li><strong>50%:</strong> Not a Fibonacci number but widely watched</li>
          <li><strong>61.8%:</strong> Golden ratio, strong support/resistance</li>
          <li><strong>78.6%:</strong> Deep retracement, trend weakness</li>
        </ul>
      </Paper>
    </Box>
  );
}

export default FibonacciAnalysis;
