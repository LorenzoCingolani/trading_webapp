import React, { useState, useEffect } from 'react';
import {
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  List,
  ListItem,
  ListItemText,
  CircularProgress
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  Storage,
  Timeline
} from '@mui/icons-material';
import api from '../services/api';

function Dashboard() {
  const [instruments, setInstruments] = useState([]);
  const [outputFiles, setOutputFiles] = useState({ output_files: [], plots: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [instrumentsResponse, outputResponse] = await Promise.all([
          api.get('/instruments'),
          api.get('/output-files')
        ]);
        setInstruments(instrumentsResponse.data);
        setOutputFiles(outputResponse.data);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Trading Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Storage color="primary" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6">{instruments.length}</Typography>
                  <Typography color="textSecondary">Instruments</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Assessment color="secondary" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6">{outputFiles.output_files.length}</Typography>
                  <Typography color="textSecondary">Output Files</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Timeline color="success" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6">{outputFiles.plots.length}</Typography>
                  <Typography color="textSecondary">Plots Generated</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <TrendingUp color="warning" sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="h6">Active</Typography>
                  <Typography color="textSecondary">System Status</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Available Instruments */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Available Instruments
              </Typography>
              <List dense>
                {instruments.slice(0, 5).map((instrument) => (
                  <ListItem key={instrument.name}>
                    <ListItemText
                      primary={instrument.name}
                      secondary={`${instrument.currency} • ${instrument.exchange}`}
                    />
                    <Chip 
                      label={`${instrument.rows} rows`} 
                      size="small" 
                      variant="outlined" 
                    />
                  </ListItem>
                ))}
                {instruments.length > 5 && (
                  <ListItem>
                    <ListItemText 
                      primary={`... and ${instruments.length - 5} more`}
                      secondary="View all in Instrument Data section"
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Output Files */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Output Files
              </Typography>
              <List dense>
                {outputFiles.output_files.slice(0, 5).map((file) => (
                  <ListItem key={file.name}>
                    <ListItemText
                      primary={file.name}
                      secondary={new Date(file.modified).toLocaleDateString()}
                    />
                    <Chip 
                      label={`${(file.size / 1024).toFixed(1)} KB`} 
                      size="small" 
                      variant="outlined" 
                    />
                  </ListItem>
                ))}
                {outputFiles.output_files.length === 0 && (
                  <ListItem>
                    <ListItemText 
                      primary="No output files yet"
                      secondary="Run analysis to generate files"
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Start Guide
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                1. View available instruments in the "Instrument Data" section
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                2. Run "Main Analysis" to process your trading strategies
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                3. Use "Fibonacci Analysis" for retracement level calculations
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                4. Validate results and calculate Sharpe ratios for performance analysis
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
