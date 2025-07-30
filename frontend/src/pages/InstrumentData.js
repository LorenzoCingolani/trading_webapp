import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Paper,
  Card,
  CardContent,
  Grid,
  Chip
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import { Storage, TrendingUp, Assessment } from '@mui/icons-material';
import api from '../services/api';

function InstrumentData() {
  const [instruments, setInstruments] = useState([]);
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [instrumentData, setInstrumentData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInstruments = async () => {
      try {
        const response = await api.get('/instruments');
        setInstruments(response.data);
      } catch (error) {
        console.error('Error fetching instruments:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchInstruments();
  }, []);

  const handleInstrumentSelect = async (instrument) => {
    setSelectedInstrument(instrument);
    try {
      const response = await api.get(`/instrument/${instrument.name}`);
      setInstrumentData(response.data);
    } catch (error) {
      console.error('Error fetching instrument data:', error);
    }
  };

  const instrumentColumns = [
    { field: 'name', headerName: 'Instrument', width: 150 },
    { field: 'currency', headerName: 'Currency', width: 100 },
    { field: 'exchange', headerName: 'Exchange', width: 150 },
    { field: 'sectype', headerName: 'Security Type', width: 150 },
    { field: 'rows', headerName: 'Data Points', width: 120, type: 'number' },
  ];

  const getDataColumns = () => {
    if (!instrumentData || !instrumentData.columns) return [];
    
    return instrumentData.columns.map((col, index) => ({
      field: col,
      headerName: col,
      width: 150,
      type: ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_CLOSE_1D'].includes(col) ? 'number' : 'string'
    }));
  };

  const getDataRows = () => {
    if (!instrumentData || !instrumentData.data) return [];
    
    return instrumentData.data.map((row, index) => ({
      id: index,
      ...row
    }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Instrument Data
      </Typography>
      
      <Typography variant="body1" color="textSecondary" paragraph>
        Browse and explore available trading instruments and their historical data.
      </Typography>

      <Grid container spacing={3}>
        {/* Instruments List */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Storage sx={{ mr: 1, verticalAlign: 'middle' }} />
                Available Instruments ({instruments.length})
              </Typography>
              <Box sx={{ height: 400, width: '100%' }}>
                <DataGrid
                  rows={instruments.map((inst, index) => ({ id: index, ...inst }))}
                  columns={instrumentColumns}
                  pageSize={10}
                  rowsPerPageOptions={[10]}
                  onRowClick={(params) => handleInstrumentSelect(params.row)}
                  loading={loading}
                  density="compact"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Instrument Details */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                Instrument Details
              </Typography>
              {selectedInstrument ? (
                <Box>
                  <Typography variant="h5" gutterBottom>
                    {selectedInstrument.name}
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Chip label={selectedInstrument.currency} sx={{ mr: 1 }} />
                    <Chip label={selectedInstrument.exchange} sx={{ mr: 1 }} />
                    <Chip label={selectedInstrument.sectype} />
                  </Box>
                  <Typography variant="body2" color="textSecondary">
                    Data Points: {selectedInstrument.rows}
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  Select an instrument from the list to view details
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Historical Data */}
        {instrumentData && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Historical Data - {instrumentData.name}
                </Typography>
                <Box sx={{ height: 600, width: '100%' }}>
                  <DataGrid
                    rows={getDataRows()}
                    columns={getDataColumns()}
                    pageSize={20}
                    rowsPerPageOptions={[20, 50, 100]}
                    density="compact"
                    sx={{
                      '& .MuiDataGrid-cell': {
                        fontSize: '0.875rem'
                      }
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Data Information
        </Typography>
        <Typography variant="body2" paragraph>
          The instrument data contains historical price information including:
        </Typography>
        <ul>
          <li><strong>PX_OPEN:</strong> Opening price</li>
          <li><strong>PX_HIGH:</strong> Highest price of the day</li>
          <li><strong>PX_LOW:</strong> Lowest price of the day</li>
          <li><strong>PX_LAST:</strong> Last traded price</li>
          <li><strong>PX_CLOSE_1D:</strong> Closing price</li>
          <li><strong>VOLUME:</strong> Trading volume</li>
          <li><strong>Contract specifications:</strong> Tick size, point value, exchange rates, etc.</li>
        </ul>
      </Paper>
    </Box>
  );
}

export default InstrumentData;
