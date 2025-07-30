import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Box } from '@mui/material';
import Navigation from './components/Navigation';
import Dashboard from './pages/Dashboard';
import MainAnalysis from './pages/MainAnalysis';
import FibonacciAnalysis from './pages/FibonacciAnalysis';
import Validation from './pages/Validation';
import PDM from './pages/PDM';
import SharpeRatio from './pages/SharpeRatio';
import InstrumentData from './pages/InstrumentData';

function App() {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Trading Web Application
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Box sx={{ display: 'flex', flexGrow: 1 }}>
        <Navigation />
        
        <Container 
          component="main" 
          sx={{ 
            flexGrow: 1, 
            py: 3,
            ml: { sm: '240px' },
            maxWidth: 'none !important'
          }}
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/main-analysis" element={<MainAnalysis />} />
            <Route path="/fibonacci" element={<FibonacciAnalysis />} />
            <Route path="/validation" element={<Validation />} />
            <Route path="/pdm" element={<PDM />} />
            <Route path="/sharpe-ratio" element={<SharpeRatio />} />
            <Route path="/instruments" element={<InstrumentData />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
