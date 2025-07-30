import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  Timeline as TimelineIcon,
  Verified as VerifiedIcon,
  PieChart as PieChartIcon,
  TrendingUp as TrendingUpIcon,
  Storage as StorageIcon
} from '@mui/icons-material';

const drawerWidth = 240;

const menuItems = [
  { text: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { text: 'Main Analysis', path: '/main-analysis', icon: <AnalyticsIcon /> },
  { text: 'Fibonacci Analysis', path: '/fibonacci', icon: <TimelineIcon /> },
  { text: 'Validation', path: '/validation', icon: <VerifiedIcon /> },
  { text: 'PDM', path: '/pdm', icon: <PieChartIcon /> },
  { text: 'Sharpe Ratio', path: '/sharpe-ratio', icon: <TrendingUpIcon /> },
  { text: 'Instrument Data', path: '/instruments', icon: <StorageIcon /> },
];

function Navigation() {
  const location = useLocation();

  return (
    <Drawer
      variant="permanent"
      sx={{
        display: { xs: 'none', sm: 'block' },
        '& .MuiDrawer-paper': {
          boxSizing: 'border-box',
          width: drawerWidth,
          top: '64px',
          height: 'calc(100% - 64px)',
        },
      }}
    >
      <Box sx={{ overflow: 'auto' }}>
        <List>
          {menuItems.map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton
                component={Link}
                to={item.path}
                selected={location.pathname === item.path}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
}

export default Navigation;
