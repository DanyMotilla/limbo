import React from 'react';
import { Tabs as MuiTabs, Tab as MuiTab, Box } from '@mui/material';

export default function TabPanel({ tabs, activeTab, onTabChange }) {
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      overflow: 'hidden',
      bgcolor: 'background.default'
    }}>
      <Box sx={{
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        position: 'relative',
        zIndex: 1,
        boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
      }}>
        <MuiTabs 
          value={activeTab} 
          onChange={(_, newValue) => onTabChange(newValue)}
          sx={{
            minHeight: '48px',
            '& .MuiTabs-indicator': {
              display: 'none'
            }
          }}
        >
          {tabs.map((tab, index) => (
            <MuiTab
              key={tab.label}
              label={tab.label}
              sx={{
                textTransform: 'none',
                fontWeight: 500,
                fontSize: '0.9rem',
                minHeight: '48px',
                borderTopLeftRadius: '8px',
                borderTopRightRadius: '8px',
                border: 1,
                borderColor: 'divider',
                borderBottom: 'none',
                mr: '-1px',
                position: 'relative',
                bgcolor: activeTab === index ? 'background.paper' : 'background.default',
                opacity: 1,
                '&::after': {
                  content: '""',
                  display: activeTab === index ? 'block' : 'none',
                  position: 'absolute',
                  bottom: -1,
                  left: 0,
                  right: 0,
                  height: '2px',
                  bgcolor: 'background.paper'
                },
                '&:hover': {
                  bgcolor: activeTab === index ? 'background.paper' : 'background.default',
                  opacity: 0.8
                }
              }}
            />
          ))}
        </MuiTabs>
      </Box>
      <Box sx={{ 
        flex: 1,
        overflow: 'hidden',
        position: 'relative',
        height: 'calc(100% - 48px)',
        bgcolor: 'background.paper',
        borderLeft: 1,
        borderRight: 1,
        borderBottom: 1,
        borderColor: 'divider'
      }}>
        {tabs[activeTab].content}
      </Box>
    </Box>
  );
}
