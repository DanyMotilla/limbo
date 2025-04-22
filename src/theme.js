import { createTheme } from '@mui/material/styles';

// Solarized Light Theme Colors
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1a1a1a',
      light: '#2c2c2c',
      dark: '#000000',
      contrastText: '#fdf6e3',
    },
    secondary: {
      main: '#93a1a1',
      light: '#eee8d5',
      dark: '#657b83',
      contrastText: '#002b36',
    },
    background: {
      default: '#fdf6e3', // Solarized Light base
      paper: '#eee8d5',   // Solarized Light background highlights
      dark: '#002b36',    // Dark background for node system
    },
    text: {
      primary: '#1a1a1a',
      secondary: '#586e75',
      disabled: '#93a1a1',
    },
    divider: 'rgba(0, 0, 0, 0.12)',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          color: '#1a1a1a',
        },
        thumb: {
          '&:hover, &.Mui-focusVisible': {
            boxShadow: '0 0 0 8px rgba(26, 26, 26, 0.16)',
          },
        },
        track: {
          opacity: 0.8,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          borderRadius: '4px 4px 0 0',
          '&.Mui-selected': {
            backgroundColor: '#eee8d5',
            color: '#268bd2',
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          backgroundColor: '#fdf6e3',
          borderBottom: '1px solid #eee8d5',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#fdf6e3',
        },
      },
    },
  },
});
