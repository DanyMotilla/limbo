import React from 'react';

export default function TabPanel({ tabs, activeTab, onTabChange }) {
  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      overflow: 'hidden'
    }}>
      <div style={{
        display: 'flex',
        borderBottom: '1px solid #ccc',
        backgroundColor: '#f0f0f0',
        minHeight: '40px'
      }}>
        {tabs.map((tab, index) => (
          <button
            key={tab.label}
            onClick={() => onTabChange(index)}
            style={{
              padding: '0.75rem 1.5rem',
              border: 'none',
              borderBottom: activeTab === index ? '2px solid #007bff' : '2px solid transparent',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              color: activeTab === index ? '#007bff' : '#333',
              fontWeight: activeTab === index ? '500' : 'normal'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{ 
        flex: 1,
        overflow: 'hidden',
        position: 'relative',
        height: 'calc(100% - 40px)'  // Subtract tab height
      }}>
        {tabs[activeTab].content}
      </div>
    </div>
  );
}
