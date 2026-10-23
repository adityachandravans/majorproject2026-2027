import React from 'react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'sim', label: '⚡ Live Simulation' },
    { id: 'config', label: '⚙️ Scenario Config' },
    { id: 'compare', label: '⚔️ Compare Policies' },
    { id: 'training', label: '📊 Training & Results' }
  ];

  return (
    <header className="navbar">
      <div className="navbar-brand">
        🤖 MARL Disaster Response System
      </div>
      <nav className="navbar-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
};
