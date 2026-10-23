import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { LiveSimulation } from './components/LiveSimulation';
import { ScenarioConfig } from './components/ScenarioConfig';
import { ComparePolicies } from './components/ComparePolicies';
import { TrainingResults } from './components/TrainingResults';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('sim');

  const handleSimulationCreated = (simId: string) => {
    setActiveTab('sim');
  };

  return (
    <div className="app-container">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="main-content">
        {activeTab === 'sim' && <LiveSimulation />}
        {activeTab === 'config' && <ScenarioConfig onSimulationCreated={handleSimulationCreated} />}
        {activeTab === 'compare' && <ComparePolicies />}
        {activeTab === 'training' && <TrainingResults />}
      </main>
    </div>
  );
};

export default App;
