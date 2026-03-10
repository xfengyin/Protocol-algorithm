import { useState, useEffect } from 'react'
import NetworkViz from './components/NetworkViz'
import MetricsChart from './components/MetricsChart'
import ControlPanel from './components/ControlPanel'

function App() {
  const [config, setConfig] = useState({
    nodes: 100,
    rounds: 100,
    p: 0.05,
    area: 100,
  })

  const [simulationState, setSimulationState] = useState<'idle' | 'running' | 'completed'>('idle')
  const [results, setResults] = useState<any>(null)

  const runSimulation = async () => {
    setSimulationState('running')
    // Simulate API call
    setTimeout(() => {
      setSimulationState('completed')
      setResults({
        rounds: config.rounds,
        alive_nodes: Math.floor(config.nodes * 0.7),
        energy_consumed: Math.random() * 100,
      })
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">
            Protocol-algorithm <span className="text-blue-600">v2.0</span>
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            WSN LEACH Protocol Simulation Platform
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Controls */}
          <div className="lg:col-span-1">
            <ControlPanel
              config={config}
              onConfigChange={setConfig}
              onRun={runSimulation}
              isRunning={simulationState === 'running'}
            />
          </div>

          {/* Right Column - Visualizations */}
          <div className="lg:col-span-2 space-y-6">
            {/* Network Visualization */}
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Network Topology
              </h2>
              <NetworkViz
                nodes={config.nodes}
                area={config.area}
                isRunning={simulationState === 'running'}
              />
            </div>

            {/* Metrics Chart */}
            {results && (
              <div className="bg-white rounded-lg shadow-sm border p-4">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                  Simulation Results
                </h2>
                <MetricsChart results={results} />
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 border-t bg-white">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          Protocol-algorithm v2.0 - High-performance WSN Simulation
        </div>
      </footer>
    </div>
  )
}

export default App
