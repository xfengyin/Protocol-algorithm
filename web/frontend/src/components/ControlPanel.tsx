import React from 'react'

interface ControlPanelProps {
  config: {
    nodes: number
    rounds: number
    p: number
    area: number
  }
  onConfigChange: (config: any) => void
  onRun: () => void
  isRunning: boolean
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  config,
  onConfigChange,
  onRun,
  isRunning,
}) => {
  const handleChange = (key: string, value: number) => {
    onConfigChange({ ...config, [key]: value })
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        Simulation Configuration
      </h2>

      <div className="space-y-4">
        {/* Number of Nodes */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Number of Nodes
          </label>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={config.nodes}
            onChange={(e) => handleChange('nodes', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            disabled={isRunning}
          />
          <div className="text-right text-sm text-gray-600 mt-1">
            {config.nodes}
          </div>
        </div>

        {/* Number of Rounds */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Number of Rounds
          </label>
          <input
            type="range"
            min="10"
            max="1000"
            step="10"
            value={config.rounds}
            onChange={(e) => handleChange('rounds', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            disabled={isRunning}
          />
          <div className="text-right text-sm text-gray-600 mt-1">
            {config.rounds}
          </div>
        </div>

        {/* Cluster Head Probability */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            CH Probability (p)
          </label>
          <input
            type="range"
            min="0.01"
            max="0.2"
            step="0.01"
            value={config.p}
            onChange={(e) => handleChange('p', parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            disabled={isRunning}
          />
          <div className="text-right text-sm text-gray-600 mt-1">
            {(config.p * 100).toFixed(1)}%
          </div>
        </div>

        {/* Area Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Area Size (m²)
          </label>
          <input
            type="range"
            min="50"
            max="200"
            step="10"
            value={config.area}
            onChange={(e) => handleChange('area', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            disabled={isRunning}
          />
          <div className="text-right text-sm text-gray-600 mt-1">
            {config.area} × {config.area}
          </div>
        </div>

        {/* Run Button */}
        <button
          onClick={onRun}
          disabled={isRunning}
          className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
            isRunning
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isRunning ? 'Running...' : 'Run Simulation'}
        </button>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-4 border-t">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Legend</h3>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-600"></div>
            <span className="text-sm text-gray-600">Normal Node</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-600"></div>
            <span className="text-sm text-gray-600">Cluster Head</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-0 h-0 border-l-4 border-r-4 border-b-8 border-l-transparent border-r-transparent border-b-green-600"></div>
            <span className="text-sm text-gray-600">Base Station</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ControlPanel
