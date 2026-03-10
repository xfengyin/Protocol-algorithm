import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from 'recharts'

interface MetricsChartProps {
  results: {
    rounds: number
    alive_nodes: number
    energy_consumed: number
  }
}

const MetricsChart: React.FC<MetricsChartProps> = ({ results }) => {
  // Generate sample data for demo
  const survivalData = Array.from({ length: results.rounds }, (_, i) => ({
    round: i,
    alive: Math.floor(results.alive_nodes * (1 - i / results.rounds * 0.3)),
    energy: Math.floor(results.energy_consumed * (i / results.rounds)),
  }))

  return (
    <div className="space-y-6">
      {/* Survival Rate Chart */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-4">
          Node Survival Over Time
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={survivalData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
            <XAxis
              dataKey="round"
              stroke="#64748B"
              fontSize={12}
              label={{ value: 'Round', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              stroke="#64748B"
              fontSize={12}
              label={{ value: 'Alive Nodes', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#F8FAFC',
                border: '1px solid #E2E8F0',
                borderRadius: '6px',
              }}
            />
            <Line
              type="monotone"
              dataKey="alive"
              stroke="#2563EB"
              strokeWidth={2}
              dot={false}
              name="Alive Nodes"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Final Alive</div>
          <div className="text-2xl font-bold text-blue-600">{results.alive_nodes}</div>
        </div>
        <div className="bg-red-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Energy Used</div>
          <div className="text-2xl font-bold text-red-600">
            {results.energy_consumed.toFixed(1)} J
          </div>
        </div>
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Survival Rate</div>
          <div className="text-2xl font-bold text-green-600">
            {((results.alive_nodes / 100) * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  )
}

export default MetricsChart
