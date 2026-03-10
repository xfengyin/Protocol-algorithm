import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface NetworkVizProps {
  nodes: number
  area: number
  isRunning: boolean
}

interface Node {
  id: number
  x: number
  y: number
  isClusterHead: boolean
}

const NetworkViz: React.FC<NetworkVizProps> = ({ nodes, area, isRunning }) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    const width = 600
    const height = 500
    const margin = 40

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)

    // Create group for content
    const g = svg.append('g')
      .attr('transform', `translate(${margin}, ${margin})`)

    // Generate random nodes
    const nodeData: Node[] = Array.from({ length: nodes }, (_, i) => ({
      id: i,
      x: Math.random() * (area - 10) + 5,
      y: Math.random() * (area - 10) + 5,
      isClusterHead: Math.random() < 0.05,
    }))

    // Ensure at least one cluster head
    if (!nodeData.some(n => n.isClusterHead)) {
      nodeData[Math.floor(Math.random() * nodes)].isClusterHead = true
    }

    const plotWidth = area
    const plotHeight = area
    const xScale = d3.scaleLinear()
      .domain([0, area])
      .range([0, plotWidth])

    const yScale = d3.scaleLinear()
      .domain([0, area])
      .range([plotHeight, 0])

    // Draw links (nodes to cluster heads)
    const clusterHeads = nodeData.filter(n => n.isClusterHead)
    const normalNodes = nodeData.filter(n => !n.isClusterHead)

    normalNodes.forEach(node => {
      const nearestCH = clusterHeads.reduce((nearest, current) => {
        const distNode = Math.hypot(node.x - current.x, node.y - current.y)
        const distNearest = Math.hypot(node.x - nearest.x, node.y - nearest.y)
        return distNode < distNearest ? current : nearest
      })

      g.append('line')
        .attr('x1', xScale(node.x))
        .attr('y1', yScale(node.y))
        .attr('x2', xScale(nearestCH.x))
        .attr('y2', yScale(nearestCH.y))
        .attr('stroke', '#94A3B8')
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.3)
    })

    // Draw normal nodes with glow effect
    normalNodes.forEach(node => {
      // Glow
      g.append('circle')
        .attr('cx', xScale(node.x))
        .attr('cy', yScale(node.y))
        .attr('r', 6)
        .attr('fill', '#2563EB')
        .attr('opacity', 0.2)

      // Node
      g.append('circle')
        .attr('cx', xScale(node.x))
        .attr('cy', yScale(node.y))
        .attr('r', 4)
        .attr('fill', '#2563EB')
        .attr('opacity', 0.8)
    })

    // Draw cluster heads with pulse effect
    clusterHeads.forEach(node => {
      // Outer pulse
      g.append('circle')
        .attr('cx', xScale(node.x))
        .attr('cy', yScale(node.y))
        .attr('r', 8)
        .attr('fill', '#DC2626')
        .attr('opacity', 0.3)

      // Inner pulse
      g.append('circle')
        .attr('cx', xScale(node.x))
        .attr('cy', yScale(node.y))
        .attr('r', 6)
        .attr('fill', '#DC2626')
        .attr('opacity', 0.5)

      // Cluster head
      g.append('circle')
        .attr('cx', xScale(node.x))
        .attr('cy', yScale(node.y))
        .attr('r', 5)
        .attr('fill', '#DC2626')
        .attr('opacity', 1.0)
    })

    // Draw base station
    const bsX = xScale(50)
    const bsY = yScale(150)
    
    g.append('polygon')
      .attr('points', `${bsX},${bsY - 8} ${bsX + 8},${bsY + 8} ${bsX - 8},${bsY + 8}`)
      .attr('fill', '#16A34A')

    g.append('text')
      .attr('x', bsX)
      .attr('y', bsY + 22)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10')
      .attr('font-weight', 'bold')
      .attr('fill', '#1E293B')
      .text('BS')

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5)
    const yAxis = d3.axisLeft(yScale).ticks(5)

    g.append('g')
      .attr('transform', `translate(0, ${plotHeight})`)
      .call(xAxis)
      .attr('font-size', '10')
      .attr('color', '#64748B')

    g.append('g')
      .call(yAxis)
      .attr('font-size', '10')
      .attr('color', '#64748B')

    // Axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 8)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12')
      .attr('fill', '#1E293B')
      .text('X Position (m)')

    svg.append('text')
      .attr('x', 15)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12')
      .attr('fill', '#1E293B')
      .attr('transform', 'rotate(-90)')
      .text('Y Position (m)')

  }, [nodes, area, isRunning])

  return (
    <div className="flex justify-center">
      <svg
        ref={svgRef}
        className="max-w-full h-auto"
        style={{ maxHeight: '500px' }}
      />
    </div>
  )
}

export default NetworkViz
