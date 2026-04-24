import PlotComponent from 'react-plotly.js'
const Plot = PlotComponent.default || PlotComponent

const COLORBAR_TITLES = {
  evolution: 'Avg. Pair Repr.',
  attention: 'Avg. Attention Weight',
}

export default function TraceExplorer({ heatmapData, viewMode, selectedResidue, onResidueClick, loading }) {
  let crosshairShapes = []
  if (selectedResidue && heatmapData && heatmapData.length > 1) {
    const idx = selectedResidue - 1
    const maxIdx = heatmapData.length - 1
    crosshairShapes = [
      { type: 'line', x0: idx, x1: idx, y0: 0, y1: maxIdx, line: { color: '#ff0055', width: 2, dash: 'dot' } },
      { type: 'line', x0: 0, x1: maxIdx, y0: idx, y1: idx, line: { color: '#ff0055', width: 2, dash: 'dot' } },
    ]
  }

  const handleClick = (event) => {
    if (event.points && event.points.length > 0) {
      onResidueClick(event.points[0].x + 1)
    }
  }

  if (loading) return <div className="panel-message">Loading trace data…</div>
  if (!heatmapData) return <div className="panel-message">No trace data available</div>

  return (
    <Plot
      data={[{
        z: heatmapData,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {
          title: { text: COLORBAR_TITLES[viewMode] || '', side: 'right' },
          tickfont: { color: '#a0a0a0' },
          titlefont: { color: '#a0a0a0' },
        },
      }]}
      layout={{
        autosize: true,
        paper_bgcolor: '#1e1e1e',
        plot_bgcolor: '#1e1e1e',
        font: { color: '#a0a0a0' },
        margin: { t: 20, r: 80, l: 50, b: 50 },
        xaxis: { title: 'Residue i' },
        yaxis: { title: 'Residue j', autorange: 'reversed' },
        shapes: crosshairShapes,
      }}
      config={{ displayModeBar: false }}
      useResizeHandler
      className="trace-plot"
      onClick={handleClick}
    />
  )
}
