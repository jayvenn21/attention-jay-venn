import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import StructureViewer from './components/StructureViewer'
import TraceExplorer from './components/TraceExplorer'
import TimelineControls from './components/TimelineControls'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

function App() {
  const [meta, setMeta] = useState(null)
  const [metaError, setMetaError] = useState(null)

  const [pdbText, setPdbText] = useState(null)
  const [structureError, setStructureError] = useState(null)

  const [viewMode, setViewMode] = useState('evolution')
  const [iteration, setIteration] = useState(0)
  const [layer, setLayer] = useState(0)
  const [heatmapData, setHeatmapData] = useState(null)
  const [heatmapLoading, setHeatmapLoading] = useState(false)

  const [renderStyle, setRenderStyle] = useState('cartoon')
  const [colorMode, setColorMode] = useState('confidence')

  const [selectedResidue, setSelectedResidue] = useState(null)

  useEffect(() => {
    axios.get(`${API_BASE}/meta`)
      .then(r => setMeta(r.data))
      .catch(() => setMetaError('Could not connect to backend'))

    axios.get(`${API_BASE}/structure?t=${Date.now()}`)
      .then(r => setPdbText(r.data))
      .catch(() => setStructureError('Failed to load PDB structure'))
  }, [])

  useEffect(() => {
    if (!meta) return
    setHeatmapLoading(true)

    const url = viewMode === 'evolution'
      ? `${API_BASE}/tensor/activations/recycle_${iteration}_s_z`
      : `${API_BASE}/tensor/attention/layer_${layer.toString().padStart(3, '0')}`

    axios.get(url)
      .then(r => setHeatmapData(r.data.data))
      .catch(() => setHeatmapData(null))
      .finally(() => setHeatmapLoading(false))
  }, [iteration, layer, viewMode, meta])

  const handleResidueClick = useCallback((resi) => {
    setSelectedResidue(resi)
  }, [])

  if (metaError) {
    return (
      <div className="dashboard-container">
        <div className="panel error-panel">
          <p className="error-message">{metaError}</p>
          <p className="error-hint">Start the backend: python3 server.py</p>
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard-container">
      <div className="panel structure-panel">
        <div className="panel-header">
          <h2>Structure Viewer {meta && <span className="status-badge">Model: {meta.model_name}</span>}</h2>
          <div className="controls-group">
            <select className="dropdown" value={renderStyle} onChange={e => setRenderStyle(e.target.value)}>
              <option value="cartoon">Cartoon</option>
              <option value="stick">Sticks</option>
              <option value="sphere">Surface (Spheres)</option>
            </select>
            <select className="dropdown" value={colorMode} onChange={e => setColorMode(e.target.value)}>
              <option value="confidence">Color: Confidence</option>
              <option value="spectrum">Color: Rainbow</option>
            </select>
          </div>
        </div>
        <StructureViewer
          pdbText={pdbText}
          renderStyle={renderStyle}
          colorMode={colorMode}
          selectedResidue={selectedResidue}
          onResidueClick={handleResidueClick}
          error={structureError}
        />
      </div>

      <div className="panel trace-panel">
        <div className="panel-header">
          <h2>Trace Explorer</h2>
          {selectedResidue && <span className="residue-target">Target: Residue {selectedResidue}</span>}
          <select className="dropdown" value={viewMode} onChange={e => setViewMode(e.target.value)}>
            <option value="evolution">Trunk Evolution (s_z)</option>
            <option value="attention">ESM-2 Attention</option>
          </select>
        </div>
        <TraceExplorer
          heatmapData={heatmapData}
          viewMode={viewMode}
          selectedResidue={selectedResidue}
          onResidueClick={handleResidueClick}
          loading={heatmapLoading}
        />
      </div>

      <TimelineControls
        viewMode={viewMode}
        iteration={iteration}
        layer={layer}
        meta={meta}
        onIterationChange={setIteration}
        onLayerChange={setLayer}
      />
    </div>
  )
}

export default App
