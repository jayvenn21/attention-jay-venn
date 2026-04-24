export default function TimelineControls({ viewMode, iteration, layer, meta, onIterationChange, onLayerChange }) {
  return (
    <div className="panel timeline-panel">
      {viewMode === 'evolution' ? (
        <>
          <h2>Recycling Iterations</h2>
          <input
            className="slider"
            type="range"
            min="0"
            max={meta ? 3 : 0}
            value={iteration}
            onChange={e => onIterationChange(parseInt(e.target.value))}
          />
          <span>Iteration: {iteration}</span>
        </>
      ) : (
        <>
          <h2>ESM-2 Transformer Layers</h2>
          <input
            className="slider"
            type="range"
            min="0"
            max="35"
            value={layer}
            onChange={e => onLayerChange(parseInt(e.target.value))}
          />
          <span>Layer: {layer}</span>
        </>
      )}
    </div>
  )
}
