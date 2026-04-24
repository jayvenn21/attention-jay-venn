import { useEffect, useRef } from 'react'

function applyStyle(viewer, renderStyle, colorMode) {
  const colorConfig = colorMode === 'spectrum'
    ? { color: 'spectrum' }
    : { colorscheme: { prop: 'b', gradient: 'rwb', min: 0.5, max: 1.0 } }
  viewer.setStyle({}, { [renderStyle]: colorConfig })
}

export default function StructureViewer({ pdbText, renderStyle, colorMode, selectedResidue, onResidueClick, error }) {
  const containerRef = useRef(null)
  const viewerRef = useRef(null)
  const zoomedRef = useRef(false)
  const internalClickRef = useRef(null)

  // Create viewer once when PDB data arrives
  useEffect(() => {
    if (!pdbText || !containerRef.current || !window.$3Dmol) return

    containerRef.current.innerHTML = ''
    const viewer = window.$3Dmol.createViewer(containerRef.current, { backgroundColor: '#1e1e1e' })
    viewer.addModel(pdbText, 'pdb')
    viewerRef.current = viewer
    zoomedRef.current = false

    viewer.setClickable({}, true, (atom) => {
      viewer.removeAllLabels()
      viewer.addLabel(`${atom.resn} ${atom.resi}`, {
        position: atom,
        backgroundColor: '#333333',
        fontColor: 'white',
        backgroundOpacity: 0.8,
      })
      viewer.render()
      internalClickRef.current = atom.resi
      onResidueClick(atom.resi)
    })

    return () => { viewerRef.current = null }
  }, [pdbText, onResidueClick])

  // Apply rendering style without recreating the viewer (preserves camera)
  useEffect(() => {
    const v = viewerRef.current
    if (!v) return
    applyStyle(v, renderStyle, colorMode)
    if (!zoomedRef.current) {
      v.zoomTo()
      zoomedRef.current = true
    }
    v.render()
  }, [pdbText, renderStyle, colorMode])

  // Highlight residue selected from the heatmap
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || selectedResidue == null) return

    // Skip if this selection originated from our own click handler
    if (internalClickRef.current === selectedResidue) {
      internalClickRef.current = null
      return
    }

    viewer.removeAllLabels()
    const atoms = viewer.selectedAtoms({ resi: selectedResidue, atom: 'CA' })
    if (atoms.length > 0) {
      const a = atoms[0]
      viewer.addLabel(`${a.resn} ${a.resi}`, {
        position: a,
        backgroundColor: '#333333',
        fontColor: 'white',
        backgroundOpacity: 0.8,
      })
    }
    viewer.render()
  }, [selectedResidue])

  if (error) return <div className="panel-message error">{error}</div>
  if (!pdbText) return <div className="panel-message">Loading structure…</div>

  return <div ref={containerRef} className="viewer-container" />
}
