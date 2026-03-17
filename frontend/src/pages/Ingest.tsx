import { useState, useRef } from 'react'
import { ingestUrl, ingestUpload } from '../api'

export default function Ingest() {
  const [url, setUrl] = useState('')
  const [urlTitle, setUrlTitle] = useState('')
  const [urlMsg, setUrlMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [urlLoading, setUrlLoading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleUrl(e: React.FormEvent) {
    e.preventDefault()
    if (!url.trim()) return
    setUrlMsg(null)
    setUrlLoading(true)
    try {
      const res = await ingestUrl(url.trim(), urlTitle.trim() || undefined)
      setUrlMsg({ type: 'ok', text: res.message })
      setUrl('')
      setUrlTitle('')
    } catch (err) {
      setUrlMsg({ type: 'err', text: err instanceof Error ? err.message : 'Failed' })
    } finally {
      setUrlLoading(false)
    }
  }

  async function processFile(file: File) {
    setUploadMsg(null)
    setUploading(true)
    try {
      const res = await ingestUpload(file)
      setUploadMsg({ type: 'ok', text: res.message })
    } catch (err) {
      setUploadMsg({ type: 'err', text: err instanceof Error ? err.message : 'Upload failed' })
    } finally {
      setUploading(false)
    }
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) processFile(file)
    e.target.value = ''
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) processFile(file)
  }

  return (
    <div className="page-container">
      <h1 className="page-title">Ingest Documents</h1>
      <p className="page-subtitle">Add documents to your knowledge base for RAG retrieval.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Upload card */}
        <div className="card">
          <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.25rem' }}>Upload File</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '1rem' }}>
            Supports .txt, .md, .json formats
          </div>

          <div
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${dragOver ? 'var(--bg-accent)' : 'var(--border-light)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '2rem',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all var(--transition)',
              background: dragOver ? 'rgba(37,99,235,0.05)' : 'transparent',
            }}
          >
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📁</div>
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
              {uploading ? 'Uploading...' : 'Drop file here or click to browse'}
            </div>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>
              .txt, .md, .json
            </div>
            <input ref={fileRef} type="file" accept=".txt,.md,.json" onChange={handleFile} disabled={uploading} style={{ display: 'none' }} />
          </div>

          {uploadMsg && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.5rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.8125rem',
              background: uploadMsg.type === 'ok' ? 'var(--bg-success)' : 'var(--bg-danger)',
              color: uploadMsg.type === 'ok' ? 'var(--text-success)' : 'var(--text-danger)',
            }}>
              {uploadMsg.text}
            </div>
          )}
        </div>

        {/* URL card */}
        <div className="card">
          <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.25rem' }}>Ingest URL</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '1rem' }}>
            Fetch and index content from a web page
          </div>

          <form onSubmit={handleUrl} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <input
              type="url"
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder="https://example.com/article"
            />
            <input
              type="text"
              value={urlTitle}
              onChange={e => setUrlTitle(e.target.value)}
              placeholder="Title (optional)"
            />
            <button type="submit" className="btn-primary" disabled={urlLoading || !url.trim()}>
              {urlLoading ? 'Ingesting...' : 'Ingest URL'}
            </button>
          </form>

          {urlMsg && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.5rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.8125rem',
              background: urlMsg.type === 'ok' ? 'var(--bg-success)' : 'var(--bg-danger)',
              color: urlMsg.type === 'ok' ? 'var(--text-success)' : 'var(--text-danger)',
            }}>
              {urlMsg.text}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
