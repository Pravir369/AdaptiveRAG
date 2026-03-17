import { useState, useEffect } from 'react'
import { listDocs, type DocItem } from '../api'

export default function Docs() {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listDocs()
      .then(res => setDocs(res.docs))
      .catch(err => setError(err instanceof Error ? err.message : 'Failed'))
      .finally(() => setLoading(false))
  }, [])

  const totalChunks = docs.reduce((sum, d) => sum + d.chunk_count, 0)

  return (
    <div className="page-container">
      <h1 className="page-title">Documents</h1>
      <p className="page-subtitle">Your indexed knowledge base for RAG retrieval.</p>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-tertiary)' }}>Loading documents...</div>
      ) : error ? (
        <div style={{ padding: '1rem', background: 'var(--bg-danger)', color: 'var(--text-danger)', borderRadius: 'var(--radius-sm)' }}>{error}</div>
      ) : docs.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>📄</div>
          <div style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.375rem' }}>No documents yet</div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>Ingest files or URLs from the Ingest page to get started.</div>
        </div>
      ) : (
        <>
          {/* Stats bar */}
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem' }}>
            <div className="card" style={{ padding: '0.75rem 1rem', flex: 1 }}>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Documents</div>
              <div style={{ fontSize: '1.375rem', fontWeight: 700 }}>{docs.length}</div>
            </div>
            <div className="card" style={{ padding: '0.75rem 1rem', flex: 1 }}>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Total Chunks</div>
              <div style={{ fontSize: '1.375rem', fontWeight: 700 }}>{totalChunks}</div>
            </div>
          </div>

          {/* Document list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {docs.map(d => (
              <div key={d.doc_id} className="card" style={{
                padding: '0.875rem 1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                transition: 'border-color var(--transition)',
              }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{d.title || d.doc_id.slice(0, 12)}</span>
                    <span className="badge badge-accent">{d.source_type}</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', display: 'flex', gap: '0.75rem' }}>
                    <span style={{ fontFamily: 'monospace' }}>{d.doc_id.slice(0, 12)}…</span>
                    {d.source_ref && <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 300 }}>{d.source_ref}</span>}
                    {d.created_at && <span>{new Date(d.created_at).toLocaleDateString()}</span>}
                  </div>
                </div>
                <div style={{
                  background: 'var(--bg-tertiary)',
                  padding: '0.375rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.8125rem',
                  fontWeight: 600,
                  color: 'var(--text-secondary)',
                  whiteSpace: 'nowrap',
                }}>
                  {d.chunk_count} chunks
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
