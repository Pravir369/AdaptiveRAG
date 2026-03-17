import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getFailuresAnalytics, type FailuresAnalytics } from '../api'

export default function FailuresAnalyticsPage() {
  const [data, setData] = useState<FailuresAnalytics | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getFailuresAnalytics()
      .then(setData)
      .catch(err => setError(err instanceof Error ? err.message : 'Failed'))
  }, [])

  if (error) return (
    <div className="page-container">
      <div style={{ padding: '1rem', background: 'var(--bg-danger)', color: 'var(--text-danger)', borderRadius: 'var(--radius-sm)' }}>{error}</div>
    </div>
  )
  if (!data) return (
    <div className="page-container">
      <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-tertiary)' }}>Loading analytics...</div>
    </div>
  )

  const labels = Object.entries(data.by_diagnosis_label ?? {})
  const topDocs = data.top_docs_by_failure_count ?? []
  const exampleTraces = data.example_trace_ids_by_label ?? {}
  const totalFailures = labels.reduce((sum, [, count]) => sum + count, 0)

  return (
    <div className="page-container">
      <h1 className="page-title">Failure Analytics</h1>
      <p className="page-subtitle">Aggregated from traces where self-healing was triggered.</p>

      {/* Summary stats */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="card" style={{ padding: '0.75rem 1rem', flex: 1 }}>
          <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Diagnosis Types</div>
          <div style={{ fontSize: '1.375rem', fontWeight: 700 }}>{labels.length}</div>
        </div>
        <div className="card" style={{ padding: '0.75rem 1rem', flex: 1 }}>
          <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Total Failures</div>
          <div style={{ fontSize: '1.375rem', fontWeight: 700 }}>{totalFailures}</div>
        </div>
        <div className="card" style={{ padding: '0.75rem 1rem', flex: 1 }}>
          <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Docs Affected</div>
          <div style={{ fontSize: '1.375rem', fontWeight: 700 }}>{topDocs.length}</div>
        </div>
      </div>

      {/* Diagnosis counts */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.625rem' }}>By Diagnosis Label</h2>
        {labels.length === 0 ? (
          <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '0.9375rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>No failure diagnoses recorded</div>
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
              Ask vague queries after ingesting documents to trigger self-healing.
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
            {labels.map(([label, count]) => {
              const maxCount = Math.max(...labels.map(([, c]) => c))
              const pct = maxCount > 0 ? (count / maxCount) * 100 : 0
              return (
                <div key={label} className="card" style={{ padding: '0.75rem 1rem', position: 'relative', overflow: 'hidden' }}>
                  <div style={{
                    position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: `${pct}%`,
                    background: 'rgba(37,99,235,0.06)',
                    transition: 'width 0.5s ease',
                  }} />
                  <div style={{ position: 'relative', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontWeight: 600, fontSize: '0.8125rem' }}>{label}</span>
                      {exampleTraces[label] && (
                        <Link to={`/trace/${exampleTraces[label]}/report`} style={{ marginLeft: '0.75rem', fontSize: '0.75rem' }}>
                          View example →
                        </Link>
                      )}
                    </div>
                    <span style={{
                      background: 'var(--bg-tertiary)',
                      padding: '0.125rem 0.5rem',
                      borderRadius: 9999,
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      color: 'var(--text-secondary)',
                    }}>{count}</span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Top docs */}
      <div>
        <h2 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.625rem' }}>Top Documents by Failure Frequency</h2>
        {topDocs.length === 0 ? (
          <div style={{ color: 'var(--text-tertiary)', fontSize: '0.8125rem' }}>No doc-level data yet.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                  <th style={{ textAlign: 'left', padding: '0.5rem 0.75rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Document</th>
                  <th style={{ textAlign: 'right', padding: '0.5rem 0.75rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Failures</th>
                </tr>
              </thead>
              <tbody>
                {topDocs.map(d => (
                  <tr key={d.doc_id} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '0.5rem 0.75rem' }}>
                      <div style={{ fontWeight: 500 }}>{d.title || d.doc_id.slice(0, 12)}</div>
                      <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{d.doc_id.slice(0, 12)}…</div>
                    </td>
                    <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right', fontWeight: 600 }}>{d.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
