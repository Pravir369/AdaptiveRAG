import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getTrace, type TraceItem } from '../api'

export default function Trace() {
  const { traceId } = useParams<{ traceId: string }>()
  const [trace, setTrace] = useState<TraceItem | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!traceId) return
    getTrace(traceId)
      .then(setTrace)
      .catch(err => setError(err instanceof Error ? err.message : 'Failed'))
  }, [traceId])

  if (error) return (
    <div className="page-container">
      <div style={{ padding: '1rem', background: 'var(--bg-danger)', color: 'var(--text-danger)', borderRadius: 'var(--radius-sm)' }}>{error}</div>
    </div>
  )
  if (!trace) return (
    <div className="page-container">
      <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-tertiary)' }}>Loading trace...</div>
    </div>
  )

  return (
    <div className="page-container">
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', fontSize: '0.8125rem' }}>
        <Link to="/">← Chat</Link>
        <span style={{ color: 'var(--text-tertiary)' }}>·</span>
        <Link to={`/trace/${traceId}/report`}>Self-Heal Report</Link>
      </div>

      <h1 className="page-title">Trace {traceId?.slice(0, 8)}…</h1>

      {/* Info cards */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <InfoCard label="Query" value={trace.query} wide />
        <InfoCard label="Provider" value={trace.embedding_provider ?? '—'} />
        <InfoCard label="Top-K" value={String(trace.top_k ?? '—')} />
        <InfoCard label="Latency" value={`${trace.latency_ms ?? '—'} ms`} />
        {trace.generator_provider && <InfoCard label="Generator" value={trace.generator_provider} />}
      </div>

      <Section title="Retrieved" count={trace.retrieved.length}>
        <JsonBlock data={trace.retrieved} />
      </Section>

      <Section title="Reranked" count={Array.isArray(trace.reranked) ? trace.reranked.length : 0}>
        <JsonBlock data={trace.reranked} />
      </Section>

      <Section title="Selected" count={Array.isArray(trace.selected) ? trace.selected.length : 0}>
        <JsonBlock data={trace.selected} />
      </Section>

      <Section title="Citations" count={trace.citations.length}>
        <JsonBlock data={trace.citations} />
      </Section>

      <Section title="Answer">
        <div style={{
          whiteSpace: 'pre-wrap',
          background: 'var(--bg-secondary)',
          padding: '1rem',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border)',
          fontSize: '0.875rem',
          lineHeight: 1.6,
        }}>
          {trace.answer ?? '—'}
        </div>
      </Section>
    </div>
  )
}

function InfoCard({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className="card" style={{
      padding: '0.625rem 0.875rem',
      flex: wide ? '1 0 100%' : '1 0 auto',
      minWidth: wide ? undefined : 100,
    }}>
      <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: '0.125rem' }}>{label}</div>
      <div style={{ fontSize: wide ? '0.875rem' : '1rem', fontWeight: wide ? 400 : 600, wordBreak: 'break-word' }}>{value}</div>
    </div>
  )
}

function Section({ title, count, children }: { title: string; count?: number; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <h2 style={{ fontSize: '0.875rem', fontWeight: 600, margin: 0 }}>{title}</h2>
        {count != null && (
          <span style={{
            fontSize: '0.6875rem',
            background: 'var(--bg-tertiary)',
            padding: '0.125rem 0.5rem',
            borderRadius: 9999,
            color: 'var(--text-secondary)',
          }}>{count}</span>
        )}
      </div>
      {children}
    </section>
  )
}

function JsonBlock({ data }: { data: unknown }) {
  return (
    <pre style={{
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      padding: '0.875rem',
      borderRadius: 'var(--radius-sm)',
      overflow: 'auto',
      fontSize: '0.75rem',
      lineHeight: 1.5,
      maxHeight: 400,
      margin: 0,
      color: 'var(--text-secondary)',
    }}>
      {JSON.stringify(data, null, 2)}
    </pre>
  )
}
