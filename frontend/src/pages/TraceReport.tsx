import { useState, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getTrace, type TraceItem } from '../api'

function highlightQueryTokens(snippet: string, query: string): string {
  if (!snippet || !query.trim()) return snippet
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (tokens.length === 0) return snippet
  const re = new RegExp(`(${tokens.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  return snippet.replace(re, '<mark>$1</mark>')
}

export default function TraceReport() {
  const { traceId } = useParams<{ traceId: string }>()
  const [trace, setTrace] = useState<TraceItem | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!traceId) return
    getTrace(traceId)
      .then(setTrace)
      .catch(err => setError(err instanceof Error ? err.message : 'Failed'))
  }, [traceId])

  const attempts = trace?.attempts ?? []
  const heals = trace?.heals_applied ?? []
  const selfHealTriggered = trace?.self_heal_triggered ?? false
  const triggerFailedThresholds = trace?.trigger_failed_thresholds ?? []
  const bestAttemptNo = trace?.best_attempt_no ?? 1
  const firstAttempt = attempts[0]
  const bestAttempt = attempts[bestAttemptNo - 1] ?? attempts[attempts.length - 1]
  const lastAttempt = attempts[attempts.length - 1]
  const beforeRetrieved = firstAttempt?.retrieved ?? []
  const afterRetrieved = selfHealTriggered ? (bestAttempt?.retrieved ?? lastAttempt?.retrieved ?? []) : beforeRetrieved
  const beforeMetrics = firstAttempt?.metrics ?? {}
  const afterMetrics = selfHealTriggered ? (bestAttempt?.metrics ?? lastAttempt?.metrics ?? {}) : beforeMetrics
  const deltas = useMemo(() => {
    if (heals.length === 0) return null
    const lastHeal = heals[heals.length - 1]
    return lastHeal.delta ?? {
      max_fused: (afterMetrics.max_fused_score ?? 0) - (beforeMetrics.max_fused_score ?? 0),
      lexical_overlap: (afterMetrics.lexical_overlap_ratio ?? 0) - (beforeMetrics.lexical_overlap_ratio ?? 0),
      score_separation: (afterMetrics.score_separation ?? 0) - (beforeMetrics.score_separation ?? 0),
    }
  }, [heals, beforeMetrics, afterMetrics])

  if (error) return <div className="page-container"><div style={{ padding: '1rem', background: 'var(--bg-danger)', color: 'var(--text-danger)', borderRadius: 'var(--radius-sm)' }}>{error}</div></div>
  if (!trace) return <div className="page-container"><div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-tertiary)' }}>Loading report...</div></div>

  return (
    <div className="page-container" style={{ maxWidth: 1000 }}>
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', fontSize: '0.8125rem' }}>
        <Link to={`/trace/${traceId}`}>← Trace</Link>
        <span style={{ color: 'var(--text-tertiary)' }}>·</span>
        <Link to="/">Chat</Link>
        <span style={{ color: 'var(--text-tertiary)' }}>·</span>
        <Link to="/analytics/failures">Analytics</Link>
      </div>

      <h1 className="page-title">Self-Heal Report</h1>
      <p className="page-subtitle" style={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{traceId}</p>

      {/* Query + Meta */}
      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: '0.375rem' }}>Query</div>
        <div style={{ fontSize: '0.9375rem', fontWeight: 500 }}>{trace.query}</div>
        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
          {(firstAttempt?.mode || bestAttempt?.mode) && (
            <span className="badge badge-accent">Mode: {firstAttempt?.mode ?? bestAttempt?.mode ?? 'normal'}</span>
          )}
          {trace.generator_provider && (
            <span className={`badge ${trace.generator_provider === 'ollama' ? 'badge-success' : trace.generator_provider === 'extractive_fallback' ? 'badge-warning' : 'badge-accent'}`}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: trace.generator_provider === 'ollama' ? 'var(--text-success)' : trace.generator_provider === 'extractive_fallback' ? 'var(--text-warning)' : 'var(--text-accent)', display: 'inline-block' }} />
              {trace.generator_provider}
            </span>
          )}
        </div>
        {trace.generator_metadata && (
          <div style={{ marginTop: '0.5rem', display: 'flex', flexWrap: 'wrap', gap: '0.75rem', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
            {trace.generator_metadata.model != null && <span>model: <strong style={{ color: 'var(--text-secondary)' }}>{String(trace.generator_metadata.model)}</strong></span>}
            {trace.generator_metadata.num_context_chunks != null && <span>chunks: {String(trace.generator_metadata.num_context_chunks)}</span>}
            {trace.generator_metadata.temperature != null && <span>temp: {String(trace.generator_metadata.temperature)}</span>}
            {trace.generator_metadata.max_tokens != null && <span>max_tokens: {String(trace.generator_metadata.max_tokens)}</span>}
            {trace.generator_metadata.citation_strict_mode != null && <span>strict_cite: {String(trace.generator_metadata.citation_strict_mode)}</span>}
          </div>
        )}
        {trace.generator_provider === 'extractive_fallback' && trace.generator_error && (
          <div style={{ marginTop: '0.5rem', padding: '0.5rem 0.75rem', background: 'var(--bg-warning)', borderRadius: 'var(--radius-sm)', fontSize: '0.8125rem', color: 'var(--text-warning)' }}>
            Ollama unavailable — used extractive fallback. Error: {trace.generator_error}
          </div>
        )}
      </div>

      {/* Self-heal status */}
      {!selfHealTriggered ? (
        <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-success)', borderRadius: 'var(--radius-sm)', color: 'var(--text-success)', fontSize: '0.875rem', marginBottom: '1.25rem', border: '1px solid rgba(52,211,153,0.2)' }}>
          Self-heal not triggered — thresholds passed on attempt 1.
        </div>
      ) : (
        <>
          {triggerFailedThresholds.length > 0 && (
            <div style={{ marginBottom: '1.25rem' }}>
              <SectionTitle>Failed Thresholds (Attempt 1)</SectionTitle>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
                {triggerFailedThresholds.map((t, i) => (
                  <div key={i} className="card" style={{ padding: '0.625rem 0.875rem', borderLeft: '3px solid var(--text-warning)' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.8125rem' }}>{t.name}</span>
                    <span style={{ color: 'var(--text-tertiary)', fontSize: '0.8125rem' }}>
                      {' — '}value <strong>{typeof t.value === 'number' ? t.value.toFixed(4) : t.value}</strong> (threshold: {typeof t.threshold === 'number' ? t.threshold.toFixed(4) : t.threshold})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Timeline */}
      <div style={{ marginBottom: '1.25rem' }}>
        <SectionTitle>Timeline</SectionTitle>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem', alignItems: 'center' }}>
          <PipelineStep label="Input" color="var(--bg-tertiary)" />
          {attempts.map((a, i) => (
            <span key={i} style={{ display: 'contents' }}>
              <Arrow />
              <PipelineStep label={`Attempt ${a.attempt_no}`} color="var(--bg-hover)" />
              {heals[i] && (
                <>
                  <Arrow />
                  <PipelineStep label={heals[i].type} color="rgba(37,99,235,0.2)" textColor="var(--text-accent)" />
                </>
              )}
            </span>
          ))}
          <Arrow />
          <PipelineStep label="Final" color="var(--bg-success)" textColor="var(--text-success)" />
        </div>
      </div>

      {/* Diagnoses */}
      {selfHealTriggered && firstAttempt?.diagnoses && firstAttempt.diagnoses.length > 0 && (
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle>Diagnoses</SectionTitle>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
            {firstAttempt.diagnoses.map((d, i) => (
              <div key={i} className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--text-warning)' }}>
                <div style={{ fontWeight: 600, fontSize: '0.8125rem' }}>{d.label}</div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>{d.reason}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Heals */}
      {heals.length > 0 && (
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle>Heals Applied</SectionTitle>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
            {heals.map((h, i) => (
              <div key={i} className="card" style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <span className="badge badge-accent">{h.type}</span>
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>{h.reason}</div>

                {h.type === 'QUERY_REWRITE' && Array.isArray(h.params?.rewrite_scores) && (
                  <div style={{ marginTop: '0.75rem', overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                          <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Rewrite</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Quality</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Max Fused</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Overlap</th>
                          <th style={{ textAlign: 'right', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Separation</th>
                          <th style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Pick</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(h.params.rewrite_scores as Array<{ rewrite: string; quality_score: number; max_fused: number; overlap_top3: number; separation: number }>).map((row, j) => {
                          const isChosen = row.rewrite === h.params?.chosen_rewrite
                          return (
                            <tr key={j} style={{ borderBottom: '1px solid var(--border)', background: isChosen ? 'var(--bg-success)' : undefined }}>
                              <td style={{ padding: '0.5rem', maxWidth: 260, wordBreak: 'break-word' }}>{row.rewrite}</td>
                              <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>{typeof row.quality_score === 'number' ? row.quality_score.toFixed(4) : '—'}</td>
                              <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>{typeof row.max_fused === 'number' ? row.max_fused.toFixed(4) : '—'}</td>
                              <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>{typeof row.overlap_top3 === 'number' ? row.overlap_top3.toFixed(4) : '—'}</td>
                              <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>{typeof row.separation === 'number' ? row.separation.toFixed(4) : '—'}</td>
                              <td style={{ padding: '0.5rem', textAlign: 'center' }}>{isChosen ? '✓' : ''}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {h.params && Object.keys(h.params).length > 0 && !Array.isArray(h.params.rewrite_scores) && (
                  <pre style={{ marginTop: '0.5rem', fontSize: '0.75rem', overflow: 'auto', color: 'var(--text-tertiary)', background: 'var(--bg-primary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                    {JSON.stringify(h.params, null, 2)}
                  </pre>
                )}

                {h.params && h.type === 'QUERY_REWRITE' && Array.isArray(h.params.rewrite_scores) && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                    Chosen: &quot;{String(h.params.chosen_rewrite ?? '—')}&quot;
                  </div>
                )}

                {h.delta && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8125rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <DeltaValue label="Δ max_fused" value={h.delta.max_fused} />
                    <DeltaValue label="Δ overlap" value={h.delta.lexical_overlap} />
                    {h.delta.overlap_top3 != null && <DeltaValue label="Δ overlap_top3" value={Number(h.delta.overlap_top3)} />}
                    <DeltaValue label="Δ separation" value={h.delta.score_separation} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metrics */}
      <div style={{ marginBottom: '1.25rem' }}>
        <SectionTitle>Metrics</SectionTitle>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.625rem' }}>
          <MetricCard label="Specificity (before)" value={beforeMetrics.query_specificity} />
          <MetricCard label="Specificity (after)" value={afterMetrics.query_specificity} />
          <MetricCard label="Overlap top3 (before)" value={beforeMetrics.overlap_top3 ?? beforeMetrics.lexical_overlap_ratio} />
          <MetricCard label="Overlap top3 (after)" value={afterMetrics.overlap_top3 ?? afterMetrics.lexical_overlap_ratio} />
          <MetricCard label="Max fused (before)" value={beforeMetrics.max_fused_score} />
          <MetricCard label="Max fused (after)" value={afterMetrics.max_fused_score} />
          <MetricCard label="Separation (before)" value={beforeMetrics.score_separation} />
          <MetricCard label="Separation (after)" value={afterMetrics.score_separation} />
          {deltas && (
            <>
              <MetricCard label="Δ max_fused" value={deltas.max_fused} delta />
              <MetricCard label="Δ overlap" value={deltas.lexical_overlap} delta />
              {deltas.overlap_top3 != null && <MetricCard label="Δ overlap_top3" value={deltas.overlap_top3} delta />}
              <MetricCard label="Δ separation" value={deltas.score_separation} delta />
            </>
          )}
        </div>

        {((typeof beforeMetrics?.max_vector_raw === 'number') || (typeof afterMetrics?.max_vector_raw === 'number')) && (
          <div className="card" style={{ marginTop: '0.75rem', padding: '0.75rem', fontSize: '0.8125rem' }}>
            <div style={{ color: 'var(--text-tertiary)', marginBottom: '0.375rem', fontSize: '0.75rem', fontWeight: 600 }}>Raw scores (pre-normalization)</div>
            <div style={{ color: 'var(--text-secondary)' }}>
              max_vector_raw: {typeof beforeMetrics?.max_vector_raw === 'number' ? beforeMetrics.max_vector_raw.toFixed(4) : '—'} → {typeof afterMetrics?.max_vector_raw === 'number' ? afterMetrics.max_vector_raw.toFixed(4) : '—'}
              {' · '}
              max_lexical_raw: {typeof beforeMetrics?.max_lexical_raw === 'number' ? beforeMetrics.max_lexical_raw.toFixed(4) : '—'} → {typeof afterMetrics?.max_lexical_raw === 'number' ? afterMetrics.max_lexical_raw.toFixed(4) : '—'}
            </div>
          </div>
        )}
      </div>

      {/* Retrieved tables */}
      <div style={{ marginBottom: '1.25rem' }}>
        <SectionTitle>Before (Attempt 1) — Top Retrieved</SectionTitle>
        <ChunkTable retrieved={beforeRetrieved} query={trace.query} />
      </div>

      {selfHealTriggered && attempts.length > 1 && (
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle>After (Attempt {bestAttemptNo}) — Top Retrieved</SectionTitle>
          <ChunkTable retrieved={afterRetrieved} query={trace.query} />
        </div>
      )}

      {/* Answer */}
      <div style={{ marginBottom: '1.25rem' }}>
        <SectionTitle>Final Answer</SectionTitle>
        <div style={{
          whiteSpace: 'pre-wrap',
          background: 'var(--bg-secondary)',
          padding: '1rem',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border)',
          fontSize: '0.875rem',
          lineHeight: 1.7,
        }}>
          {trace.answer ?? '—'}
        </div>
      </div>
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.625rem', color: 'var(--text-primary)' }}>{children}</h2>
  )
}

function PipelineStep({ label, color, textColor }: { label: string; color: string; textColor?: string }) {
  return (
    <span style={{
      padding: '0.25rem 0.625rem',
      background: color,
      borderRadius: 'var(--radius-sm)',
      fontSize: '0.75rem',
      fontWeight: 500,
      color: textColor || 'var(--text-primary)',
      whiteSpace: 'nowrap',
    }}>{label}</span>
  )
}

function Arrow() {
  return <span style={{ color: 'var(--text-tertiary)', fontSize: '0.75rem' }}>→</span>
}

function DeltaValue({ label, value }: { label: string; value?: number }) {
  const v = value ?? 0
  return (
    <span style={{ color: v > 0 ? 'var(--text-success)' : v < 0 ? 'var(--text-danger)' : 'var(--text-tertiary)' }}>
      {label}: {v > 0 ? '+' : ''}{v.toFixed(3)}
    </span>
  )
}

function MetricCard({ label, value, delta }: { label: string; value?: number; delta?: boolean }) {
  const v = value ?? 0
  return (
    <div className="card" style={{ padding: '0.625rem', background: delta ? 'var(--bg-success)' : undefined }}>
      <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '0.125rem' }}>{label}</div>
      <div style={{ fontSize: '1rem', fontWeight: 600, fontFamily: 'monospace' }}>{typeof v === 'number' ? v.toFixed(3) : '—'}</div>
    </div>
  )
}

function ChunkTable({ retrieved, query }: { retrieved: Array<{ chunk_id: string; doc_id?: string; fused_score?: number; snippet?: string; section_title?: string; chunk_type?: string }>; query: string }) {
  if (retrieved.length === 0) return <div style={{ color: 'var(--text-tertiary)', fontSize: '0.8125rem' }}>No chunks retrieved.</div>

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Chunk</th>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Section / Type</th>
            <th style={{ textAlign: 'right', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Score</th>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-tertiary)', fontWeight: 500, fontSize: '0.75rem' }}>Snippet</th>
          </tr>
        </thead>
        <tbody>
          {retrieved.slice(0, 10).map((r, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{r.chunk_id.slice(0, 8)}…</td>
              <td style={{ padding: '0.5rem', color: 'var(--text-tertiary)' }}>{r.section_title || '—'} / {r.chunk_type || '—'}</td>
              <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{(r.fused_score ?? 0).toFixed(3)}</td>
              <td style={{ padding: '0.5rem', maxWidth: 400 }}>
                <span dangerouslySetInnerHTML={{ __html: highlightQueryTokens(r.snippet ?? '', query) }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
