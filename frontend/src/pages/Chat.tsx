import { useState, useEffect, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  chat,
  listConversations,
  getConversation,
  deleteConversation,
  type Citation,
  type ConversationItem,
  type MessageItem,
} from '../api'

interface LocalMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  traceId?: string
  generatorProvider?: string
}

export default function Chat() {
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [activeConvId, setActiveConvId] = useState<string | null>(null)
  const [messages, setMessages] = useState<LocalMessage[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [convLoading, setConvLoading] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  const loadConversations = useCallback(async () => {
    try {
      const res = await listConversations()
      setConversations(res.conversations)
    } catch {
      /* ignore */
    } finally {
      setConvLoading(false)
    }
  }, [])

  useEffect(() => { loadConversations() }, [loadConversations])

  async function loadConversation(convId: string) {
    setActiveConvId(convId)
    setError(null)
    try {
      const detail = await getConversation(convId)
      const msgs: LocalMessage[] = detail.messages.map((m: MessageItem) => ({
        role: m.role,
        content: m.content,
        traceId: m.trace_id || undefined,
        generatorProvider: m.generator_provider || undefined,
      }))
      setMessages(msgs)
    } catch {
      setMessages([])
    }
    setTimeout(() => inputRef.current?.focus(), 100)
  }

  function startNewConversation() {
    setActiveConvId(null)
    setMessages([])
    setError(null)
    setQuery('')
    setTimeout(() => inputRef.current?.focus(), 100)
  }

  async function handleDeleteConversation(convId: string, e: React.MouseEvent) {
    e.stopPropagation()
    try {
      await deleteConversation(convId)
      if (activeConvId === convId) {
        startNewConversation()
      }
      loadConversations()
    } catch { /* ignore */ }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const q = query.trim()
    if (!q || loading) return

    setQuery('')
    setError(null)

    const userMsg: LocalMessage = { role: 'user', content: q }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await chat(q, 12, activeConvId || undefined)

      if (!activeConvId) {
        setActiveConvId(res.conversation_id)
      }

      const assistantMsg: LocalMessage = {
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        traceId: res.trace_id,
        generatorProvider: res.generator_provider,
      }
      setMessages(prev => [...prev, assistantMsg])
      loadConversations()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  function getProviderBadge(provider?: string) {
    if (!provider) return null
    const isOllama = provider === 'ollama'
    const isFallback = provider === 'extractive_fallback'
    return (
      <span className={`badge ${isFallback ? 'badge-warning' : isOllama ? 'badge-success' : 'badge-accent'}`}>
        <span style={{
          width: 6, height: 6, borderRadius: '50%',
          background: isFallback ? 'var(--text-warning)' : isOllama ? 'var(--text-success)' : 'var(--text-accent)',
          display: 'inline-block',
        }} />
        {isOllama ? 'Ollama' : isFallback ? 'Fallback' : 'Extractive'}
      </span>
    )
  }

  function formatTime(date?: string) {
    if (!date) return ''
    const d = new Date(date)
    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffMin = Math.floor(diffMs / 60000)
    if (diffMin < 1) return 'Just now'
    if (diffMin < 60) return `${diffMin}m ago`
    const diffHr = Math.floor(diffMin / 60)
    if (diffHr < 24) return `${diffHr}h ago`
    const diffDay = Math.floor(diffHr / 24)
    if (diffDay < 7) return `${diffDay}d ago`
    return d.toLocaleDateString()
  }

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Conversation sidebar */}
      <div style={{
        width: 280,
        minWidth: 280,
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-primary)',
      }}>
        <div style={{
          padding: '1rem 1rem 0.75rem',
          borderBottom: '1px solid var(--border)',
        }}>
          <button
            className="btn-primary"
            onClick={startNewConversation}
            style={{ width: '100%', padding: '0.625rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
          >
            <span style={{ fontSize: '1.125rem' }}>+</span>
            New Chat
          </button>
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '0.5rem' }}>
          {convLoading ? (
            <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '0.8125rem' }}>
              Loading...
            </div>
          ) : conversations.length === 0 ? (
            <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '0.8125rem' }}>
              No conversations yet.
              <br />Start a new chat below.
            </div>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.conversation_id}
                onClick={() => loadConversation(conv.conversation_id)}
                style={{
                  padding: '0.625rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  background: activeConvId === conv.conversation_id ? 'var(--bg-tertiary)' : 'transparent',
                  marginBottom: '0.125rem',
                  transition: 'background var(--transition)',
                  display: 'flex',
                  flexDirection: 'column',
                  position: 'relative',
                }}
                onMouseEnter={e => (e.currentTarget.querySelector('.del-btn') as HTMLElement)?.style.setProperty('opacity', '1')}
                onMouseLeave={e => (e.currentTarget.querySelector('.del-btn') as HTMLElement)?.style.setProperty('opacity', '0')}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                  <div style={{
                    fontSize: '0.8125rem',
                    fontWeight: activeConvId === conv.conversation_id ? 600 : 400,
                    color: 'var(--text-primary)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    flex: 1,
                  }}>
                    {conv.title}
                  </div>
                  <button
                    className="del-btn"
                    onClick={(e) => handleDeleteConversation(conv.conversation_id, e)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--text-tertiary)',
                      cursor: 'pointer',
                      fontSize: '0.75rem',
                      padding: '0 0.25rem',
                      opacity: 0,
                      transition: 'opacity var(--transition)',
                      flexShrink: 0,
                    }}
                    title="Delete"
                  >
                    ✕
                  </button>
                </div>
                <div style={{
                  fontSize: '0.6875rem',
                  color: 'var(--text-tertiary)',
                  marginTop: '0.125rem',
                  display: 'flex',
                  gap: '0.5rem',
                  alignItems: 'center',
                }}>
                  <span>{conv.message_count} msgs</span>
                  <span>{formatTime(conv.updated_at)}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Messages */}
        <div style={{
          flex: 1,
          overflow: 'auto',
          padding: '1.5rem 0',
        }}>
          {messages.length === 0 && !loading ? (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'var(--text-tertiary)',
              gap: '1rem',
            }}>
              <div style={{
                width: 56,
                height: 56,
                borderRadius: 'var(--radius-lg)',
                background: 'linear-gradient(135deg, rgba(37,99,235,0.15), rgba(124,58,237,0.15))',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem',
              }}>
                💡
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.375rem' }}>
                  Ask about your documents
                </div>
                <div style={{ fontSize: '0.8125rem', maxWidth: 360, lineHeight: 1.6 }}>
                  Your questions are answered with grounded citations from your ingested data. Self-healing retrieval ensures high quality results.
                </div>
              </div>
            </div>
          ) : (
            <div style={{ maxWidth: 720, margin: '0 auto', padding: '0 1.5rem' }}>
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className="fade-in"
                  style={{
                    marginBottom: '1.25rem',
                    display: 'flex',
                    gap: '0.75rem',
                    alignItems: 'flex-start',
                  }}
                >
                  <div style={{
                    width: 32,
                    height: 32,
                    borderRadius: 'var(--radius-sm)',
                    background: msg.role === 'user'
                      ? 'var(--bg-accent)'
                      : 'linear-gradient(135deg, #059669, #0d9488)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: 'white',
                    flexShrink: 0,
                    marginTop: '0.125rem',
                  }}>
                    {msg.role === 'user' ? 'U' : 'A'}
                  </div>

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      marginBottom: '0.375rem',
                    }}>
                      <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {msg.role === 'user' ? 'You' : 'AdaptiveRAG'}
                      </span>
                      {msg.role === 'assistant' && getProviderBadge(msg.generatorProvider)}
                    </div>

                    <div style={{
                      fontSize: '0.875rem',
                      lineHeight: 1.7,
                      color: 'var(--text-primary)',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}>
                      {msg.content}
                    </div>

                    {msg.role === 'assistant' && msg.generatorProvider === 'extractive_fallback' && (
                      <div className="badge badge-warning" style={{ marginTop: '0.5rem' }}>
                        Ollama unavailable — used extractive fallback
                      </div>
                    )}

                    {msg.citations && msg.citations.length > 0 && (
                      <div style={{ marginTop: '0.75rem' }}>
                        <div style={{
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          color: 'var(--text-tertiary)',
                          textTransform: 'uppercase',
                          letterSpacing: '0.05em',
                          marginBottom: '0.375rem',
                        }}>
                          Sources ({msg.citations.length})
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
                          {msg.citations.map((c, ci) => (
                            <div key={ci} style={{
                              padding: '0.5rem 0.75rem',
                              background: 'var(--bg-secondary)',
                              border: '1px solid var(--border)',
                              borderRadius: 'var(--radius-sm)',
                              fontSize: '0.8125rem',
                            }}>
                              <div style={{
                                fontSize: '0.6875rem',
                                color: 'var(--text-tertiary)',
                                fontFamily: 'monospace',
                                marginBottom: '0.25rem',
                              }}>
                                {c.chunk_id.slice(0, 12)}… · doc {c.doc_id.slice(0, 8)}…
                              </div>
                              <div style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>{c.snippet}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {msg.traceId && (
                      <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.75rem', fontSize: '0.75rem' }}>
                        <Link to={`/trace/${msg.traceId}`}>View trace</Link>
                        <Link to={`/trace/${msg.traceId}/report`}>Self-heal report</Link>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="fade-in" style={{
                  marginBottom: '1.25rem',
                  display: 'flex',
                  gap: '0.75rem',
                  alignItems: 'flex-start',
                }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: 'var(--radius-sm)',
                    background: 'linear-gradient(135deg, #059669, #0d9488)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.75rem', fontWeight: 700, color: 'white', flexShrink: 0,
                  }}>A</div>
                  <div style={{ paddingTop: '0.5rem' }}>
                    <div style={{ display: 'flex', gap: '0.375rem' }}>
                      <span className="loading-dot" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--text-tertiary)', display: 'inline-block' }} />
                      <span className="loading-dot" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--text-tertiary)', display: 'inline-block' }} />
                      <span className="loading-dot" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--text-tertiary)', display: 'inline-block' }} />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {error && (
          <div style={{
            padding: '0.5rem 1rem',
            background: 'var(--bg-danger)',
            color: 'var(--text-danger)',
            fontSize: '0.8125rem',
            textAlign: 'center',
          }}>
            {error}
          </div>
        )}

        {/* Input area */}
        <div style={{
          borderTop: '1px solid var(--border)',
          padding: '1rem 1.5rem',
          background: 'var(--bg-secondary)',
        }}>
          <form onSubmit={handleSubmit} style={{ maxWidth: 720, margin: '0 auto', display: 'flex', gap: '0.625rem', alignItems: 'flex-end' }}>
            <div style={{ flex: 1, position: 'relative' }}>
              <textarea
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your documents... (Enter to send, Shift+Enter for new line)"
                rows={1}
                style={{
                  width: '100%',
                  resize: 'none',
                  minHeight: 44,
                  maxHeight: 160,
                  padding: '0.625rem 0.875rem',
                  fontSize: '0.875rem',
                  lineHeight: 1.5,
                }}
              />
            </div>
            <button
              type="submit"
              className="btn-primary"
              disabled={loading || !query.trim()}
              style={{
                height: 44,
                padding: '0 1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.375rem',
                flexShrink: 0,
              }}
            >
              {loading ? 'Thinking...' : 'Send'}
              {!loading && <span style={{ fontSize: '1rem' }}>→</span>}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
