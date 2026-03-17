import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Chat from './pages/Chat'
import Ingest from './pages/Ingest'
import Docs from './pages/Docs'
import Trace from './pages/Trace'
import TraceReport from './pages/TraceReport'
import FailuresAnalytics from './pages/FailuresAnalytics'

const NAV_ITEMS = [
  { to: '/', label: 'Chat', icon: '💬' },
  { to: '/ingest', label: 'Ingest', icon: '📥' },
  { to: '/docs', label: 'Documents', icon: '📄' },
  { to: '/analytics/failures', label: 'Analytics', icon: '📊' },
]

function Sidebar() {
  const location = useLocation()

  return (
    <aside style={{
      width: 220,
      minWidth: 220,
      height: '100vh',
      position: 'sticky',
      top: 0,
      background: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '1.25rem 0',
      zIndex: 10,
    }}>
      <Link to="/" style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.625rem',
        padding: '0 1.25rem',
        marginBottom: '2rem',
        textDecoration: 'none',
      }}>
        <div style={{
          width: 32,
          height: 32,
          borderRadius: 'var(--radius-sm)',
          background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '0.875rem',
          fontWeight: 700,
          color: 'white',
        }}>A</div>
        <div>
          <div style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.025em' }}>AdaptiveRAG</div>
          <div style={{ fontSize: '0.625rem', color: 'var(--text-tertiary)', fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Self-Healing AI</div>
        </div>
      </Link>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', padding: '0 0.75rem' }}>
        {NAV_ITEMS.map(item => {
          const active = item.to === '/'
            ? location.pathname === '/'
            : location.pathname.startsWith(item.to)
          return (
            <Link
              key={item.to}
              to={item.to}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.625rem',
                padding: '0.5rem 0.625rem',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.8125rem',
                fontWeight: active ? 600 : 400,
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                background: active ? 'var(--bg-tertiary)' : 'transparent',
                textDecoration: 'none',
                transition: 'all var(--transition)',
              }}
            >
              <span style={{ fontSize: '1rem', width: 20, textAlign: 'center' }}>{item.icon}</span>
              {item.label}
            </Link>
          )
        })}
      </nav>

      <div style={{ flex: 1 }} />

      <div style={{
        padding: '0.75rem 1.25rem',
        borderTop: '1px solid var(--border)',
        fontSize: '0.6875rem',
        color: 'var(--text-tertiary)',
        lineHeight: 1.5,
      }}>
        <div>Local-first RAG system</div>
        <div>Ollama + Self-Healing</div>
      </div>
    </aside>
  )
}

export default function App() {
  const location = useLocation()
  const isChatPage = location.pathname === '/'

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar />
      <main style={{
        flex: 1,
        minWidth: 0,
        height: isChatPage ? '100vh' : 'auto',
        overflow: isChatPage ? 'hidden' : 'auto',
      }}>
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/ingest" element={<Ingest />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/trace/:traceId" element={<Trace />} />
          <Route path="/trace/:traceId/report" element={<TraceReport />} />
          <Route path="/analytics/failures" element={<FailuresAnalytics />} />
        </Routes>
      </main>
    </div>
  )
}
