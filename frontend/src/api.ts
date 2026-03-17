const API_BASE = import.meta.env.VITE_API_URL || '';

export interface Citation {
  doc_id: string;
  chunk_id: string;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  trace_id: string;
  generator_provider?: string;
  conversation_id: string;
}

export interface MessageItem {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  trace_id?: string;
  generator_provider?: string;
  created_at?: string;
}

export interface ConversationItem {
  conversation_id: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  message_count: number;
  last_message?: string;
}

export interface ConversationDetail {
  conversation_id: string;
  title: string;
  messages: MessageItem[];
}

export interface DocItem {
  doc_id: string;
  title: string;
  source_type: string;
  source_ref: string | null;
  created_at: string | null;
  chunk_count: number;
}

export interface TraceAttemptRetrieved {
  chunk_id: string;
  doc_id: string;
  vector_score?: number;
  lexical_score?: number;
  fused_score?: number;
  snippet?: string;
  section_title?: string;
  chunk_type?: string;
}

export interface TraceAttempt {
  attempt_no: number;
  query_used: string;
  weights: { w_vec: number; w_lex: number };
  top_k?: number;
  retrieved: TraceAttemptRetrieved[];
  metrics: Record<string, number>;
  diagnoses: Array<{ label: string; reason: string }>;
  mode?: 'vague_summary' | 'normal';
}

export interface TraceHeal {
  type: string;
  params: Record<string, unknown>;
  reason: string;
  before_metrics?: Record<string, number>;
  after_metrics?: Record<string, number>;
  delta?: Record<string, number>;
}

export interface TraceItem {
  trace_id: string;
  created_at: string | null;
  query: string;
  embedding_provider: string | null;
  top_k: number | null;
  retrieved: Array<{ chunk_id: string; doc_id?: string; score?: number }>;
  retrieved_doc_ids?: string[];
  reranked: unknown[];
  selected: unknown[];
  citations: Array<{ doc_id: string; chunk_id: string; snippet: string }>;
  answer: string | null;
  latency_ms: number | null;
  attempts?: TraceAttempt[];
  heals_applied?: TraceHeal[];
  final_selected?: Array<{ chunk_id: string; score: number }>;
  self_heal_triggered?: boolean;
  trigger_failed_thresholds?: Array<{ name: string; value: number; threshold: number }>;
  best_attempt_no?: number;
  generator_provider?: string;
  generator_metadata?: Record<string, unknown>;
  generator_error?: string;
}

export async function health(): Promise<{ status: string }> {
  const r = await fetch(`${API_BASE}/api/health`);
  if (!r.ok) throw new Error('Health check failed');
  return r.json();
}

export async function chat(query: string, topK = 12, conversationId?: string): Promise<ChatResponse> {
  const r = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, conversation_id: conversationId || null }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || r.statusText);
  }
  return r.json();
}

export async function listConversations(): Promise<{ conversations: ConversationItem[] }> {
  const r = await fetch(`${API_BASE}/api/conversations`);
  if (!r.ok) throw new Error('Failed to list conversations');
  return r.json();
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const r = await fetch(`${API_BASE}/api/conversations/${id}`);
  if (!r.ok) throw new Error('Conversation not found');
  return r.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/conversations/${id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error('Failed to delete');
}

export async function ingestUrl(url: string, title?: string): Promise<{ doc_id: string; message: string }> {
  const r = await fetch(`${API_BASE}/api/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, title }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || r.statusText);
  }
  return r.json();
}

export async function ingestUpload(file: File): Promise<{ doc_id: string; message: string }> {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API_BASE}/api/ingest/upload`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || r.statusText);
  }
  return r.json();
}

export async function listDocs(): Promise<{ docs: DocItem[] }> {
  const r = await fetch(`${API_BASE}/api/docs`);
  if (!r.ok) throw new Error('Failed to list docs');
  return r.json();
}

export async function getTrace(traceId: string): Promise<TraceItem> {
  const r = await fetch(`${API_BASE}/api/trace/${traceId}`);
  if (!r.ok) throw new Error('Trace not found');
  return r.json();
}

export interface FailuresAnalytics {
  by_diagnosis_label: Record<string, number>;
  top_docs_by_failure_count: Array<{ doc_id: string; title: string; count: number }>;
  example_trace_ids_by_label: Record<string, string>;
}

export async function getFailuresAnalytics(): Promise<FailuresAnalytics> {
  const r = await fetch(`${API_BASE}/api/analytics/failures`);
  if (!r.ok) throw new Error('Failed to load analytics');
  return r.json();
}
