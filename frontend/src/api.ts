export type Source = { id: string; title: string; url: string; excerpt: string; retrieval_rank: number; retrieval_score: number | null }
export type ChatResult = { answer: string; sources: Source[] }
export type HistoryTurn = { role: 'user' | 'assistant'; content: string }

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export async function askAssistant(question: string, conversation: HistoryTurn[] = [], signal?: AbortSignal): Promise<ChatResult> {
  let response: Response
  try {
    response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, conversation: conversation.slice(-8) }), signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new Error('Could not reach the assistant. Check that the API is running, then try again.')
  }

  if (!response.ok) {
    let message = 'The assistant could not process your question. Please try again.'
    try {
      const body = await response.json() as { detail?: string }
      if (typeof body.detail === 'string') message = body.detail
    } catch { /* Keep the friendly fallback when the server response is not JSON. */ }
    throw new Error(message)
  }

  const result = await response.json() as Partial<ChatResult>
  if (typeof result.answer !== 'string' || !Array.isArray(result.sources)) {
    throw new Error('The assistant returned an unexpected response. Please try again.')
  }
  return { answer: result.answer, sources: result.sources.filter((source): source is Source =>
    !!source && typeof source.id === 'string' && typeof source.title === 'string' &&
    typeof source.url === 'string' && typeof source.excerpt === 'string' &&
    typeof source.retrieval_rank === 'number' &&
    (typeof source.retrieval_score === 'number' || source.retrieval_score === null)) }
}
