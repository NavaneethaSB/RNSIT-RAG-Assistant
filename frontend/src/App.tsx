import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { askAssistant, ChatResult, HistoryTurn } from './api'

type Message = { id: number; role: 'user' | 'assistant'; text: string; sources?: ChatResult['sources']; error?: boolean; retryQuestion?: string; retryUserId?: number }

const suggestions = [
  'What programs does RNSIT offer?',
  'Tell me about the MCA department',
  'What is the vision and mission of RNSIT?',
]

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState<number | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const requestRef = useRef<AbortController | null>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, loading])

  async function submitQuestion(value = question, retryUserId?: number) {
    const clean = value.trim()
    if (!clean || loading) return
    const userMessage: Message | undefined = retryUserId === undefined
      ? { id: Date.now(), role: 'user', text: clean }
      : undefined
    if (userMessage) setMessages(current => [...current, userMessage])
    setQuestion('')
    const controller = new AbortController()
    requestRef.current = controller
    setLoading(true)
    try {
      const conversation = messages.filter(message => !message.error && message.id !== retryUserId)
      const history: HistoryTurn[] = conversation
        .slice(-8)
        .map(message => ({ role: message.role, content: message.text }))
      const result = await askAssistant(clean, history, controller.signal)
      setMessages(current => [...current, { id: Date.now() + 1, role: 'assistant', text: result.answer, sources: result.sources }])
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      setMessages(current => [...current, {
        id: Date.now() + 1, role: 'assistant', text: error instanceof Error ? error.message : 'Something went wrong. Please try again.', error: true, retryQuestion: clean, retryUserId: userMessage?.id ?? retryUserId,
      }])
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null
        setLoading(false)
        inputRef.current?.focus()
      }
    }
  }

  function retryMessage(message: Message) {
    if (!message.retryQuestion || message.retryUserId === undefined || loading) return
    setMessages(current => current.filter(item => item.id !== message.id))
    void submitQuestion(message.retryQuestion, message.retryUserId)
  }

  function startNewChat() {
    requestRef.current?.abort()
    requestRef.current = null
    setMessages([])
    setQuestion('')
    setLoading(false)
    inputRef.current?.focus()
  }

  function onSubmit(event: FormEvent) { event.preventDefault(); void submitQuestion() }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submitQuestion() }
  }
  async function copyAnswer(message: Message) {
    try { await navigator.clipboard.writeText(message.text); setCopied(message.id); window.setTimeout(() => setCopied(null), 1600) } catch { setCopied(null) }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark" aria-hidden="true">R</div><div><strong>RNSIT</strong><span>AI ASSISTANT</span></div></div>
      <button className="new-chat" onClick={startNewChat} aria-label="Start a new chat"><span>＋</span> New conversation</button>
      <div className="sidebar-note"><span className="note-icon">✳</span><p>Your guide to RNS Institute of Technology.</p><small>Ask about programs, campus, departments, and more.</small></div>
      <div className="sidebar-bottom"><span className="online-dot" /> Connected to RNSIT knowledge base</div>
    </aside>

    <main className="main-panel">
      <header className="topbar"><div className="mobile-brand"><div className="brand-mark">R</div><strong>RNSIT AI</strong></div><div className="topbar-label"><span className="online-dot" /> RNSIT Knowledge Assistant</div><a className="website-link" href="https://www.rnsit.ac.in/" target="_blank" rel="noreferrer">Official website <span aria-hidden="true">↗</span></a><button className="mobile-new-chat" onClick={startNewChat} aria-label="Start a new conversation" title="New conversation">＋</button></header>

      <section className={`chat-area ${messages.length === 0 ? 'welcome-state' : ''}`} aria-label="Conversation">
        {messages.length === 0 ? <div className="welcome">
          <div className="welcome-icon" aria-hidden="true"><span>R</span><i>✳</i></div>
          <p className="eyebrow">YOUR CAMPUS, IN CONTEXT</p>
          <h1>How can I help you<br className="desktop-break" /> explore <em>RNSIT?</em></h1>
          <p className="welcome-copy">Ask a question about RNS Institute of Technology. I’ll find an answer in the college’s published information and share the source.</p>
          <div className="suggestion-grid" aria-label="Suggested questions">{suggestions.map((item, index) => <button key={item} className="suggestion" onClick={() => void submitQuestion(item)} disabled={loading}><span>{['⌕', '⌘', '◉'][index]}</span>{item}<b aria-hidden="true">↗</b></button>)}</div>
        </div> : <div className="thread">{messages.map(message => <article key={message.id} className={`message-row ${message.role}`}>
          <div className={`avatar ${message.role}`} aria-hidden="true">{message.role === 'user' ? 'Y' : <span>R</span>}</div>
          <div className="message-content"><div className="message-meta">{message.role === 'user' ? 'You' : 'RNSIT Assistant'}{message.error && <span className="error-label"> · Unable to respond</span>}</div>
            <div className={`message-bubble ${message.error ? 'error-bubble' : ''}`}>{message.role === 'assistant' && !message.error ? <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{a: ({ href, children }) => {
              const source = message.sources?.find(item => item.url === href)
              return source ? <a href={source.url} target="_blank" rel="noreferrer">{children}</a> : <span>{children}</span>
            }}}>{message.text}</ReactMarkdown></div> : <p>{message.text}</p>}</div>
            {message.error && message.retryQuestion && <button className="copy-button" onClick={() => retryMessage(message)} disabled={loading}>Retry</button>}
            {message.role === 'assistant' && !message.error && <>
              {message.sources && message.sources.length > 0 && <div className="sources"><div className="sources-title"><span>↗</span> SOURCE FOR THIS ANSWER</div><div className="source-list">{message.sources.slice(0, 1).map(source => <a className="source-card" key={source.id} href={source.url} target="_blank" rel="noreferrer"><span className="source-number">{source.id}</span><span className="source-text"><strong>{source.title}</strong><small>{source.url.replace(/^https?:\/\//, '')}</small><span className="source-excerpt">{source.excerpt}</span></span><span className="source-open" aria-hidden="true">↗</span></a>)}</div><p className="source-note">This official page supports the answer.</p></div>}
              <button className="copy-button" onClick={() => void copyAnswer(message)} aria-label="Copy answer">▢ {copied === message.id ? 'Copied' : 'Copy answer'}</button>
            </>}
          </div>
        </article>)}{loading && <div className="message-row assistant"><div className="avatar assistant" aria-hidden="true"><span>R</span></div><div className="message-content"><div className="message-meta">RNSIT Assistant</div><div className="loading-bubble" role="status" aria-label="Thinking"><span /><span /><span /><small>Searching RNSIT sources</small></div></div></div>}<div ref={bottomRef} /></div>}
      </section>

      <div className="composer-wrap"><form className="composer" onSubmit={onSubmit}><label htmlFor="question" className="sr-only">Ask a question about RNSIT</label><textarea ref={inputRef} id="question" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={onKeyDown} placeholder="Ask anything about RNSIT..." rows={1} maxLength={2000} disabled={loading} /><div className="composer-footer"><span>Answers are grounded in RNSIT sources</span><div><small>↵ Send</small><button type="submit" disabled={!question.trim() || loading} aria-label="Send message"><span>↑</span></button></div></div></form><p className="disclaimer">AI can make mistakes. Verify important details with the <a href="https://www.rnsit.ac.in/" target="_blank" rel="noreferrer">official RNSIT website</a>.</p></div>
    </main>
  </div>
}

export default App
