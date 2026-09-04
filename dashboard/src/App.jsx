import { useState, useEffect, useCallback } from 'react';
import { api } from './api/client';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend, Area, AreaChart, ReferenceArea
} from 'recharts';

/* ═══════════════════════════════════════════════════════════════════
   RiskGuard AI Dashboard — Single-file React application
   ═══════════════════════════════════════════════════════════════════ */

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'incidents', label: 'Incidents' },
  { id: 'threshold', label: 'Threshold Lab' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'evaluation', label: 'Evaluation Matrix' },
];

function formatCurrency(val) {
  if (val >= 10000000) return `₹${(val / 10000000).toFixed(2)}Cr`;
  if (val >= 100000) return `₹${(val / 100000).toFixed(2)}L`;
  if (val >= 1000) return `₹${(val / 1000).toFixed(1)}K`;
  return `₹${val.toFixed(0)}`;
}

function formatNumber(val) {
  if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
  if (val >= 1000) return `${(val / 1000).toFixed(1)}K`;
  return val.toString();
}

function RiskBadge({ level }) {
  const l = (level || '').toLowerCase();
  return <span className={`risk-badge ${l}`}>● {level}</span>;
}

function MockBanner({ mode, reason }) {
  if (mode === 'live') return null;
  if (reason === 'rate_limited') {
    return (
      <div className="mock-banner" style={{ background: 'rgba(245, 158, 11, 0.15)', borderColor: 'rgba(245, 158, 11, 0.4)', color: '#fbbf24' }}>
        <span className="icon">⚠️</span>
        <span>
          <strong>MOCK MODE — Live agent rate-limited (free-tier quota).</strong> Try again in a minute, or this is expected if multiple investigations were run in quick succession.
        </span>
      </div>
    );
  }
  return (
    <div className="mock-banner">
      <span className="icon">⚠️</span>
      <span><strong>MOCK MODE</strong> — This investigation was generated without an AI agent. Do not treat this as a real investigation.</span>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   APP ROOT
   ═══════════════════════════════════════════════════════════════════ */
export default function App() {
  const [tab, setTab] = useState('overview');
  const [selectedIncident, setSelectedIncident] = useState(null);

  return (
    <div className="app">
      <header className="header">
        <div className="header-title">
          <h1>🛡️ RiskGuard AI</h1>
          <span className="subtitle">Agentic Risk Intelligence Platform</span>
        </div>
        <span className="header-badge">Razorpay AI Buildathon</span>
      </header>

      <main className="main-content">
        <nav className="nav-tabs">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`nav-tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => { setTab(t.id); setSelectedIncident(null); }}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {tab === 'overview' && <OverviewView />}
        {tab === 'incidents' && (
          selectedIncident
            ? <IncidentDetail id={selectedIncident} onBack={() => setSelectedIncident(null)} />
            : <IncidentList onSelect={setSelectedIncident} />
        )}
        {tab === 'threshold' && <ThresholdLab />}
        {tab === 'benchmarks' && <BenchmarksView />}
        {tab === 'evaluation' && <EvaluationView />}
      </main>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   OVERVIEW VIEW
   ═══════════════════════════════════════════════════════════════════ */
function OverviewView() {
  const [metrics, setMetrics] = useState(null);
  const [trend, setTrend] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.getDashboardMetrics(), api.getRiskTrend()])
      .then(([m, t]) => { setMetrics(m); setTrend(t.data || []); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!metrics) return <div className="loading">Loading dashboard</div>;

  return (
    <>
      <div className="metrics-grid">
        <div className="glass-card metric-card blue">
          <div className="card-title">Total Transactions</div>
          <div className="metric-value">{formatNumber(metrics.total_transactions)}</div>
          <div className="metric-label">in test period</div>
        </div>
        <div className="glass-card metric-card red">
          <div className="card-title">Active Alerts</div>
          <div className="metric-value">{metrics.active_alerts}</div>
          <div className="metric-label">incidents detected</div>
        </div>
        <div className="glass-card metric-card amber">
          <div className="card-title">Potential Exposure</div>
          <div className="metric-value">{formatCurrency(metrics.total_exposure)}</div>
          <div className="metric-label">identified, not claimed prevented</div>
        </div>
        <div className="glass-card metric-card green">
          <div className="card-title">Current Risk Rate</div>
          <div className="metric-value">{(metrics.current_risk_rate * 100).toFixed(2)}%</div>
          <div className="metric-label">of transactions flagged</div>
        </div>
        <div className="glass-card metric-card purple">
          <div className="card-title">Best Model PR-AUC</div>
          <div className="metric-value">{metrics.model_pr_auc?.toFixed(4) || '—'}</div>
          <div className="metric-label">{metrics.best_model}</div>
        </div>
      </div>

      <div className="glass-card chart-container">
        <div className="card-title">Risk Rate Trend (Test Period)</div>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trend}>
              <defs>
                <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={d => d?.slice(5)} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(1)}%`} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(v) => [`${(v * 100).toFixed(2)}%`, 'Risk Rate']}
              />
              <Area type="monotone" dataKey="risk_rate" stroke="#3b82f6" fill="url(#riskGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   INCIDENT LIST
   ═══════════════════════════════════════════════════════════════════ */
function IncidentList({ onSelect }) {
  const [incidents, setIncidents] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getIncidents()
      .then(d => setIncidents(d.incidents || []))
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!incidents.length) return <div className="loading">Loading incidents</div>;

  return (
    <div className="glass-card">
      <div className="card-title">Active Incidents</div>
      <table className="incident-table">
        <thead>
          <tr>
            <th>Incident ID</th>
            <th>Merchant</th>
            <th>Trigger</th>
            <th>Risk Level</th>
            <th>Exposure</th>
            <th>Confidence</th>
            <th>Affected</th>
          </tr>
        </thead>
        <tbody>
          {incidents.map(inc => (
            <tr key={inc.incident_id} className="data-row" onClick={() => onSelect(inc.incident_id)}>
              <td style={{ color: '#3b82f6', fontWeight: 600 }}>{inc.incident_id}</td>
              <td>{inc.merchant_id}</td>
              <td>{inc.trigger_type}</td>
              <td><RiskBadge level={inc.risk_level} /></td>
              <td>{formatCurrency(inc.estimated_exposure)}</td>
              <td>{(inc.confidence * 100).toFixed(0)}%</td>
              <td>{inc.affected_payment_count} txns</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   INCIDENT DETAIL
   ═══════════════════════════════════════════════════════════════════ */
function IncidentDetail({ id, onBack }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [investigateError, setInvestigateError] = useState(null);
  const [cooldown, setCooldown] = useState(0);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setInterval(() => {
      setCooldown(c => {
        if (c <= 1) {
          clearInterval(timer);
          return 0;
        }
        return c - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [cooldown]);

  useEffect(() => {
    api.getIncidentDetail(id)
      .then(setData)
      .catch(e => setError(e.message));
  }, [id]);

  if (error) return <><button className="back-btn" onClick={onBack}>← Back</button><ErrorState message={error} /></>;
  if (!data) return <div className="loading">Loading investigation</div>;

  const { incident, investigation, exposure, agent_mode, mock_reason } = data;
  const currentMockReason = mock_reason || investigation?.mock_reason;

  const handleInvestigate = async () => {
    setInvestigating(true);
    setInvestigateError(null);
    try {
      const report = await api.triggerInvestigation(id);
      setData(prev => ({
        ...prev,
        investigation: report,
        exposure: report.exposure || prev.exposure,
        agent_mode: report.agent_mode || 'mock',
        mock_reason: report.mock_reason,
      }));
      setCooldown(20);
    } catch (err) {
      setInvestigateError(err.message || 'Investigation failed');
    } finally {
      setInvestigating(false);
    }
  };

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
        <button className="back-btn" style={{ marginBottom: 0 }} onClick={onBack}>← Back to Incidents</button>
        <button
          className="investigate-btn"
          onClick={handleInvestigate}
          disabled={investigating || cooldown > 0}
          title={cooldown > 0 ? `Please wait ${cooldown}s cooldown before next investigation` : 'Investigate incident'}
        >
          {investigating ? (
            <>
              <span style={{ display: 'inline-block', animation: 'spin 1.2s linear infinite' }}>⚙️</span> Investigating with AI Agent...
            </>
          ) : cooldown > 0 ? (
            <>⏳ Cooldown ({cooldown}s)</>
          ) : (
            <>🔍 Investigate Incident</>
          )}
        </button>
      </div>

      {investigateError && (
        <div style={{
          padding: '12px 16px',
          borderRadius: 8,
          background: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#f87171',
          marginBottom: 16,
          fontSize: '0.85rem',
        }}>
          ⚠️ Investigation error: {investigateError}
        </div>
      )}

      {investigating && (
        <div style={{
          padding: '16px 20px',
          borderRadius: 12,
          background: 'rgba(59, 130, 246, 0.1)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          color: '#60a5fa',
          marginBottom: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          fontSize: '0.9rem',
        }}>
          <div style={{ animation: 'spin 1.5s linear infinite', fontSize: '1.25rem' }}>⚙️</div>
          <div>
            <strong>AI Agent Investigation in Progress...</strong>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
              Gemini model is querying evidence tools in sequence (transactions, velocity windows, device clusters, baseline deviation) to synthesize findings.
            </div>
          </div>
        </div>
      )}

      <MockBanner mode={agent_mode} reason={currentMockReason} />

      <div className="detail-grid">
        <div className="glass-card">
          <div className="card-title">Incident Summary</div>
          <div style={{ display: 'grid', gap: '10px' }}>
            <div><strong>ID:</strong> {incident.incident_id}</div>
            <div><strong>Merchant:</strong> {incident.merchant_id}</div>
            <div><strong>Trigger:</strong> {incident.trigger_type}</div>
            <div><strong>Risk Level:</strong> <RiskBadge level={incident.risk_level} /></div>
            <div><strong>Risk Increase:</strong> {incident.risk_increase_pct?.toFixed(1)}%</div>
            <div><strong>Confidence:</strong> {(incident.confidence * 100).toFixed(0)}%</div>
            <div><strong>Affected Payments:</strong> {incident.affected_payment_count}</div>
          </div>
        </div>

        <div className="glass-card">
          <div className="card-title">Financial Exposure</div>
          <div style={{ display: 'grid', gap: '10px' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Potential Exposure</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-amber)' }}>
                {formatCurrency(exposure?.potential_exposure || incident.estimated_exposure)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>High-Confidence Exposure</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-red)' }}>
                {formatCurrency(exposure?.high_confidence_exposure || incident.high_confidence_exposure)}
              </div>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Note: These figures represent potential exposure identified, not fraud prevented.
            </div>
          </div>
        </div>
      </div>

      {incident.flagged_transaction_ids?.length > 0 && (
        <div className="glass-card" style={{ marginBottom: 16 }}>
          <div className="card-title">Flagged Transactions (sample)</div>
          <ul className="evidence-list">
            {incident.flagged_transaction_ids.slice(0, 8).map(tid => (
              <li key={tid} className="evidence-item">{tid}</li>
            ))}
          </ul>
        </div>
      )}

      {incident.related_entity_ids?.length > 0 && (
        <div className="glass-card" style={{ marginBottom: 16 }}>
          <div className="card-title">Related Entities</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {incident.related_entity_ids.slice(0, 12).map(eid => (
              <span key={eid} style={{
                padding: '4px 10px', borderRadius: 20, background: 'rgba(59,130,246,0.1)',
                color: '#3b82f6', fontSize: '0.75rem', border: '1px solid rgba(59,130,246,0.2)'
              }}>{eid}</span>
            ))}
          </div>
        </div>
      )}

      {/* AI Investigation Narrative — grayed out in mock mode */}
      {investigation && (
        <div className="glass-card" style={{
          marginBottom: 16,
          ...(agent_mode !== 'live' ? { opacity: 0.5, border: '1px dashed rgba(245,158,11,0.4)' } : {}),
        }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            AI Investigation Narrative
            {agent_mode !== 'live' ? (
              <span style={{
                fontSize: '0.65rem', padding: '2px 8px', borderRadius: 12,
                background: 'rgba(245,158,11,0.15)', color: '#f59e0b',
                border: '1px solid rgba(245,158,11,0.3)', fontWeight: 600,
              }}>PLACEHOLDER — NOT A REAL INVESTIGATION</span>
            ) : (
              <span style={{
                fontSize: '0.65rem', padding: '2px 8px', borderRadius: 12,
                background: 'rgba(16,185,129,0.15)', color: '#10b981',
                border: '1px solid rgba(16,185,129,0.3)', fontWeight: 600,
              }}>LIVE GEMINI AGENT</span>
            )}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
            {investigation.summary || investigation.narrative || 'Investigation narrative will appear here when agent runs.'}
          </div>
          {(investigation.recommendations || investigation.recommendation) && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Recommendations</div>
              <ul style={{ margin: 0, paddingLeft: 16 }}>
                {(Array.isArray(investigation.recommendations)
                  ? investigation.recommendations
                  : [investigation.recommendations || investigation.recommendation]
                ).filter(Boolean).map((r, i) => (
                  <li key={i} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {investigation.evidence && investigation.evidence.length > 0 && (
            <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 12 }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8 }}>
                Agent Tool Execution Audit Trail ({investigation.evidence.length} tool calls)
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {investigation.evidence.map((ev, i) => (
                  <span key={i} style={{
                    padding: '3px 8px', borderRadius: 6,
                    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                    fontSize: '0.72rem', color: '#94a3b8', fontFamily: 'monospace'
                  }}>
                    🔧 {ev.tool}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   THRESHOLD LAB
   ═══════════════════════════════════════════════════════════════════ */
const DEFAULT_THRESHOLD = 0.33;

function ThresholdLab() {
  const [data, setData] = useState([]);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [error, setError] = useState(null);

  useEffect(() => {
    setThreshold(DEFAULT_THRESHOLD);
    api.getThresholdLab()
      .then(d => {
        setData(d.thresholds || []);
        if (d.default_threshold !== undefined && d.default_threshold !== null) {
          setThreshold(d.default_threshold);
        } else {
          setThreshold(DEFAULT_THRESHOLD);
        }
      })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!data.length) return <div className="loading">Loading threshold data</div>;

  // Find closest point to current threshold
  const closest = data.reduce((prev, curr) =>
    Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
  );

  return (
    <>
      <div className="glass-card" style={{ marginBottom: 16, textAlign: 'center' }}>
        <div className="card-title">Decision Threshold</div>
        <div className="threshold-value">{threshold.toFixed(2)}</div>
        <input
          type="range" min="0.01" max="0.99" step="0.01"
          value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          className="threshold-slider"
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>More Alerts (0.01)</span>
          <span>Fewer Alerts (0.99)</span>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="glass-card metric-card green">
          <div className="card-title">Precision</div>
          <div className="metric-value">{(closest.precision * 100).toFixed(1)}%</div>
          <div className="metric-label">of flagged are truly risky</div>
        </div>
        <div className="glass-card metric-card blue">
          <div className="card-title">Recall</div>
          <div className="metric-value">{(closest.recall * 100).toFixed(1)}%</div>
          <div className="metric-label">of risk caught</div>
        </div>
        <div className="glass-card metric-card purple">
          <div className="card-title">F1 Score</div>
          <div className="metric-value">{(closest.f1 * 100).toFixed(1)}%</div>
          <div className="metric-label">harmonic mean</div>
        </div>
        <div className="glass-card metric-card amber">
          <div className="card-title">Expected Loss</div>
          <div className="metric-value">{formatCurrency(closest.expected_loss)}</div>
          <div className="metric-label">FP cost + FN cost</div>
        </div>
      </div>

      <div className="two-col">
        <div className="glass-card metric-card red">
          <div className="card-title">False Positives</div>
          <div className="metric-value">{formatNumber(closest.fp_count)}</div>
          <div className="metric-label">legitimate tx flagged</div>
        </div>
        <div className="glass-card metric-card amber">
          <div className="card-title">False Negatives</div>
          <div className="metric-value">{formatNumber(closest.fn_count)}</div>
          <div className="metric-label">risky tx missed</div>
        </div>
      </div>

      <div className="glass-card chart-container" style={{ marginTop: 16 }}>
        <div className="card-title">Precision / Recall / F1 vs Threshold</div>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="threshold" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 1]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2} dot={false} name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="#3b82f6" strokeWidth={2} dot={false} name="Recall" />
              <Line type="monotone" dataKey="f1" stroke="#8b5cf6" strokeWidth={2} dot={false} name="F1" />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   BENCHMARKS VIEW
   ═══════════════════════════════════════════════════════════════════ */
function BenchmarksView() {
  const [cost, setCost] = useState([]);
  const [adversarial, setAdversarial] = useState(null);
  const [drift, setDrift] = useState([]);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      api.getCostOfInaction().catch(() => ({ conditions: [] })),
      api.getAdversarial().catch(() => null),
      api.getDrift().catch(() => ({ days: [] })),
      api.getLatency().catch(() => null),
    ]).then(([c, a, d, l]) => {
      setCost(c.conditions || []);
      setAdversarial(a);
      setDrift(d.days || []);
      setLatency(l);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;

  const maxLoss = cost.length ? Math.max(...cost.map(c => c.total_loss)) : 1;
  const barColors = ['#ef4444', '#f59e0b', '#10b981'];

  return (
    <>
      {/* Cost of Inaction */}
      <div className="glass-card" style={{ marginBottom: 16 }}>
        <div className="card-title">Cost of Inaction — Expected Loss Comparison</div>
        {cost.length > 0 ? (
          <div className="bar-chart-container">
            {cost.map((c, i) => (
              <div key={c.name} className="bar-item">
                <div className="bar-value">{formatCurrency(c.total_loss)}</div>
                <div className="bar" style={{
                  height: `${Math.max(20, (c.total_loss / maxLoss) * 200)}px`,
                  background: `linear-gradient(180deg, ${barColors[i]}, ${barColors[i]}80)`,
                }} />
                <div className="bar-label">{c.name}</div>
                <div className="bar-label" style={{ fontSize: '0.65rem' }}>
                  {c.detected_fraud_pct}% detected
                </div>
              </div>
            ))}
          </div>
        ) : <div className="loading">Loading</div>}
      </div>

      {/* Adversarial Comparison */}
      {adversarial && (
        <div className="glass-card" style={{ marginBottom: 16 }}>
          <div className="card-title">Adversarial Robustness — Naive Rules vs RiskGuard</div>
          <table className="incident-table">
            <thead>
              <tr><th>Scenario</th><th>Naive Rule Detection</th><th>RiskGuard Detection</th></tr>
            </thead>
            <tbody>
              {Object.keys(adversarial.naive_rule_detection_rate || {}).sort().map(scenario => (
                <tr key={scenario} className="data-row">
                  <td style={{ fontWeight: 500 }}>{scenario}</td>
                  <td>{(adversarial.naive_rule_detection_rate[scenario] * 100).toFixed(1)}%</td>
                  <td style={{ color: adversarial.riskguard_detection_rate[scenario] > adversarial.naive_rule_detection_rate[scenario] ? '#10b981' : 'inherit' }}>
                    {(adversarial.riskguard_detection_rate[scenario] * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
              <tr className="data-row">
                <td style={{ fontWeight: 600 }}>False Positive Rate</td>
                <td>{(adversarial.naive_rule_fp_rate * 100).toFixed(2)}%</td>
                <td>{(adversarial.riskguard_fp_rate * 100).toFixed(2)}%</td>
              </tr>
              <tr className="data-row">
                <td style={{ fontWeight: 600 }}>Exposure Missed</td>
                <td>{formatCurrency(adversarial.exposure_missed_naive)}</td>
                <td style={{ color: '#10b981' }}>{formatCurrency(adversarial.exposure_missed_riskguard)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Drift Timeline */}
      {drift.length > 0 && (
        <div className="glass-card" style={{ marginBottom: 16 }}>
          <div className="card-title">Drift Re-baselining Timeline</div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={drift} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <ReferenceArea x1={1} x2={5} fill="#ef4444" fillOpacity={0.06} />
                <ReferenceArea x1={5} x2={10} fill="#f59e0b" fillOpacity={0.06} />
                <ReferenceArea x1={10} x2={20} fill="#10b981" fillOpacity={0.06} />
                <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} label={{ value: 'Day', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 11 }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                  formatter={(v, name) => [v.toFixed(4), name]} />
                <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: 8, fontSize: '0.75rem' }} />
                <Line type="monotone" dataKey="baseline_risk_rate" stroke="#3b82f6" strokeWidth={2} dot={false} name="Baseline" />
                <Line type="monotone" dataKey="current_risk_rate" stroke="#ef4444" strokeWidth={2} dot={false} name="Current" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            {[
              { status: 'CRITICAL', label: 'CRITICAL (Days 1–5)', color: '#ef4444' },
              { status: 'MONITOR', label: 'MONITOR (Days 6–10)', color: '#f59e0b' },
              { status: 'NEW_NORMAL', label: 'NEW NORMAL (Days 11–20)', color: '#10b981' },
            ].map(b => (
              <span key={b.status} style={{
                fontSize: '0.7rem', padding: '4px 12px', borderRadius: 20,
                background: b.color + '18',
                color: b.color,
                border: `1px solid ${b.color}33`,
                fontWeight: 500,
              }}>{b.label}</span>
            ))}
          </div>
        </div>
      )}

      {/* Latency */}
      {latency && (
        <div className="glass-card">
          <div className="card-title">Latency & Throughput (Measured)</div>
          <div className="metrics-grid">
            <div className="glass-card metric-card blue">
              <div className="card-title">ML Inference</div>
              <div className="metric-value">{latency.ml_inference_ms.toFixed(1)}ms</div>
              <div className="metric-label">{latency.batch_size} transactions</div>
            </div>
            <div className="glass-card metric-card green">
              <div className="card-title">Throughput</div>
              <div className="metric-value">{formatNumber(Math.round(latency.events_per_sec))}</div>
              <div className="metric-label">events/sec</div>
            </div>
            <div className="glass-card metric-card purple">
              <div className="card-title">Agent Investigation</div>
              {latency.agent_measured ? (
                <>
                  <div className="metric-value">{latency.agent_investigation_ms.toFixed(0)}ms</div>
                  <div className="metric-label">single-run measurement</div>
                </>
              ) : (
                <>
                  <div className="metric-value" style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>—</div>
                  <div className="metric-label">Not measured in this run — requires live API key</div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   EVALUATION MATRIX
   ═══════════════════════════════════════════════════════════════════ */
function EvaluationView() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getEvaluationMatrix()
      .then(setData)
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!data) return <div className="loading">Loading evaluation</div>;

  const rows = [
    { category: 'ML Performance', metric: 'Precision', value: `${((data.ml?.precision || 0) * 100).toFixed(2)}%` },
    { category: '', metric: 'Recall', value: `${((data.ml?.recall || 0) * 100).toFixed(2)}%` },
    { category: '', metric: 'F1 Score', value: `${((data.ml?.f1 || 0) * 100).toFixed(2)}%` },
    { category: '', metric: 'PR-AUC', value: (data.ml?.pr_auc || 0).toFixed(4) },
    { category: 'Financial', metric: 'FP Cost', value: formatCurrency(data.financial?.fp_cost || 0) },
    { category: '', metric: 'FN Cost', value: formatCurrency(data.financial?.fn_cost || 0) },
    { category: '', metric: 'Expected Loss', value: formatCurrency(data.financial?.expected_loss || 0) },
    { category: '', metric: 'Potential Exposure', value: formatCurrency(data.financial?.potential_exposure || 0) },
    { category: 'Operational', metric: 'Detection Latency', value: `${data.operational?.detection_latency_ms || 0}ms` },
    { category: '', metric: 'Throughput', value: `${formatNumber(data.operational?.events_per_sec || 0)} events/sec` },
    { category: '', metric: 'Investigation Time', value: (data.operational?.investigation_time_ms || 0) > 0 ? `${data.operational.investigation_time_ms}ms` : 'Not measured (requires API key)' },
    { category: 'Explainability', metric: 'Counterfactual Coverage', value: `${((data.explainability?.counterfactual_coverage || 0) * 100).toFixed(0)}%` },
    { category: '', metric: 'Avg Perturbations', value: (data.explainability?.avg_perturbations || 0).toFixed(1) },
  ];

  return (
    <div className="glass-card">
      <div className="card-title">Evaluation Matrix — Full System Assessment</div>
      <table className="eval-table">
        <thead>
          <tr><th>Category</th><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="category">{r.category}</td>
              <td>{r.metric}</td>
              <td style={{ fontWeight: 600 }}>{r.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   ERROR STATE
   ═══════════════════════════════════════════════════════════════════ */
function ErrorState({ message }) {
  return (
    <div className="error-state glass-card">
      <div className="error-icon">⚠️</div>
      <p>{message}</p>
      <p style={{ marginTop: 8 }}>Run the pipeline first: <code>python scripts/run_pipeline.py</code></p>
    </div>
  );
}
