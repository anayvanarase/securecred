import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { ShieldCheck, AlertTriangle, ShieldAlert, Activity, CreditCard, Crosshair, HelpCircle, Map as MapIcon, GitCommit, FileCode, CheckCircle, XCircle } from 'lucide-react';
import { ComposableMap, Geographies, Geography, Marker } from "react-simple-maps";
import { ReactFlow, Background, Controls } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './index.css';

const API_BASE = 'http://localhost:8000';
const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

function App() {
  const [activeTab, setActiveTab] = useState('monitor');
  const [stats, setStats] = useState({ total_transactions: 0, fraud_count: 0, fraud_rate: 0 });
  const [transactions, setTransactions] = useState([]);
  const [rules, setRules] = useState([]);
  const [threshold, setThreshold] = useState(0.80);

  // Predict Form State
  const [amount, setAmount] = useState('150');
  const [merchant, setMerchant] = useState('retail');
  const [isInternational, setIsInternational] = useState('0');
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Graph State
  const [selectedTxn, setSelectedTxn] = useState(null);

  // Rules Form State
  const [ruleField, setRuleField] = useState('transaction_amount_inr');
  const [ruleOp, setRuleOp] = useState('>');
  const [ruleVal, setRuleVal] = useState('10000');

  // Fetching Data
  const fetchData = async () => {
    try {
      const statsRes = await axios.get(`${API_BASE}/stats`);
      setStats(statsRes.data);
      const txRes = await axios.get(`${API_BASE}/transactions`);
      setTransactions(txRes.data);
      const rulesRes = await axios.get(`${API_BASE}/rules`);
      setRules(rulesRes.data);
    } catch (e) {
      console.warn("Could not fetch data from API. Is it running?", e);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    const now = new Date();
    const payload = {
      transaction_amount_inr: parseFloat(amount),
      is_international: parseInt(isInternational),
      merchant_category: merchant,
      country_code: isInternational === '1' ? 'US' : 'IN',
      pos_entry_mode: "chip",
      txn_hour: now.getHours(),
      txn_day: now.getDate(),
      txn_month: now.getMonth() + 1,
      txn_weekday: now.getDay(),
      is_weekend: (now.getDay() === 0 || now.getDay() === 6) ? 1 : 0,
      is_night_txn: (now.getHours() <= 6 || now.getHours() >= 22) ? 1 : 0
    };
    try {
      const res = await axios.post(`${API_BASE}/predict?threshold=${threshold}`, payload);
      setPredictionResult(res.data);
      fetchData();
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const simulateTransaction = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/simulate?threshold=${threshold}`);
      fetchData();
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (id, status) => {
    try {
      await axios.post(`${API_BASE}/action`, { id, status });
      fetchData();
      if (selectedTxn && selectedTxn.id === id) {
        setSelectedTxn(prev => ({ ...prev, status }));
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddRule = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API_BASE}/rules`, {
        field: ruleField, operator: ruleOp, value: parseFloat(ruleVal), action: 'BLOCK'
      });
      fetchData();
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearRules = async () => {
    try {
      await axios.post(`${API_BASE}/rules/clear`);
      fetchData();
    } catch (err) {
      console.error(err);
    }
  };

  // Process data for Map
  const mapMarkers = transactions.slice(0, 50).map(tx => {
    const isUS = tx.country_code === 'US';
    return {
      coordinates: isUS ? [-95.7, 37.0] : [78.9, 20.5],
      isFraud: tx.prediction === 'FRAUD' || tx.risk_level === 'CRITICAL (RULE)',
      id: tx.id
    };
  });

  // Process graph nodes
  const graphNodes = useMemo(() => {
    if (!selectedTxn) return [];
    return [
      { id: 'user', type: 'input', position: { x: 250, y: 50 }, data: { label: 'Cardholder' } },
      { id: 'txn', position: { x: 250, y: 150 }, data: { label: `Txn #${selectedTxn.id} (₹${selectedTxn.amount})` } },
      { id: 'merchant', position: { x: 100, y: 250 }, data: { label: `Merchant: ${selectedTxn.merchant}` } },
      { id: 'loc', position: { x: 400, y: 250 }, data: { label: `Loc: ${selectedTxn.country_code}` } },
      { id: 'risk', type: 'output', position: { x: 250, y: 350 }, data: { label: `Status: ${selectedTxn.status.toUpperCase()}` } }
    ];
  }, [selectedTxn]);

  const graphEdges = useMemo(() => {
    if (!selectedTxn) return [];
    return [
      { id: 'e1', source: 'user', target: 'txn', animated: true },
      { id: 'e2', source: 'txn', target: 'merchant', animated: selectedTxn.prediction === 'FRAUD', style: { stroke: selectedTxn.prediction === 'FRAUD' ? 'red' : 'gray' } },
      { id: 'e3', source: 'txn', target: 'loc', animated: selectedTxn.prediction === 'FRAUD' },
      { id: 'e4', source: 'merchant', target: 'risk' },
      { id: 'e5', source: 'loc', target: 'risk' }
    ];
  }, [selectedTxn]);

  return (
    <div className="min-h-screen p-6 bg-slate-900 text-slate-200 font-sans">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <ShieldCheck className="text-blue-500 w-8 h-8" />
            Securecred <span className="font-light text-blue-400">Fraud Guard</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">Advanced AI & Rules-Driven Security Platform</p>
        </div>
        <button
          onClick={simulateTransaction}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-md font-medium transition-colors border border-blue-500 flex items-center gap-2"
        >
          <Activity className="w-4 h-4" /> Simulate Random Txn
        </button>
      </header>

      {/* TABS */}
      <div className="flex gap-4 mb-6 border-b border-slate-700">
        <button onClick={() => setActiveTab('monitor')} className={`pb-3 px-2 font-medium flex gap-2 items-center ${activeTab === 'monitor' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-slate-500 hover:text-slate-300'}`}>
          <MapIcon className="w-4 h-4" /> Live Monitor
        </button>
        <button onClick={() => setActiveTab('queue')} className={`pb-3 px-2 font-medium flex gap-2 items-center ${activeTab === 'queue' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-slate-500 hover:text-slate-300'}`}>
          <GitCommit className="w-4 h-4" /> Action Queue & Graphs
        </button>
        <button onClick={() => setActiveTab('rules')} className={`pb-3 px-2 font-medium flex gap-2 items-center ${activeTab === 'rules' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-slate-500 hover:text-slate-300'}`}>
          <FileCode className="w-4 h-4" /> Dynamic Rules
        </button>
      </div>

      {activeTab === 'monitor' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-6">
            <div className="glass-panel p-5 rounded-xl border-t-2 border-t-blue-500 bg-slate-800/50 block">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Crosshair className="text-blue-400 w-5 h-5" /> Sensitivity Engine</h2>
              <div className="flex justify-between items-end mb-2">
                <span className="text-sm text-slate-400">ML Alert Threshold</span>
                <span className="text-xl font-bold">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input type="range" min="0.50" max="0.99" step="0.01" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-slate-700 rounded-lg accent-blue-500 cursor-pointer mb-2" />
            </div>

            <div className="glass-panel p-5 rounded-xl bg-slate-800/50 block">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><CreditCard className="w-5 h-5 text-slate-400" /> Manual Review</h2>
              <form onSubmit={handlePredict} className="space-y-4">
                <div><label className="block text-xs font-medium text-slate-400 mb-1">Amount (INR)</label><input type="number" value={amount} onChange={e => setAmount(e.target.value)} required className="w-full bg-slate-900 border border-slate-700 rounded-md py-2 px-3 text-sm focus:outline-none focus:border-blue-500" /></div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Merchant</label>
                  <select value={merchant} onChange={e => setMerchant(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded-md py-2 px-3 text-sm focus:outline-none focus:border-blue-500">
                    <option value="retail">Retail/Shopping</option><option value="online">Online</option><option value="travel">Travel</option><option value="high_risk">High Risk Goods</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Location</label>
                  <select value={isInternational} onChange={e => setIsInternational(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded-md py-2 px-3 text-sm focus:outline-none focus:border-blue-500">
                    <option value="0">Domestic (India)</option><option value="1">International (US)</option>
                  </select>
                </div>
                <button type="submit" disabled={loading} className="w-full bg-slate-700 hover:bg-slate-600 py-2.5 rounded-md font-medium text-sm transition-colors mt-2">{loading ? 'Checking...' : 'Run ML Check'}</button>
              </form>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="glass-panel p-5 rounded-xl bg-slate-800/50 h-[400px]">
              <h2 className="text-sm font-semibold mb-4 text-slate-300">Global Threat Heatmap (Ping Map)</h2>
              <ComposableMap projectionConfig={{ scale: 140 }}>
                <Geographies geography={geoUrl}>
                  {({ geographies }) => geographies.map(geo => <Geography key={geo.rsmKey} geography={geo} fill="#334155" stroke="#1e293b" />)}
                </Geographies>
                {mapMarkers.map(m => (
                  <Marker key={m.id} coordinates={m.coordinates}>
                    <circle r={m.isFraud ? 8 : 3} fill={m.isFraud ? "#ef4444" : "#10b981"} opacity={m.isFraud ? 0.8 : 0.4} />
                  </Marker>
                ))}
              </ComposableMap>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'queue' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[700px]">
          <div className="glass-panel p-5 rounded-xl bg-slate-800/50 overflow-y-auto">
            <h2 className="text-sm font-semibold mb-4 text-slate-300">Pending Investigation Queue</h2>
            <div className="space-y-3">
              {transactions.filter(t => t.prediction === 'FRAUD' || t.risk_level === 'CRITICAL (RULE)').map((tx) => (
                <div key={tx.id} onClick={() => setSelectedTxn(tx)} className={`p-4 rounded-lg border cursor-pointer transition-colors ${selectedTxn?.id === tx.id ? 'border-blue-500 bg-slate-800' : 'border-slate-700 bg-slate-800/30'}`}>
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-mono text-sm text-slate-300">#{tx.id}</span>
                    <span className={`text-xs px-2 py-1 rounded bg-opacity-20 font-semibold ${tx.status === 'pending' ? 'bg-amber-500 text-amber-400' : tx.status === 'frozen' ? 'bg-blue-500 text-blue-400' : 'bg-emerald-500 text-emerald-400'}`}>
                      {tx.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-lg font-bold text-red-400">₹{tx.amount.toLocaleString()} <span className="text-xs font-normal text-slate-500">at {tx.merchant}</span></div>
                  <p className="text-xs italic text-slate-400 mt-2">"{tx.investigation_summary}"</p>

                  {tx.status === 'pending' && (
                    <div className="mt-4 flex gap-2">
                      <button onClick={(e) => { e.stopPropagation(); handleAction(tx.id, 'frozen') }} className="flex-1 bg-blue-600 hover:bg-blue-500 text-white py-1.5 rounded text-xs font-medium flex justify-center items-center gap-1"><ShieldAlert className="w-3 h-3" /> Freeze Card</button>
                      <button onClick={(e) => { e.stopPropagation(); handleAction(tx.id, 'approved') }} className="flex-1 bg-slate-700 hover:bg-slate-600 text-white py-1.5 rounded text-xs font-medium flex justify-center items-center gap-1"><CheckCircle className="w-3 h-3" /> False Positive</button>
                    </div>
                  )}
                </div>
              ))}
              {transactions.filter(t => t.prediction === 'FRAUD').length === 0 && <p className="text-sm text-slate-500 text-center mt-10">Queue is clean. No high-risk transactions detected.</p>}
            </div>
          </div>
          <div className="glass-panel p-5 rounded-xl bg-slate-800/50 flex flex-col">
            <h2 className="text-sm font-semibold mb-4 text-slate-300">Fraud Ring Network Graph</h2>
            <div className="flex-1 border border-slate-700 rounded-lg overflow-hidden bg-slate-900 relative">
              {selectedTxn ? (
                <ReactFlow nodes={graphNodes} edges={graphEdges} fitView>
                  <Background color="#334155" gap={16} />
                  <Controls />
                </ReactFlow>
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">Select a transaction from the queue to view its network graph.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'rules' && (
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="glass-panel p-6 rounded-xl bg-slate-800/50">
            <h2 className="text-lg font-semibold mb-4">Hard Velocity Rules Engine</h2>
            <p className="text-sm text-slate-400 mb-6">Rules take precedence over machine learning models. If a transaction triggers a rule, it is automatically marked as CRITICAL.</p>
            <form onSubmit={handleAddRule} className="flex gap-4 items-end bg-slate-900 p-4 border border-slate-700 rounded-lg">
              <div className="flex-1">
                <label className="block text-xs font-medium text-slate-400 mb-1">Field</label>
                <select value={ruleField} onChange={e => setRuleField(e.target.value)} className="w-full bg-slate-800 border-none rounded py-2 text-sm text-slate-200"><option value="transaction_amount_inr">Amount (INR)</option><option value="is_international">Is International (1=Yes)</option></select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Operator</label>
                <select value={ruleOp} onChange={e => setRuleOp(e.target.value)} className="w-32 bg-slate-800 border-none rounded py-2 text-sm text-center text-slate-200"><option value="&gt;">&gt; (Greater)</option><option value="&lt;">&lt; (Less)</option><option value="==">== (Equals)</option></select>
              </div>
              <div className="flex-1">
                <label className="block text-xs font-medium text-slate-400 mb-1">Value</label>
                <input type="number" value={ruleVal} onChange={e => setRuleVal(e.target.value)} required className="w-full bg-slate-800 border-none rounded py-2 px-3 text-sm text-slate-200" />
              </div>
              <button type="submit" className="bg-emerald-600 hover:bg-emerald-500 px-4 py-2 rounded text-sm font-medium">Add Rule</button>
            </form>

            <div className="mt-8">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-semibold text-slate-300">Active Rules ({rules.length})</h3>
                {rules.length > 0 && <button onClick={handleClearRules} className="text-xs text-red-400 hover:text-red-300">Clear All</button>}
              </div>
              <div className="space-y-2">
                {rules.map((r, i) => (
                  <div key={i} className="flex items-center justify-between bg-slate-900 border border-slate-700 px-4 py-3 rounded-lg">
                    <code className="text-sm text-blue-300">IF {r.field} {r.operator} {r.value}</code>
                    <span className="text-xs font-bold bg-red-500 text-white px-2 py-0.5 rounded uppercase">THEN {r.action}</span>
                  </div>
                ))}
                {rules.length === 0 && <div className="text-sm text-slate-500 italic p-4 text-center border border-dashed border-slate-700 rounded-lg">No active rules. System is running exclusively on AI/ML.</div>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
