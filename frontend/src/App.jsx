import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  ShieldCheck, AlertTriangle, ShieldAlert, Activity, CreditCard, Crosshair, 
  Map as MapIcon, GitCommit, FileCode, CheckCircle, Info, ChevronDown, 
  ChevronUp, Upload, FileText, Download, PieChart, ExternalLink, RefreshCw,
  User, Zap, TrendingUp, Clock
} from 'lucide-react';
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
  const [merchant, setMerchant] = useState('online_retail');
  const [isInternational, setIsInternational] = useState('0');
  const [distance, setDistance] = useState('5');
  const [cardAge, setCardAge] = useState('365');
  const [creditLimit, setCreditLimit] = useState('50000');
  const [avgAmount, setAvgAmount] = useState('500');
  const [stdAmount, setStdAmount] = useState('100');
  const [v1h, setV1h] = useState('1');
  const [v24h, setV24h] = useState('5');
  const [entryMode, setEntryMode] = useState('chip');
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Map State
  const [isMapMinimized, setIsMapMinimized] = useState(false);

  // Batch State
  const [batchFile, setBatchFile] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Graph State
  const [selectedTxn, setSelectedTxn] = useState(null);

  // Rules Form State
  const [ruleField, setRuleField] = useState('transaction_amount_inr');
  const [ruleOp, setRuleOp] = useState('>');
  const [ruleVal, setRuleVal] = useState('10000');
  const [activeQueueTab, setActiveQueueTab] = useState('high');
  const [showExportOptions, setShowExportOptions] = useState(false);

  const merchantCategories = [
    'online_retail', 'entertainment', 'travel', 'jewellery', 
    'electronics', 'grocery', 'restaurant', 'utilities', 
    'healthcare', 'fuel'
  ];

  // Profile Presets
  const applyPreset = (type) => {
    switch(type) {
      case 'student':
        setCreditLimit('15000');
        setAvgAmount('300');
        setStdAmount('80');
        setCardAge('180');
        setEntryMode('chip');
        setAmount('500');
        break;
      case 'vip':
        setCreditLimit('1000000');
        setAvgAmount('8500');
        setStdAmount('2000');
        setCardAge('1460');
        setEntryMode('chip');
        setAmount('15000');
        break;
      case 'new':
        setCreditLimit('30000');
        setAvgAmount('0');
        setStdAmount('0');
        setCardAge('5');
        setEntryMode('manual');
        setAmount('500');
        break;
      default:
        setCreditLimit('50000');
        setAvgAmount('500');
        setStdAmount('100');
        setCardAge('365');
        setEntryMode('chip');
        setAmount('150');
    }
  };

  // Fetching Data
  const fetchData = async () => {
    try {
      const statsRes = await axios.get(`${API_BASE}/stats`);
      setStats(statsRes.data || { total_transactions: 0, fraud_count: 0, fraud_rate: 0 });
      const txRes = await axios.get(`${API_BASE}/transactions`);
      setTransactions(txRes.data || []);
      const rulesRes = await axios.get(`${API_BASE}/rules`);
      setRules(rulesRes.data || []);
    } catch (e) {
      console.warn("API Connection Issue:", e.message);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    const now = new Date();
    const payload = {
      transaction_amount_inr: parseFloat(amount),
      is_international: parseInt(isInternational),
      merchant_category: merchant,
      country_code: isInternational === '1' ? 'US' : 'IN',
      pos_entry_mode: entryMode,
      txn_hour: now.getHours(),
      txn_day: now.getDate(),
      txn_month: now.getMonth() + 1,
      txn_weekday: now.getDay(),
      is_weekend: (now.getDay() === 0 || now.getDay() === 6) ? 1 : 0,
      is_night_txn: (now.getHours() <= 6 || now.getHours() >= 22) ? 1 : 0,
      distance_from_home_km: parseFloat(distance),
      card_age_days: parseInt(cardAge),
      credit_limit_inr: parseFloat(creditLimit),
      avg_txn_amount_30d: parseFloat(avgAmount),
      std_txn_amount_30d: parseFloat(stdAmount),
      velocity_last_1h: parseInt(v1h),
      velocity_last_24h: parseInt(v24h)
    };
    try {
      const res = await axios.post(`${API_BASE}/predict?threshold=${threshold}`, payload);
      setPredictionResult(res.data);
      fetchData();
    } catch (err) {
      console.error("Prediction Error:", err);
      alert("Backend unreachable or rule violation error.");
    } finally {
      setLoading(false);
    }
  };

  const handleBatchUpload = async (e) => {
    e.preventDefault();
    if (!batchFile) return;
    setBatchLoading(true);
    const formData = new FormData();
    formData.append('file', batchFile);
    try {
      const res = await axios.post(`${API_BASE}/predict/bulk?threshold=${threshold}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setBatchResults(res.data);
      fetchData();
    } catch (err) {
      console.error("Batch Error:", err);
      alert("Error processing batch file.");
    } finally {
      setBatchLoading(false);
    }
  };

  const simulateTransaction = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/simulate?threshold=${threshold}`);
      setPredictionResult(res.data);
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
        setSelectedTxn(prev => ({...prev, status}));
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
    } catch(err) {
      console.error(err);
    }
  };

  const handleClearRules = async () => {
    try {
      await axios.post(`${API_BASE}/rules/clear`);
      fetchData();
    } catch(err) {
      console.error(err);
    }
  };

  const handleExportReport = (format) => {
    if (!batchResults || !batchResults.results) return;
    
    let content = '';
    let fileName = `risk_report_${new Date().toISOString().split('T')[0]}`;
    let mimeType = '';

    if (format === 'csv') {
      const headers = ['ID', 'AMOUNT', 'MERCHANT', 'RISK_LEVEL', 'PREDICTION'];
      const rows = batchResults.results.map(r => [
        r.id,
        r.amount,
        r.merchant,
        r.risk_level,
        r.prediction
      ].join(','));
      content = [headers.join(','), ...rows].join('\n');
      fileName += '.csv';
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify(batchResults.results, null, 2);
      fileName += '.json';
      mimeType = 'application/json';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
    setShowExportOptions(false);
  };

  // Process data for Map
  const mapMarkers = useMemo(() => {
    return transactions.slice(0, 50).map(tx => {
      const isUS = tx.country_code === 'US';
      return {
        coordinates: isUS ? [-95.7, 37.0] : [78.9, 20.5],
        isFraud: tx.prediction === 'FRAUD' || tx.risk_level?.includes('RULE'),
        id: tx.id
      };
    });
  }, [transactions]);

  // Process graph nodes
  const graphNodes = useMemo(() => {
    if (!selectedTxn) return [];
    return [
      { id: 'user', type: 'input', position: { x: 250, y: 50 }, data: { label: 'Cardholder' }, style: { background: '#f1f5f9', color: '#1e293b', border: '1px solid #e2e8f0' } },
      { id: 'txn', position: { x: 250, y: 150 }, data: { label: `Txn #${selectedTxn.id} (₹${selectedTxn.amount})` }, style: { background: '#f1f5f9', color: '#1e293b', border: '1px solid #e2e8f0' } },
      { id: 'merchant', position: { x: 100, y: 250 }, data: { label: `Merchant: ${selectedTxn.merchant.replace('_', ' ').toUpperCase()}` }, style: { background: '#f1f5f9', color: '#1e293b', border: '1px solid #e2e8f0' } },
      { id: 'loc', position: { x: 400, y: 250 }, data: { label: `Loc: ${selectedTxn.country_code}` }, style: { background: '#f1f5f9', color: '#1e293b', border: '1px solid #e2e8f0' } },
      { id: 'risk', type: 'output', position: { x: 250, y: 350 }, data: { label: `Status: ${selectedTxn.status.toUpperCase()}` }, style: { background: selectedTxn.status === 'frozen' ? '#3b82f6' : '#10b981', color: '#fff' } }
    ];
  }, [selectedTxn]);

  const graphEdges = useMemo(() => {
    if (!selectedTxn) return [];
    const isFraud = selectedTxn.prediction === 'FRAUD' || selectedTxn.risk_level?.includes('RULE');
    return [
      { id: 'e1', source: 'user', target: 'txn', animated: true },
      { id: 'e2', source: 'txn', target: 'merchant', animated: isFraud, style: { stroke: isFraud ? '#ef4444' : '#94a3b8' } },
      { id: 'e3', source: 'txn', target: 'loc', animated: isFraud, style: { stroke: isFraud ? '#ef4444' : '#94a3b8' } },
      { id: 'e4', source: 'merchant', target: 'risk' },
      { id: 'e5', source: 'loc', target: 'risk' }
    ];
  }, [selectedTxn]);

  return (
    <div className="min-h-screen p-6 bg-slate-50 text-slate-800 font-sans selection:bg-blue-100 selection:text-blue-900">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold flex items-center gap-3 tracking-tight text-slate-900">
            <ShieldCheck className="text-blue-600 w-10 h-10" />
            Securecred <span className="font-medium text-slate-400">Fraud Center</span>
          </h1>
          <p className="text-slate-500 text-sm mt-1 font-medium">Enterprise Real-time Intelligence & Bulk Analytics</p>
        </div>
        <div className="flex gap-4 items-center">
          <div className="flex bg-white p-1.5 rounded-xl border border-slate-200 shadow-sm">
             <div className="px-4 py-1 flex flex-col items-center border-r border-slate-100">
                <span className="text-[10px] text-slate-400 uppercase font-black">Volume</span>
                <span className="text-base font-bold text-slate-800">{stats.total_transactions}</span>
             </div>
             <div className="px-4 py-1 flex flex-col items-center border-r border-slate-100">
                <span className="text-[10px] text-slate-400 uppercase font-black">Blocked</span>
                <span className="text-base font-bold text-rose-600">{stats.fraud_count}</span>
             </div>
             <div className="px-4 py-1 flex flex-col items-center">
                <span className="text-[10px] text-slate-400 uppercase font-black">Rate</span>
                <span className="text-base font-bold text-blue-600">{(stats.fraud_rate * 100).toFixed(1)}%</span>
             </div>
          </div>
          <button 
            onClick={simulateTransaction}
            className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-xl font-bold transition-all shadow-md active:scale-95 flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /> Simulate
          </button>
        </div>
      </header>

      {/* TABS */}
      <nav className="flex gap-10 mb-8 border-b border-slate-200">
        <button onClick={() => setActiveTab('monitor')} className={`pb-4 px-1 text-sm flex gap-2 items-center transition-all ${activeTab === 'monitor' ? 'tab-active' : 'tab-inactive'}`}>
          <Activity className="w-4 h-4" /> Live Monitor
        </button>
        <button onClick={() => setActiveTab('batch')} className={`pb-4 px-1 text-sm flex gap-2 items-center transition-all ${activeTab === 'batch' ? 'tab-active' : 'tab-inactive'}`}>
          <Upload className="w-4 h-4" /> Batch Processing
        </button>
        <button onClick={() => setActiveTab('queue')} className={`pb-4 px-1 text-sm flex gap-2 items-center transition-all ${activeTab === 'queue' ? 'tab-active' : 'tab-inactive'}`}>
          <GitCommit className="w-4 h-4" /> Investigation Hub
        </button>
        <button onClick={() => setActiveTab('rules')} className={`pb-4 px-1 text-sm flex gap-2 items-center transition-all ${activeTab === 'rules' ? 'tab-active' : 'tab-inactive'}`}>
          <FileCode className="w-4 h-4" /> Rules Engine
        </button>
      </nav>

      <main className="animate-in fade-in duration-500">
        {activeTab === 'monitor' && (
          <div className="space-y-6">
            <div className={`grid grid-cols-1 ${isMapMinimized ? 'lg:grid-cols-1' : 'lg:grid-cols-3'} gap-6 transition-all duration-300`}>
              <div className="space-y-6">
                <section className="glass-panel p-6 bg-white border-t-4 border-t-blue-600">
                  <h2 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center justify-between">
                    Risk Parameters
                    <span className="text-blue-600 bg-blue-50 px-2 py-0.5 rounded text-[10px]">{(threshold * 100).toFixed(0)}% Threshold</span>
                  </h2>

                  {/* PROFILE PRESETS */}
                  <div className="flex gap-2 mb-6 overflow-x-auto pb-2 scrollbar-none">
                     <button onClick={() => applyPreset('default')} className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-[10px] font-black uppercase flex items-center gap-1.5 transition-all outline-none">
                        <User className="w-3 h-3"/> Standard
                     </button>
                     <button onClick={() => applyPreset('student')} className="px-3 py-1.5 bg-sky-50 hover:bg-sky-100 text-sky-600 rounded-lg text-[10px] font-black uppercase flex items-center gap-1.5 transition-all outline-none">
                        <Zap className="w-3 h-3"/> Student
                     </button>
                     <button onClick={() => applyPreset('vip')} className="px-3 py-1.5 bg-amber-50 hover:bg-amber-100 text-amber-600 rounded-lg text-[10px] font-black uppercase flex items-center gap-1.5 transition-all outline-none">
                        <TrendingUp className="w-3 h-3"/> VIP Executive
                     </button>
                     <button onClick={() => applyPreset('new')} className="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-600 rounded-lg text-[10px] font-black uppercase flex items-center gap-1.5 transition-all outline-none border border-rose-100">
                        <Clock className="w-3 h-3"/> New User
                     </button>
                  </div>

                  <input type="range" min="0.50" max="0.99" step="0.01" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-slate-100 rounded-lg accent-blue-600 cursor-pointer mb-6"/>
                  
                  <form onSubmit={handlePredict} className="space-y-5">
                    {/* SECTION 1: ESSENTIALS */}
                    <div className="space-y-4">
                       <h3 className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em]">Transaction Essentials</h3>
                       <div className="grid grid-cols-2 gap-4">
                          <div className="col-span-2">
                            <label className="text-[10px] font-black text-slate-400 uppercase mb-1.5 block">Amount (INR)</label>
                            <input type="number" value={amount} onChange={e => setAmount(e.target.value)} required className="w-full bg-slate-50 border border-slate-200 rounded-xl py-3 px-4 text-sm focus:ring-2 focus:ring-blue-500/20 outline-none transition-all"/>
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-400 uppercase mb-1.5 block">Category</label>
                            <select value={merchant} onChange={e => setMerchant(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-xl py-2.5 px-3 text-xs outline-none focus:ring-2 focus:ring-blue-500/20">
                              {merchantCategories.map(cat => <option key={cat} value={cat}>{cat.replace('_', ' ').toUpperCase()}</option>)}
                            </select>
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-400 uppercase mb-1.5 block">Mode</label>
                            <select value={entryMode} onChange={e => setEntryMode(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-xl py-2.5 px-3 text-xs outline-none focus:ring-2 focus:ring-blue-500/20">
                              <option value="chip">CHIP (SECURE)</option>
                              <option value="swipe">SWIPE (LEGACY)</option>
                              <option value="manual">MANUAL (ONLINE)</option>
                            </select>
                          </div>
                       </div>
                    </div>

                    {/* SECTION 2: USER PROFILE */}
                    <div className="pt-4 border-t border-slate-50 space-y-4">
                       <h3 className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em]">Dynamic User Profile</h3>
                       <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="text-[10px] font-bold text-slate-500 block mb-1">Credit Limit</label>
                            <input type="number" value={creditLimit} onChange={e => setCreditLimit(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                          </div>
                          <div>
                            <label className="text-[10px] font-bold text-slate-500 block mb-1">30d Avg Txn</label>
                            <input type="number" value={avgAmount} onChange={e => setAvgAmount(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                          </div>
                          <div>
                            <label className="text-[10px] font-bold text-slate-500 block mb-1">30d Std Dev</label>
                            <input type="number" value={stdAmount} onChange={e => setStdAmount(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                          </div>
                          <div>
                            <label className="text-[10px] font-bold text-slate-500 block mb-1">Card Age (Days)</label>
                            <input type="number" value={cardAge} onChange={e => setCardAge(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                          </div>
                       </div>
                    </div>

                    {/* SECTION 3: CONTEXTUAL DATA */}
                    <div className="pt-4 border-t border-slate-50 space-y-4">
                      <h3 className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em]">Security Environment</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-[10px] font-black text-slate-400 uppercase mb-1.5 block">Region</label>
                          <select value={isInternational} onChange={e => setIsInternational(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-xl py-2.5 px-3 text-xs outline-none focus:ring-2 focus:ring-blue-500/20">
                            <option value="0">Domestic (IN)</option><option value="1">Intl (US)</option>
                          </select>
                        </div>
                        <div>
                          <label className="text-[10px] font-bold text-slate-500 block mb-1">Distance (km)</label>
                          <input type="number" value={distance} onChange={e => setDistance(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                        </div>
                        <div>
                          <label className="text-[10px] font-bold text-slate-500 block mb-1">Txns (1h)</label>
                          <input type="number" value={v1h} onChange={e => setV1h(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                        </div>
                        <div>
                          <label className="text-[10px] font-bold text-slate-500 block mb-1">Txns (24h)</label>
                          <input type="number" value={v24h} onChange={e => setV24h(e.target.value)} className="w-full bg-slate-50 border border-slate-200 rounded-lg py-1.5 px-3 text-xs"/>
                        </div>
                      </div>
                    </div>

                    <button type="submit" disabled={loading} className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3.5 rounded-xl font-bold text-sm transition-all shadow-md active:scale-95 disabled:opacity-50">
                      {loading ? 'CALCULATING RISK...' : 'RUN SECURITY SCAN'}
                    </button>
                  </form>
                </section>

                {predictionResult && (
                  <div className={`p-6 rounded-2xl border-2 animate-in slide-in-from-bottom-2 duration-300 ${predictionResult.prediction === 'FRAUD' ? 'bg-rose-50 border-rose-100' : 'bg-emerald-50 border-emerald-100'}`}>
                    <div className="flex justify-between items-center mb-4">
                      <div className={`text-xl font-black ${predictionResult.prediction === 'FRAUD' ? 'text-rose-600' : 'text-emerald-600'}`}>
                        {predictionResult.prediction}
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] font-black text-slate-400 uppercase block">AI Score</span>
                        <div className="text-2xl font-black text-slate-900">{(predictionResult.fraud_probability * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                    <div className="space-y-2.5 pt-4 border-t border-white/50">
                       <p className="text-[10px] text-slate-500 flex items-center gap-1 font-black uppercase"><Info className="w-3 h-3"/> ML Attribution</p>
                       <ul className="text-xs text-slate-600 space-y-2 list-none">
                          {predictionResult.explanation.map((e, i) => <li key={i} className="flex gap-2 items-start"><CheckCircle className="w-3 h-3 text-emerald-500 mt-0.5 shrink-0"/> {e}</li>)}
                       </ul>
                    </div>
                  </div>
                )}
              </div>
              
              {!isMapMinimized && (
                <div className="lg:col-span-2 glass-panel p-6 bg-white flex flex-col min-h-[520px] shadow-sm relative overflow-hidden transition-all duration-300">
                  <h2 className="text-sm font-bold mb-6 text-slate-400 flex items-center justify-between">
                    <span className="flex items-center gap-2"><MapIcon className="w-4 h-4 text-slate-400"/> Geographic Origin Tracker</span>
                    <button onClick={() => setIsMapMinimized(true)} className="p-1 hover:bg-slate-50 rounded-lg transition-colors border border-slate-100"><ChevronUp className="w-4 h-4"/></button>
                  </h2>
                  <div className="flex-1 bg-slate-50 rounded-2xl border border-slate-100 overflow-hidden">
                    <ComposableMap projectionConfig={{ scale: 160 }} className="w-full h-full">
                      <Geographies geography={geoUrl}>
                        {({ geographies }) => geographies.map(geo => <Geography key={geo.rsmKey} geography={geo} fill="#e2e8f0" stroke="#f8fafc" strokeWidth={0.5} style={{ default: { outline: 'none' }, hover: { fill: '#cbd5e1', outline: 'none' } }} />)}
                      </Geographies>
                      {mapMarkers.map(m => (
                        <Marker key={m.id} coordinates={m.coordinates}>
                          <circle r={m.isFraud ? 10 : 4} fill={m.isFraud ? "#ef4444" : "#10b981"} className={m.isFraud ? "animate-pulse" : ""} opacity={m.isFraud ? 0.7 : 0.4} />
                        </Marker>
                      ))}
                    </ComposableMap>
                  </div>
                </div>
              )}
              
              {isMapMinimized && (
                <div className="glass-panel p-4 bg-white flex items-center justify-between shadow-sm">
                   <div className="flex items-center gap-3 text-slate-400 font-bold text-sm italic">
                      <MapIcon className="w-4 h-4" /> Origins Monitor is collapsed. High-latency visual off.
                   </div>
                   <button onClick={() => setIsMapMinimized(false)} className="flex items-center gap-2 text-xs font-black text-blue-600 bg-blue-50 px-4 py-2 rounded-xl hover:bg-blue-100 transition-all">
                      <ChevronDown className="w-3 h-3"/> RESTORE MAP VIEW
                   </button>
                </div>
              )}
            </div>

            <section className="glass-panel p-6 bg-white overflow-hidden shadow-sm">
              <h2 className="text-sm font-black mb-6 text-slate-400 flex justify-between items-center">
                 <span className="flex items-center gap-2 underline decoration-blue-500 decoration-4 underline-offset-8">Live Operating Stream</span>
                 <span className="text-[10px] font-black uppercase text-white bg-slate-900 px-3 py-1 rounded-full flex items-center gap-2">
                   <RefreshCw className="w-3 h-3 animate-spin"/> Connected
                 </span>
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="text-[10px] uppercase text-slate-400 font-black tracking-widest border-b border-slate-50">
                    <tr><th className="px-5 py-4">Auth Ref</th><th className="px-5 py-4">Time</th><th className="px-5 py-4">Amount</th><th className="px-5 py-4">Context</th><th className="px-5 py-4 text-right">Confidence</th><th className="px-5 py-4 text-center">Outcome</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {transactions.slice(0, 10).map((tx, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 transition-all group">
                        <td className="px-5 py-4 font-mono text-slate-400 text-xs">#{tx.id.toString().padStart(6, '0')}</td>
                        <td className="px-5 py-4 text-slate-500 font-medium">{tx.time.split(' ')[1]}</td>
                        <td className="px-5 py-4 text-slate-900 font-extrabold">₹{tx.amount.toLocaleString()}</td>
                        <td className="px-5 py-4 text-slate-400 italic text-[10px] uppercase">{tx.merchant.replace('_', ' ')} • {tx.country_code}</td>
                        <td className={`px-5 py-4 text-right font-black ${tx.fraud_probability > 0.7 ? 'text-rose-600' : 'text-slate-400'}`}>{(tx.fraud_probability * 100).toFixed(1)}%</td>
                        <td className="px-5 py-4 text-center">
                          <span className={`px-3 py-1 rounded-lg text-[10px] font-black uppercase border ${tx.prediction === 'FRAUD' || tx.risk_level?.includes('RULE') ? 'bg-rose-50 text-rose-600 border-rose-100' : tx.risk_level === 'MEDIUM' ? 'bg-amber-50 text-amber-600 border-amber-100' : 'bg-emerald-50 text-emerald-600 border-emerald-100'}`}>
                            {tx.prediction === 'FRAUD' || tx.risk_level?.includes('RULE') ? 'BLOCKED' : tx.risk_level === 'MEDIUM' ? 'MONITORED' : 'APPROVED'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'batch' && (
          <div className="max-w-5xl mx-auto space-y-6">
            <div className="glass-panel p-10 bg-white border-dashed border-2 border-slate-200 hover:border-blue-400 transition-all group text-center flex flex-col items-center">
              <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-6 text-blue-600 group-hover:scale-110 transition-transform">
                <Upload className="w-8 h-8"/>
              </div>
              <h2 className="text-2xl font-extrabold text-slate-900 mb-2">High-Capacity Batch Analysis</h2>
              <p className="text-slate-500 max-w-lg mb-8">Process thousands of transactions in seconds. Upload a CSV or JSON file following the banking spec to run deep AI scans across entire volumes.</p>
              
              <div className="flex gap-4">
                 <label className="cursor-pointer bg-slate-900 hover:bg-slate-800 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg active:scale-95">
                    SELECT FILE
                    <input type="file" className="hidden" accept=".csv,.json" onChange={(e) => setBatchFile(e.target.files[0])}/>
                 </label>
                 <button onClick={handleBatchUpload} disabled={!batchFile || batchLoading} className="bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-3 rounded-xl font-bold transition-all shadow-lg active:scale-95 disabled:opacity-50">
                    {batchLoading ? 'PROCESSING VOLUME...' : 'EXECUTE SCAN'}
                 </button>
              </div>
              {batchFile && <div className="mt-4 text-sm font-bold text-blue-600 flex items-center gap-2"><FileText className="w-4 h-4"/> Ready: {batchFile.name}</div>}
            </div>

            {batchResults && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in slide-in-from-top-4 duration-500">
                <div className="glass-panel p-8 bg-white text-center">
                   <div className="flex justify-center mb-4"><PieChart className="w-12 h-12 text-blue-600"/></div>
                   <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-6 border-b border-slate-50 pb-2">Volume Integrity</h3>
                   <div className="grid grid-cols-2 gap-4">
                       <div className="p-4 bg-slate-50 rounded-2xl">
                          <span className="text-[10px] text-slate-400 font-bold block">TOTAL SCAN</span>
                          <span className="text-2xl font-black text-slate-900">{batchResults.total_processed}</span>
                       </div>
                       <div className="p-4 bg-rose-50 rounded-2xl">
                          <span className="text-[10px] text-rose-400 font-bold block">BLOCKED</span>
                          <span className="text-2xl font-black text-rose-600">{batchResults.fraud_detected}</span>
                       </div>
                       <div className="p-4 bg-amber-50 rounded-2xl col-span-2">
                          <span className="text-[10px] text-amber-500 font-bold block">MONITORED (Medium Risk)</span>
                          <span className="text-2xl font-black text-amber-600">
                            {batchResults.results.filter(r => r.risk_level === 'MEDIUM' && r.prediction !== 'FRAUD').length}
                          </span>
                       </div>
                    </div>
                   <div className="relative">
                      {!showExportOptions ? (
                        <button 
                          onClick={() => setShowExportOptions(true)}
                          className="mt-8 w-full bg-slate-900 hover:bg-slate-800 text-white py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-lg transition-all active:scale-95"
                        >
                          <Download className="w-4 h-4"/> EXPORT RISK REPORT
                        </button>
                      ) : (
                        <div className="mt-8 flex gap-2 animate-in fade-in zoom-in duration-200">
                           <button 
                             onClick={() => handleExportReport('csv')}
                             className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-xl font-bold text-xs uppercase tracking-tighter"
                           >
                              Download CSV
                           </button>
                           <button 
                             onClick={() => handleExportReport('json')}
                             className="flex-1 bg-blue-600 hover:bg-blue-500 text-white py-3 rounded-xl font-bold text-xs uppercase tracking-tighter"
                           >
                              Download JSON
                           </button>
                           <button 
                             onClick={() => setShowExportOptions(false)}
                             className="px-4 bg-slate-100 text-slate-400 rounded-xl hover:bg-slate-200"
                           >
                              ×
                           </button>
                        </div>
                      )}
                   </div>
                </div>

                <div className="lg:col-span-2 glass-panel p-6 bg-white overflow-hidden">
                    <div className="flex items-center justify-between mb-6">
                       <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest">All Scan Results</h3>
                       <button
                         onClick={() => setActiveTab('queue')}
                         className="flex items-center gap-2 text-[10px] font-black bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 transition-all"
                       >
                         <GitCommit className="w-3 h-3"/> VIEW IN INVESTIGATION HUB
                       </button>
                    </div>
                    <div className="overflow-y-auto max-h-[400px]">
                       <table className="w-full text-left text-xs">
                         <thead className="sticky top-0 bg-white border-b border-slate-100">
                            <tr className="text-slate-400 uppercase font-black tracking-tighter"><th className="pb-3">INDEX</th><th className="pb-3">AMOUNT</th><th className="pb-3">MERCHANT</th><th className="pb-3">LEVEL</th><th className="pb-3 text-right">STATUS</th></tr>
                         </thead>
                         <tbody className="divide-y divide-slate-50">
                            {batchResults.results.slice(0, 100).map((r, i) => (
                              <tr key={i} className="hover:bg-slate-50 transition-colors">
                                 <td className="py-3 text-slate-400">#{r.id}</td>
                                 <td className="py-3 font-bold text-slate-900">₹{r.amount.toLocaleString()}</td>
                                 <td className="py-3 italic text-slate-500">{r.merchant.replace('_', ' ').toUpperCase()}</td>
                                 <td className="py-3">
                                    <span className={`px-2 py-0.5 rounded text-[8px] font-black uppercase ${
                                      r.risk_level === 'LOW' ? 'bg-emerald-50 text-emerald-600' :
                                      r.risk_level === 'MEDIUM' ? 'bg-amber-50 text-amber-600' :
                                      'bg-rose-50 text-rose-600'
                                    }`}>{r.risk_level}</span>
                                 </td>
                                 <td className="py-3 text-right">
                                    <span className={`font-black ${
                                      r.prediction === 'FRAUD' ? 'text-rose-600' :
                                      r.risk_level === 'MEDIUM' ? 'text-amber-500' :
                                      'text-emerald-600'
                                    }`}>
                                      {r.prediction === 'FRAUD' ? 'BLOCKED' : r.risk_level === 'MEDIUM' ? 'MONITORED' : 'APPROVED'}
                                    </span>
                                 </td>
                              </tr>
                            ))}
                         </tbody>
                       </table>
                    </div>
                 </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'queue' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[740px]">
            <div className="glass-panel p-6 bg-white overflow-y-auto shadow-sm">
              <h2 className="text-sm font-black mb-6 text-slate-400 flex items-center gap-2 uppercase tracking-widest mt-2"><GitCommit className="w-4 h-4 text-emerald-600"/> Investigation Queue</h2>
              
              {/* QUEUE TABS */}
              <div className="flex bg-slate-100 p-1.5 rounded-xl mb-6 gap-1">
                 <button onClick={() => setActiveQueueTab('high')} className={`flex-1 py-2 text-[10px] font-black uppercase rounded-lg transition-all ${activeQueueTab === 'high' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}>
                    High Priority (Blocked)
                 </button>
                 <button onClick={() => setActiveQueueTab('low')} className={`flex-1 py-2 text-[10px] font-black uppercase rounded-lg transition-all ${activeQueueTab === 'low' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}>
                    Low Priority (Medium)
                 </button>
              </div>

              <div className="space-y-4">
                {transactions.filter(t => {
                   if (activeQueueTab === 'high') {
                      return t.prediction === 'FRAUD' || t.risk_level?.includes('RULE');
                   } else {
                      return t.risk_level === 'MEDIUM' && t.prediction !== 'FRAUD';
                   }
                }).map((tx) => (
                  <div key={tx.id} onClick={() => setSelectedTxn(tx)} className={`p-6 rounded-2xl border-2 transition-all cursor-pointer ${selectedTxn?.id === tx.id ? 'border-blue-600 bg-blue-50/50 shadow-lg' : 'border-slate-100 hover:border-slate-200'}`}>
                    <div className="flex justify-between items-start mb-4">
                      <span className="text-[10px] font-black text-slate-400 tracking-widest uppercase">ID_REF #{tx.id}</span>
                      <span className={`text-[10px] px-3 py-1 rounded-full font-black uppercase border ${tx.status === 'pending' ? 'bg-amber-50 text-amber-600 border-amber-100' : tx.status === 'frozen' ? 'bg-blue-50 text-blue-600 border-blue-100' : 'bg-emerald-50 text-emerald-600 border-emerald-100'}`}>
                        {tx.status}
                      </span>
                    </div>
                    <div className="text-3xl font-black text-slate-900 mb-1">₹{tx.amount.toLocaleString()}</div>
                    <div className="text-xs font-bold text-slate-400 uppercase mb-5 flex items-center gap-2">{tx.merchant} • {tx.country_code} • {tx.risk_level}</div>
                    
                    <div className="bg-slate-100 p-5 rounded-xl border-l-4 border-slate-900 mb-6 group-hover:bg-slate-200">
                       <p className="text-[10px] font-black text-slate-500 uppercase mb-2 flex items-center gap-1"><ShieldAlert className="w-3 h-3"/> Agent Intel Report</p>
                       <p className="text-sm text-slate-600 leading-relaxed font-semibold italic">"{tx.investigation_summary}"</p>
                    </div>

                    {tx.status === 'pending' && (
                      <div className="flex gap-4">
                        <button onClick={(e) => { e.stopPropagation(); handleAction(tx.id, 'frozen')}} className="flex-1 bg-slate-900 text-white py-3 rounded-xl text-xs font-black shadow-md hover:bg-black transition-all">FREEZE ASSET</button>
                        <button onClick={(e) => { e.stopPropagation(); handleAction(tx.id, 'approved')}} className="flex-1 bg-slate-100 text-slate-600 py-3 rounded-xl text-xs font-black hover:bg-slate-200 transition-all">EXPRESS APPROVAL</button>
                      </div>
                    )}
                  </div>
                ))}
                {transactions.filter(t => {
                   if (activeQueueTab === 'high') {
                      return t.prediction === 'FRAUD' || t.risk_level?.includes('RULE');
                   } else {
                      return t.risk_level === 'MEDIUM' && t.prediction !== 'FRAUD';
                   }
                }).length === 0 && (
                  <div className="text-center py-40">
                    <CheckCircle className="w-12 h-12 text-emerald-200 mx-auto mb-4" />
                    <p className="text-sm text-slate-400 font-bold uppercase tracking-widest">Pipeline Operational. No Incidents.</p>
                  </div>
                )}
              </div>
            </div>
            <div className="glass-panel p-6 bg-white flex flex-col shadow-sm border-l-8 border-l-blue-600">
              <h2 className="text-sm font-black mb-6 text-slate-400 flex items-center gap-2 uppercase tracking-widest mt-2"><GitCommit className="w-4 h-4 text-blue-600"/> Identity Linkage Analytics</h2>
              <div className="flex-1 border border-slate-100 rounded-3xl overflow-hidden bg-slate-50 relative">
                {selectedTxn ? (
                  <ReactFlow nodes={graphNodes} edges={graphEdges} fitView>
                    <Background color="#cbd5e1" gap={20} />
                    <Controls />
                  </ReactFlow>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-300">
                    <ExternalLink className="w-10 h-10 mb-2 opacity-50"/>
                    <p className="text-xs font-black uppercase">Project identity node from queue to analyze</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'rules' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div className="glass-panel p-10 bg-white shadow-xl">
              <h2 className="text-2xl font-black mb-2 flex items-center gap-3 text-slate-900"><FileCode className="text-blue-600 w-8 h-8"/> Central Policy Control</h2>
              <p className="text-sm text-slate-500 mb-10 font-medium">Define high-priority business logic to instantly override the AI Neural Engine. Policies are applied in real-time to all ingestion endpoints.</p>
              
              <form onSubmit={handleAddRule} className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-slate-50 p-6 rounded-2xl mb-12 border border-slate-100 shadow-inner">
                <div className="md:col-span-1">
                  <label className="block text-[10px] font-black text-slate-400 uppercase mb-2">Policy Target</label>
                  <select value={ruleField} onChange={e=>setRuleField(e.target.value)} className="w-full bg-white border border-slate-200 rounded-xl py-3 px-3 text-xs text-slate-900 focus:ring-2 focus:ring-blue-500/20 outline-none"><option value="transaction_amount_inr">AMOUNT_INR</option><option value="is_international">GEOLOC_INTL</option></select>
                </div>
                <div className="md:col-span-1">
                  <label className="block text-[10px] font-black text-slate-400 uppercase mb-2">Operator</label>
                  <select value={ruleOp} onChange={e=>setRuleOp(e.target.value)} className="w-full bg-white border border-slate-200 rounded-xl py-3 px-3 text-xs text-slate-900 outline-none"><option value="&gt;">GREATER THAN</option><option value="&lt;">LESS THAN</option></select>
                </div>
                <div className="md:col-span-1">
                  <label className="block text-[10px] font-black text-slate-400 uppercase mb-2">Constant</label>
                  <input type="number" value={ruleVal} onChange={e=>setRuleVal(e.target.value)} required className="w-full bg-white border border-slate-200 rounded-xl py-3 px-4 text-xs font-bold text-slate-900 outline-none"/>
                </div>
                <button type="submit" className="bg-slate-900 hover:bg-black text-white font-bold rounded-xl text-xs py-3 md:mt-6 transition-all shadow-md active:scale-95">ACTIVATE POLICY</button>
              </form>
              
              <div className="space-y-6">
                <div className="flex justify-between items-center px-2">
                  <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest">Active Neural Overrides</h3>
                  {rules.length > 0 && <button onClick={handleClearRules} className="text-[10px] font-black bg-rose-50 text-rose-600 px-4 py-1.5 rounded-full border border-rose-100 hover:bg-rose-600 hover:text-white transition-all">TERMINATE ALL</button>}
                </div>
                <div className="grid grid-cols-1 gap-3">
                  {rules.map((r, i) => (
                    <div key={i} className="flex items-center justify-between bg-white border border-slate-200 p-5 rounded-2xl hover:border-blue-600 transition-all border-l-8 border-l-blue-600 group shadow-sm">
                      <div className="flex items-center gap-4">
                         <span className="text-[10px] bg-slate-100 px-3 py-1.5 rounded text-slate-500 font-black">LOGIC_GATE</span>
                         <code className="text-sm font-black text-slate-900">{r.field.toUpperCase()} {r.operator} {r.value}</code>
                      </div>
                      <span className="text-[10px] font-black bg-rose-600 text-white px-4 py-1.5 rounded-lg uppercase tracking-tighter shadow-md">FORCE_BLOCK</span>
                    </div>
                  ))}
                  {rules.length === 0 && <div className="text-sm text-slate-400 italic p-16 text-center border-4 border-dotted border-slate-50 rounded-3xl font-medium">Neural engine is currently operating without external constraints.</div>}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
      
      <footer className="mt-12 pt-8 border-t border-slate-200 text-center text-slate-400 text-[10px] font-black tracking-widest uppercase">
        Securecred Intelligence Protocol v4.0.1 // End-to-End Encryption Enabled
      </footer>
    </div>
  );
}

export default App;
