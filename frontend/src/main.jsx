import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls, Stars, Line, Html, useTexture } from "@react-three/drei";
import * as THREE from "three";
import {
  Activity, AlertTriangle, Database, Download, Gauge, Globe2,
  Pause, Play, Radio, RefreshCw, Search, ShieldCheck,
  SlidersHorizontal, Upload, X, Info, Satellite, Zap, Eye
} from "lucide-react";
import "./styles.css";

/* ───────────── CONFIG ───────────── */
const API = import.meta.env.VITE_API_URL || "http://localhost:8001";

const DEMO_TLE_OPTIONS = [
  { value: "scenario-a", label: "Scenario A · ISS group" },
  { value: "scenario-b", label: "Scenario B · CSS group" },
  { value: "scenario-c", label: "Scenario C · close approach" },
  { value: "scenario-d", label: "Scenario D · mixed orbits" },
];
const ORBIT_COLORS = ["#28c7ff", "#ffb84d", "#c084fc", "#66e3a4"];

/* ───────────── ORBIT / COORDINATE HELPERS ───────────── */
const EARTH_RADIUS_KM = 6371;
const EARTH_UNITS = 3;

function temeToScene(positionKm) {
  if (!positionKm || positionKm.length !== 3) return null;
  const r = Math.hypot(positionKm[0], positionKm[1], positionKm[2]);
  if (!Number.isFinite(r) || r < 1) return null;
  const units = EARTH_UNITS * (r / EARTH_RADIUS_KM);
  return [
    (positionKm[0] / r) * units,
    (positionKm[2] / r) * units,
    (positionKm[1] / r) * units,
  ];
}

function trajectoryLinePoints(states, line2) {
  if (!states.length) return null;
  const line2Parts = line2?.trim().split(/\s+/) || [];
  const meanMotion = Number(line2Parts[7]);
  const periodMs = Number.isFinite(meanMotion) && meanMotion > 0
    ? (1440 / meanMotion) * 60 * 1000
    : 2 * 60 * 60 * 1000;
  const startMs = new Date(states[0].timestamp).getTime();
  const visibleStates = states.filter((state) =>
    new Date(state.timestamp).getTime() <= startMs + periodMs
  );
  const every = Math.max(1, Math.ceil(visibleStates.length / 240));
  const points = [];
  for (let i = 0; i < visibleStates.length; i += every) {
    const p = temeToScene(visibleStates[i].position_km);
    if (p) points.push(p);
  }
  const last = temeToScene(visibleStates[visibleStates.length - 1]?.position_km);
  if (last) points.push(last);
  return points.length > 1 ? points : null;
}

/* ───────────── 3D EARTH WITH TEXTURE ───────────── */
function Earth() {
  const meshRef = useRef();
  // Use a procedural Earth with continent outlines
  const canvas = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = 1024;
    c.height = 512;
    const ctx = c.getContext("2d");

    // Deep ocean gradient
    const grad = ctx.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0, "#0a1628");
    grad.addColorStop(0.3, "#0d1f3c");
    grad.addColorStop(0.5, "#0a1830");
    grad.addColorStop(0.7, "#0d1f3c");
    grad.addColorStop(1, "#0a1628");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 1024, 512);

    // Simplified continent shapes as bright land patches
    ctx.fillStyle = "rgba(30, 90, 150, 0.35)";
    // North America
    ctx.beginPath(); ctx.ellipse(250, 150, 80, 55, -0.2, 0, Math.PI * 2); ctx.fill();
    // South America
    ctx.beginPath(); ctx.ellipse(320, 310, 40, 70, 0.15, 0, Math.PI * 2); ctx.fill();
    // Europe/Africa
    ctx.beginPath(); ctx.ellipse(530, 180, 35, 45, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(540, 300, 40, 60, 0, 0, Math.PI * 2); ctx.fill();
    // Asia
    ctx.beginPath(); ctx.ellipse(680, 150, 100, 55, 0, 0, Math.PI * 2); ctx.fill();
    // Australia
    ctx.beginPath(); ctx.ellipse(800, 340, 40, 25, 0.1, 0, Math.PI * 2); ctx.fill();

    // Bright coastline edges
    ctx.strokeStyle = "rgba(60, 180, 255, 0.18)";
    ctx.lineWidth = 1.5;
    // NA
    ctx.beginPath(); ctx.ellipse(250, 150, 80, 55, -0.2, 0, Math.PI * 2); ctx.stroke();
    // SA
    ctx.beginPath(); ctx.ellipse(320, 310, 40, 70, 0.15, 0, Math.PI * 2); ctx.stroke();
    // EU
    ctx.beginPath(); ctx.ellipse(530, 180, 35, 45, 0, 0, Math.PI * 2); ctx.stroke();
    // AF
    ctx.beginPath(); ctx.ellipse(540, 300, 40, 60, 0, 0, Math.PI * 2); ctx.stroke();
    // AS
    ctx.beginPath(); ctx.ellipse(680, 150, 100, 55, 0, 0, Math.PI * 2); ctx.stroke();
    // AU
    ctx.beginPath(); ctx.ellipse(800, 340, 40, 25, 0.1, 0, Math.PI * 2); ctx.stroke();

    return c;
  }, []);

  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(canvas);
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  }, [canvas]);

  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 0.02;
  });

  return (
    <group ref={meshRef}>
      {/* Main globe body */}
      <mesh>
        <sphereGeometry args={[3, 64, 64]} />
        <meshStandardMaterial
          map={texture}
          metalness={0.2}
          roughness={0.8}
          color="#1a4a7a"
        />
      </mesh>
      {/* Inner atmosphere glow */}
      <mesh>
        <sphereGeometry args={[3.08, 48, 48]} />
        <meshBasicMaterial
          color="#4dc9f6"
          transparent
          opacity={0.06}
          side={THREE.BackSide}
        />
      </mesh>
      {/* Outer atmosphere glow */}
      <mesh>
        <sphereGeometry args={[3.25, 48, 48]} />
        <meshBasicMaterial
          color="#1e90ff"
          transparent
          opacity={0.035}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
}

/* ───────────── SATELLITE MARKER ───────────── */
function SatelliteMarker({ position, label, risk, selected, onClick }) {
  const ref = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (ref.current) {
      // Gentle pulse animation
      const s = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
      ref.current.scale.setScalar(selected ? s * 1.3 : hovered ? s * 1.1 : s);
    }
  });

  const color = risk ? "#ff4466" : "#00ccff";
  const glowColor = risk ? "#ff2244" : "#00aaff";

  return (
    <group
      position={position}
      onClick={(e) => { e.stopPropagation(); onClick?.(); }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* Core dot */}
      <mesh ref={ref}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {/* Inner glow ring */}
      <mesh>
        <sphereGeometry args={[0.11, 16, 16]} />
        <meshBasicMaterial color={glowColor} transparent opacity={0.2} />
      </mesh>
      {/* Outer glow */}
      <mesh>
        <sphereGeometry args={[0.18, 16, 16]} />
        <meshBasicMaterial color={glowColor} transparent opacity={0.06} />
      </mesh>
      {/* Label - only show on hover or selected */}
      {(hovered || selected) && (
        <Html distanceFactor={7} style={{ pointerEvents: "none" }}>
          <div className={`sat-label ${risk ? "danger" : "safe"} ${selected ? "selected" : ""}`}>
            <span className="sat-label-dot" style={{ background: color }} />
            {label}
          </div>
        </Html>
      )}
    </group>
  );
}

/* ───────────── MAIN 3D SCENE ───────────── */
function SpaceScene({ trajectories, selected, setSelected, playTime }) {
  const live = trajectories?.length ? trajectories : [];

  return (
    <>
      <color attach="background" args={["#020810"]} />

      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 3, 5]} intensity={2.5} color="#e0f0ff" />
      <pointLight position={[-4, -2, -3]} intensity={5} color="#6b4cff" />
      <pointLight position={[3, 5, -2]} intensity={3} color="#00aaff" />

      {/* Stars */}
      <Stars radius={120} depth={50} count={4000} factor={2.5} saturation={0.1} fade speed={0.08} />

      {/* Earth */}
      <Earth />

      {/* Real trajectories */}
      {live.map((trajectory, index) => {
        const satellite = trajectory.satellite || {};
        const satelliteId = String(satellite.norad_id ?? index);
        const satelliteName = satellite.name || `SAT-${satelliteId}`;
        const linePoints = trajectoryLinePoints(trajectory.states || [], satellite.line2);
        if (!linePoints) return null;

        const stateCount = trajectory.states.length;
        const animIndex = Math.floor(
          ((playTime * 0.00018 * (index + 1)) % 1) * (stateCount - 1)
        );
        const markerPos =
          temeToScene(trajectory.states[animIndex]?.position_km) || linePoints[0];

        return (
          <group key={`satellite-${satelliteId}`}>
            {/* Orbit path */}
            <Line
              points={linePoints}
              color={ORBIT_COLORS[index % ORBIT_COLORS.length]}
              transparent
              opacity={0.7}
              lineWidth={0.45}
            />
            {/* Satellite marker */}
            <SatelliteMarker
              position={markerPos}
              label={satelliteName}
              risk={false}
              selected={selected === satelliteName}
              onClick={() => setSelected(satelliteName)}
            />
          </group>
        );
      })}

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={5.5}
        maxDistance={16}
        rotateSpeed={0.5}
      />
    </>
  );
}

/* ───────────── GLOBE CONTAINER ───────────── */
function SpaceView({ trajectories, selected, setSelected, playTime }) {
  return (
    <Canvas
      camera={{ position: [7, 4, 7], fov: 42 }}
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: "high-performance", alpha: false }}
    >
      <SpaceScene
        trajectories={trajectories}
        selected={selected}
        setSelected={setSelected}
        playTime={playTime}
      />
    </Canvas>
  );
}

/* ───────────── LEGEND PANEL (overlays the globe) ───────────── */
function GlobeLegend({ trajectories, conjunctions }) {
  const satCount = trajectories?.length || 0;
  return (
    <div className="globe-legend">
      <div className="legend-title"><Eye size={12}/> LEGEND</div>
      <div className="legend-item">
        <span className="legend-dot safe-dot"/>
        <span>Safe Satellite</span>
      </div>
      <div className="legend-item">
        <span className="legend-line safe-line"/>
        <span>Orbit Path</span>
      </div>
      <div className="legend-separator"/>
      <div className="legend-stat">{satCount} satellite{satCount !== 1 ? "s" : ""} tracked</div>
      <div className="legend-stat">Orbit paths only · risk in table</div>
    </div>
  );
}

/* ───────────── HELPER COMPONENTS ───────────── */
function Stat({ icon: Icon, label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-icon"><Icon size={16} /></div>
      <div>
        <span>{label}</span>
        <b>{value}</b>
        {sub && <small>{sub}</small>}
      </div>
    </div>
  );
}

function riskOrder(x) { return { RED: 0, YELLOW: 1, GREEN: 2 }[x] ?? 9; }

/* ───────────── MAIN APP ───────────── */
function App() {
  const [tle, setTle] = useState("");
  const [tleExample, setTleExample] = useState(DEMO_TLE_OPTIONS[0].value);
  const [radius, setRadius] = useState(100);
  const [step, setStep] = useState(60);
  const [hours, setHours] = useState(2);
  const [status, setStatus] = useState("READY");
  const [result, setResult] = useState(null);
  const [trajectories, setTrajectories] = useState([]);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");
  const [sort, setSort] = useState("risk");
  const [selected, setSelected] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [playTime, setPlayTime] = useState(0);
  const [remoteGroup, setRemoteGroup] = useState("stations");
  const [loadingTle, setLoadingTle] = useState(false);
  const fileRef = useRef();

  // Tab state
  const [activeTab, setActiveTab] = useState(
    window.location.hash === "#model" ? "ML" : "SGP4"
  );

  // CDM ML prediction state
  const [cdmResult, setCdmResult] = useState(null);
  const [cdmStatus, setCdmStatus] = useState("READY");
  const [cdmError, setCdmError] = useState("");
  const [cdmDragOver, setCdmDragOver] = useState(false);
  const cdmFileRef = useRef();

  function navigateTo(mode) {
    const hash = mode === "ML" ? "#model" : "#sgp4";
    if (window.location.hash !== hash) window.history.pushState({}, "", hash);
    setActiveTab(mode);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  useEffect(() => {
    const syncPage = () => setActiveTab(window.location.hash === "#model" ? "ML" : "SGP4");
    window.addEventListener("popstate", syncPage);
    window.addEventListener("hashchange", syncPage);
    return () => {
      window.removeEventListener("popstate", syncPage);
      window.removeEventListener("hashchange", syncPage);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`/demo/tle/${tleExample}.tle`)
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load the selected TLE example");
        return response.text();
      })
      .then((text) => { if (!cancelled) setTle(text); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [tleExample]);

  // Animation loop
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setPlayTime((t) => t + 1000), 80);
    return () => clearInterval(id);
  }, [playing]);

  /* ── SGP4 simulation ── */
  async function run() {
    setStatus("RUNNING"); setError(""); setResult(null); setTrajectories([]); setPlayTime(0);
    const start = new Date(Date.now() + 60000);
    const end = new Date(start.getTime() + Number(hours) * 3600000);
    try {
      const r = await fetch(`${API}/api/simulations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tle_text: tle,
          start_time: start.toISOString(),
          end_time: end.toISOString(),
          time_step_seconds: Number(step),
          safety_radius_km: Number(radius),
          ml_enabled: false,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw Error(d.detail || "Simulation failed");
      setResult(d); setStatus("COMPLETED"); setPlaying(false);
      const tr = await fetch(`${API}/api/simulations/${d.simulation_id}/trajectories`);
      if (tr.ok) {
        const td = await tr.json();
        setTrajectories(td.trajectories || []);
      }
    } catch (e) { setError(e.message); setStatus("ERROR"); }
  }

  /* ── TLE fetch from remote ── */
  async function fetchRemote() {
    setLoadingTle(true); setError("");
    try {
      const r = await fetch(`${API}/api/tle/fetch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ group: remoteGroup }),
      });
      const d = await r.json();
      if (!r.ok) throw Error(d.detail || "TLE fetch failed");
      setTle(d.records.map((x) => `${x.name}\n${x.line1}\n${x.line2}`).join("\n"));
    } catch (e) { setError(e.message); }
    finally { setLoadingTle(false); }
  }

  async function importFile(e) {
    const f = e.target.files?.[0];
    if (f) setTle(await f.text());
  }

  /* ── CDM upload ── */
  async function uploadCdm(file) {
    if (!file) return;
    setCdmStatus("ANALYZING"); setCdmError(""); setCdmResult(null);
    const form = new FormData(); form.append("file", file);
    try {
      const r = await fetch(`${API}/api/cdm/predict`, { method: "POST", body: form });
      const d = await r.json();
      if (!r.ok) throw Error(d.detail || "Prediction failed");
      setCdmResult(d); setCdmStatus("COMPLETED");
    } catch (e) { setCdmError(e.message); setCdmStatus("ERROR"); }
  }
  function handleCdmFile(e) { const f = e.target.files?.[0]; if (f) uploadCdm(f); }
  function handleCdmDrop(e) { e.preventDefault(); setCdmDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) uploadCdm(f); }
  async function loadDemoCdm() {
    try {
      const response = await fetch("/demo/model_demo.csv");
      if (!response.ok) throw new Error("Unable to load the model demo CSV");
      uploadCdm(new File([await response.blob()], "model_demo.csv", { type: "text/csv" }));
    } catch (e) { setCdmError(e.message); setCdmStatus("ERROR"); }
  }

  /* ── Table rows ── */
  const rows = useMemo(() => {
    const nameById = {};
    trajectories.forEach((t) => {
      if (t.satellite) nameById[String(t.satellite.norad_id)] = t.satellite.name || `SAT-${t.satellite.norad_id}`;
    });
    const pairLabel = (id) => nameById[String(id)] || `NORAD ${id}`;

    let a = activeTab === "SGP4" ? [...(result?.conjunctions || [])].map((x) => ({
      id: `${pairLabel(x.satellite_a)} ↔ ${pairLabel(x.satellite_b)}`,
      source: "SGP4", tca: x.tca, miss: x.miss_distance_km, relV: x.relative_velocity_km_s,
      pc: x.pc, mlPc: null, confidence: "MEDIUM", risk: x.risk_level, review: false,
    })) : [];
    let b = activeTab === "ML" ? [...(cdmResult?.events || [])].map((x) => ({
      id: `CDM Event ${x.event_id}`,
      source: "XGBoost", tca: null, miss: x.miss_distance_km < 900 ? x.miss_distance_km : null, relV: null,
      pc: null, mlPc: x.probability, confidence: x.confidence, risk: x.risk_level, review: x.needs_review,
    })) : [];

    let combined = [...a, ...b];
    combined = combined.filter((x) => x.id.toLowerCase().includes(query.toLowerCase()));
    if (riskFilter !== "ALL") combined = combined.filter((x) => x.risk === riskFilter);
    combined.sort((x, y) =>
      sort === "distance" ? (x.miss || 9999) - (y.miss || 9999) :
      sort === "tca" ? new Date(x.tca || 0) - new Date(y.tca || 0) :
      riskOrder(x.risk) - riskOrder(y.risk)
    );
    return combined;
  }, [activeTab, result, cdmResult, query, riskFilter, sort, trajectories]);

  const activeRisk = rows.some((x) => x.risk === "RED") ? "RED" :
    rows.some((x) => x.risk === "YELLOW") ? "YELLOW" : "GREEN";

  /* ───────── RENDER ───────── */
  return (
    <div className="app">
      {/* HEADER */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Satellite size={20} /></div>
          <div>
            <strong>SATNET</strong>
            <small>COLLISION RISK ANALYSIS</small>
          </div>
        </div>
        <div className="top-right">
          <nav className="page-nav" aria-label="Application pages">
            <button className={activeTab === "SGP4" ? "active" : ""} onClick={() => navigateTo("SGP4")}>
              <Zap size={12} /> SGP4 PAGE
            </button>
            <button className={activeTab === "ML" ? "active" : ""} onClick={() => navigateTo("ML")}>
              <Gauge size={12} /> MODEL PAGE
            </button>
          </nav>
          <span className="pulse" /> SYSTEM ONLINE
          <i /> SGP4 + XGBoost ML
          <i /> V2.0
        </div>
      </header>

      <main>
        {/* ── HERO: Globe + Intro ── */}
        <section className={`hero ${activeTab === "ML" ? "model-hero" : ""}`}>
          <div className="hero-copy">
            <div className="eyebrow"><ShieldCheck size={13} /> HYBRID COLLISION INTELLIGENCE</div>
            <h1>{activeTab === "SGP4" ? <>See the orbit.<br /><em>Screen the risk.</em></> : <>Read the data.<br /><em>Predict the risk.</em></>}</h1>
            <p>
              {activeTab === "SGP4"
                ? "SGP4 propagation screens TLE trajectories and calculates geometric close approaches."
                : "XGBoost classifies conjunction risk from uploaded Conjunction Data Message observations."}
            </p>
            <div className="stats">
              {activeTab === "SGP4" ? (
                <>
                  <Stat icon={Database} label="SATELLITES" value={result?.satellite_count ?? "—"} />
                  <Stat icon={Gauge} label="CANDIDATES" value={result?.candidate_pairs ?? "—"} />
                  <Stat icon={AlertTriangle} label="CONJUNCTIONS" value={result?.conjunction_count ?? "—"} />
                  <Stat icon={Activity} label="ENGINE" value="SGP4" sub="Physics" />
                </>
              ) : (
                <>
                  <Stat icon={Database} label="EVENTS" value={cdmResult?.total_events ?? "—"} />
                  <Stat icon={AlertTriangle} label="RED" value={cdmResult?.red_count ?? "—"} />
                  <Stat icon={Gauge} label="YELLOW" value={cdmResult?.yellow_count ?? "—"} />
                  <Stat icon={Activity} label="ENGINE" value="XGBOOST" sub="CDM" />
                </>
              )}
            </div>
          </div>

          {activeTab === "SGP4" ? (
            <div className="scene">
              <div className="scene-head">
                <span>
                  <span className="pulse" />
                  {trajectories.length
                    ? `${trajectories.length} SATELLITES · LIVE TEME TRAJECTORIES`
                    : "READY · SELECT TLE DATASET AND RUN ANALYSIS"}
                </span>
                <span>DRAG TO ROTATE · SCROLL TO ZOOM · CLICK SATELLITE</span>
              </div>
              <GlobeLegend trajectories={trajectories} conjunctions={result?.conjunctions || []} />
              <SpaceView
                trajectories={trajectories}
                conjunctions={result?.conjunctions || []}
                selected={selected}
                setSelected={setSelected}
                playTime={playTime}
              />
            </div>
          ) : null}
        </section>

        {/* ── CONTROLS + TABLE ── */}
        <section className="workspace">
          <aside className="control panel">
            {activeTab === "SGP4" ? (
              <>
                <div className="panel-kicker"><SlidersHorizontal size={13} /> SIMULATION CONTROL</div>

                <div className="field">
                  <label>TLE SOURCE</label>
                  <div className="source-row">
                    <select value={tleExample} onChange={(e) => setTleExample(e.target.value)}>
                      {DEMO_TLE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                    <select value={remoteGroup} onChange={(e) => setRemoteGroup(e.target.value)}>
                      <option value="stations">Stations</option>
                      <option value="starlink">Starlink</option>
                      <option value="active">Active</option>
                      <option value="weather">Weather</option>
                      <option value="gps-ops">GPS Operational</option>
                    </select>
                    <button className="ghost" onClick={fetchRemote} disabled={loadingTle}>
                      {loadingTle ? <RefreshCw className="spin" size={14} /> : <Globe2 size={14} />} FETCH
                    </button>
                  </div>
                </div>

                <div className="field">
                  <label>TLE DATASET</label>
                  <textarea value={tle} onChange={(e) => setTle(e.target.value)} spellCheck="false" />
                  <input ref={fileRef} hidden type="file" accept=".tle,.txt" onChange={importFile} />
                  <button className="ghost wide" onClick={() => fileRef.current?.click()}>
                    <Upload size={14} /> IMPORT TLE FILE
                  </button>
                </div>

                <div className="grid2">
                  <div className="field">
                    <label>TIME STEP / SEC</label>
                    <input type="number" min="1" value={step} onChange={(e) => setStep(e.target.value)} />
                  </div>
                  <div className="field">
                    <label>WINDOW / HOURS</label>
                    <input type="number" min="1" max="48" value={hours} onChange={(e) => setHours(e.target.value)} />
                  </div>
                </div>

                <div className="field">
                  <label>SCREENING RADIUS / KM <b>{radius}</b></label>
                  <input className="range" type="range" min="1" max="200" value={radius} onChange={(e) => setRadius(e.target.value)} />
                </div>

                <div className="how-to-box">
                  <Info size={13} />
                  <div>
                    <strong>How to use:</strong> The TLE data is pre-loaded with 4 satellites.
                    Set screening radius to <b>100 km</b>, then click <b>RUN ANALYSIS</b>.
                    The globe will animate real SGP4-propagated orbits. Red lines show conjunction risks.
                  </div>
                </div>

                <button className="run" onClick={run} disabled={status === "RUNNING"}>
                  {status === "RUNNING" ? <RefreshCw className="spin" size={16} /> : <Play size={16} fill="currentColor" />}
                  {status === "RUNNING" ? " RUNNING ANALYSIS..." : " RUN ANALYSIS"}
                </button>

                <div className={`status ${status.toLowerCase()}`}>
                  <Activity size={14} /><span>{status}</span>
                  {result && <small>{result.satellite_count} sats · {result.duration_seconds / 3600} h</small>}
                </div>
                {result?.warning && <div className="warning"><AlertTriangle size={14} />{result.warning}</div>}
                {error && <div className="error"><X size={14} />{error}</div>}
              </>
            ) : (
              <>
                <div className="panel-kicker"><Gauge size={13} /> ML RISK PREDICTION (CDM)</div>

                <div className="how-to-box">
                  <Info size={13} />
                  <div>
                    <strong>How to use:</strong> Upload a CDM (Conjunction Data Message) CSV file.
                    The XGBoost model will classify each event as RED / YELLOW / GREEN risk.
                  </div>
                </div>

                <div
                  className={`cdm-drop ${cdmDragOver ? "drag-over" : ""} ${cdmStatus === "ANALYZING" ? "analyzing" : ""}`}
                  onDragOver={(e) => { e.preventDefault(); setCdmDragOver(true); }}
                  onDragLeave={() => setCdmDragOver(false)}
                  onDrop={handleCdmDrop}
                  onClick={() => cdmFileRef.current?.click()}
                >
                  <input ref={cdmFileRef} hidden type="file" accept=".csv" onChange={handleCdmFile} />
                  {cdmStatus === "ANALYZING" ? (
                    <div className="cdm-drop-inner">
                      <RefreshCw className="spin" size={24} />
                      <span>Analyzing CDM data...</span>
                    </div>
                  ) : (
                    <div className="cdm-drop-inner">
                      <Upload size={24} />
                      <span>Drop CDM CSV here</span>
                      <small>or click to browse</small>
                    </div>
                  )}
                </div>
                <button className="ghost wide" onClick={loadDemoCdm} disabled={cdmStatus === "ANALYZING"}>
                  <Database size={14} /> LOAD MODEL DEMO CSV
                </button>

                {cdmError && <div className="error" style={{ marginTop: 8 }}><X size={14} />{cdmError}</div>}
                {cdmResult && (
                  <div className="cdm-summary">
                    <div className="cdm-badges">
                      <span className="cdm-badge red">{cdmResult.red_count} RED</span>
                      <span className="cdm-badge yellow">{cdmResult.yellow_count} YELLOW</span>
                      <span className="cdm-badge green">{cdmResult.green_count} GREEN</span>
                    </div>
                    <small>{cdmResult.total_events} events analyzed via XGBoost</small>
                  </div>
                )}
              </>
            )}
          </aside>

          {/* ── RISK TABLE ── */}
          <section className="panel intelligence">
            <div className="table-head">
              <div>
                <div className="panel-kicker">{activeTab === "SGP4" ? "SGP4 CONJUNCTION SCREENING" : "XGBOOST MODEL RESULTS"}</div>
                  <h2>{activeTab === "SGP4" ? "Conjunction Events" : "Predicted Risk Events"}</h2>
              </div>
              <div className="table-tools">
                <div className="search">
                  <Search size={14} />
                  <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search events..." />
                </div>
                <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
                  <option>ALL</option><option>RED</option><option>YELLOW</option><option>GREEN</option>
                </select>
                <select value={sort} onChange={(e) => setSort(e.target.value)}>
                  <option value="risk">Sort: Risk</option>
                  <option value="distance">Sort: Distance</option>
                </select>
              </div>
            </div>

            <div className="risk-strip">
              <span className={`risk-dot ${activeRisk.toLowerCase()}`} />
              <b>{rows.length} event{rows.length === 1 ? "" : "s"} detected</b>
              <span>{activeTab === "SGP4" ? "Physics-based screening from propagated TLE trajectories" : "Risk classification from uploaded CDM observations"}</span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>EVENT / PAIR</th>
                    <th>ENGINE</th>
                    <th>MISS DISTANCE</th>
                    <th>PHYSICS Pc</th>
                    <th>ML Pc</th>
                    <th>CONFIDENCE</th>
                    <th>RISK</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.length ? rows.map((r, i) => (
                    <tr key={i} className={r.risk.toLowerCase()}>
                      <td>
                        <strong>{r.id}</strong>
                        {r.review && (
                          <span className="review-badge">
                            <AlertTriangle size={10} /> REVIEW
                          </span>
                        )}
                      </td>
                      <td>
                        <span className={`engine-badge ${r.source === "XGBoost" ? "ml" : "physics"}`}>
                          {r.source}
                        </span>
                      </td>
                      <td>{r.miss != null ? `${r.miss.toFixed(3)} km` : "N/A"}</td>
                      <td className="na">{r.pc == null ? "N/A" : Number(r.pc).toExponential(2)}</td>
                      <td className="na">{r.mlPc == null ? "N/A" : Number(r.mlPc).toExponential(2)}</td>
                      <td>
                        <span className={`confidence-badge ${r.confidence.toLowerCase()}`}>
                          {r.confidence}
                        </span>
                      </td>
                      <td><span className={`risk-pill ${r.risk.toLowerCase()}`}>{r.risk}</span></td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="7" className="empty">
                        <ShieldCheck size={28} />
                        <b>No Events Detected</b>
                        <span>Run an SGP4 simulation or upload CDM data to detect collision risks.</span>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {(result || cdmResult) && (
              <div className="result-foot">
                <span>
                  {result
                    ? `Simulation ${result.simulation_id.slice(0, 8)} · ${result.processing_time_seconds.toFixed(3)}s`
                    : "ML Prediction Complete"}
                </span>
                <div>
                  <a href={`${API}/api/reports/${result?.simulation_id || "latest"}/csv`}>
                    <Download size={13} /> Export CSV
                  </a>
                </div>
              </div>
            )}
          </section>
        </section>

        {/* ── PLAYBACK BAR ── */}
        {activeTab === "SGP4" && <section className="playback panel">
          <div>
            <div className="panel-kicker">MISSION TIMELINE</div>
            <b>{playing ? "▶ PLAYBACK ACTIVE" : "READY FOR PLAYBACK"}</b>
            <span> · {selected || "Hover a satellite to select"}</span>
          </div>
          <div className="play-controls">
            <button onClick={() => setPlaying(!playing)}>
              {playing ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <input type="range" min="0" max="100000" value={playTime % 100000} onChange={(e) => setPlayTime(+e.target.value)} />
            <span>T+ {(playTime / 1000).toFixed(0)}s</span>
          </div>
        </section>}
      </main>

      <footer>
        <span>SATNET 2.0</span>
        <span>SGP4 · XGBOOST · HYBRID FUSION · WEBGL</span>
        <span>ANALYSIS ONLY · NO AUTONOMOUS CONTROL</span>
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
