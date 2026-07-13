const $ = (selector) => document.querySelector(selector);

const state = {
  data: null,
  soundEnabled: localStorage.getItem("heartbeat-sound") !== "off",
  acknowledgedAt: sessionStorage.getItem("heartbeat-acknowledged-at"),
  alarmIncidentAt: null,
  alarmStartedAt: null,
  alarmTimer: null,
  systemVolumeTimer: null,
  audioContext: null,
  alarmNodes: new Set(),
  soundPromptShown: false,
  toastTimer: null,
};

const checkCopy = {
  positions: {
    icon: '<svg viewBox="0 0 24 24"><path d="M4 19V8m5 11V5m5 14v-7m5 7V3"/></svg>',
    passed: "Account positions endpoint responded with a valid payload.",
    skipped: "This diagnostic is not enabled for the current cycle.",
  },
  price_range: {
    icon: '<svg viewBox="0 0 24 24"><path d="M5 17 9 9l4 4 6-8"/><path d="M5 21h14"/></svg>',
    passed: "The latest underlying quote is inside your configured guardrails.",
    skipped: "No price range is configured. Enable this guard in settings.",
  },
  option_preview: {
    icon: '<svg viewBox="0 0 24 24"><path d="M5 4h14v16H5z"/><path d="M8 8h8m-8 4h8m-8 4h4"/></svg>',
    passed: "ATM put selection, quote, and single-leg order preview succeeded.",
    skipped: "This diagnostic is not enabled for the current cycle.",
  },
  balance_drawdown: {
    icon: '<svg viewBox="0 0 24 24"><path d="M4 7h16v12H4z"/><path d="M16 11h4v4h-4a2 2 0 0 1 0-4ZM7 7V5h10v2"/></svg>',
    passed: "Current equity is within the configured daily drawdown limit.",
    skipped: "This diagnostic is not enabled for the current cycle.",
  },
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMoney(value) {
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
}

function formatTime(iso) {
  if (!iso) return "—";
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" }).format(new Date(iso));
}

function relativeTime(iso) {
  if (!iso) return "—";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return formatTime(iso);
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("visible"), 3200);
}

function alarmIsPending() {
  return Boolean(state.alarmIncidentAt);
}

function updateAlarmUi() {
  const pending = alarmIsPending();
  $("#acknowledgeButton").classList.toggle("hidden", !pending);
  document.body.classList.toggle("alarming", pending);
}

function audioIsReady() {
  return state.audioContext?.state === "running";
}

function updateSoundUi() {
  const button = $("#soundButton");
  button.classList.toggle("sound-muted", !state.soundEnabled);
  button.classList.toggle("sound-locked", state.soundEnabled && !audioIsReady());
  button.title = !state.soundEnabled
    ? "Enable browser alarm sound"
    : audioIsReady()
      ? "Browser alarm sound is armed"
      : "Click once to enable browser alarm sound";
}

function getAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return null;
  state.audioContext ??= new AudioContextClass();
  return state.audioContext;
}

function stopAlarmAudio() {
  clearInterval(state.alarmTimer);
  clearInterval(state.systemVolumeTimer);
  state.alarmTimer = null;
  state.systemVolumeTimer = null;
  state.alarmStartedAt = null;
  for (const oscillator of state.alarmNodes) {
    try { oscillator.stop(); } catch (_error) { /* Already stopped. */ }
  }
  state.alarmNodes.clear();
}

function beginAlarmLoop() {
  if (!audioIsReady() || !alarmIsPending() || !state.soundEnabled || state.alarmTimer) return;
  state.alarmStartedAt = Date.now();
  playAlarmBurst();
  state.alarmTimer = setInterval(playAlarmBurst, 1400);
  startSystemVolumeEscalation();
  updateSoundUi();
}

function startSystemVolumeEscalation() {
  if (!alarmIsPending() || !state.soundEnabled || state.systemVolumeTimer) return;
  state.systemVolumeTimer = setInterval(() => {
    api("/api/alarm/volume", { method: "POST", body: JSON.stringify({ step: 5 }) }).catch(() => {});
  }, 15000);
}

function playSoundConfirmation() {
  const audio = state.audioContext;
  if (!audio || audio.state !== "running" || alarmIsPending()) return;
  const start = audio.currentTime + 0.02;
  [660, 880].forEach((frequency, index) => {
    const oscillator = audio.createOscillator();
    const gain = audio.createGain();
    const toneStart = start + index * 0.16;
    oscillator.type = "sine";
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0.0001, toneStart);
    gain.gain.exponentialRampToValueAtTime(0.1, toneStart + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, toneStart + 0.13);
    oscillator.connect(gain).connect(audio.destination);
    oscillator.start(toneStart);
    oscillator.stop(toneStart + 0.14);
  });
}

function unlockBrowserAudio({ confirm = false } = {}) {
  if (!state.soundEnabled) return Promise.resolve(false);
  const audio = getAudioContext();
  if (!audio) {
    showToast("This browser does not support dashboard alarm audio.", true);
    return Promise.resolve(false);
  }
  return audio.resume().then(() => {
    updateSoundUi();
    if (alarmIsPending()) beginAlarmLoop();
    else if (confirm) playSoundConfirmation();
    return audio.state === "running";
  }).catch(() => {
    updateSoundUi();
    return false;
  });
}

function playAlarmBurst() {
  const audio = state.audioContext;
  if (!audio || audio.state !== "running" || !alarmIsPending() || !state.soundEnabled) return;

  const start = audio.currentTime + 0.02;
  const duration = 1.28;
  const elapsed = Math.max(0, Date.now() - (state.alarmStartedAt || Date.now()));
  const escalation = Math.min(1, elapsed / 120000);
  const browserGain = 0.07 + escalation * 0.18;
  const gain = audio.createGain();
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(browserGain, start + 0.04);
  gain.gain.setValueAtTime(browserGain, start + duration - 0.1);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  gain.connect(audio.destination);

  const siren = audio.createOscillator();
  siren.type = "sawtooth";
  siren.frequency.setValueAtTime(540, start);
  siren.frequency.exponentialRampToValueAtTime(940, start + 0.3);
  siren.frequency.exponentialRampToValueAtTime(540, start + 0.62);
  siren.frequency.exponentialRampToValueAtTime(940, start + 0.94);
  siren.frequency.exponentialRampToValueAtTime(540, start + duration);
  siren.connect(gain);

  const undertone = audio.createOscillator();
  const undertoneGain = audio.createGain();
  undertone.type = "square";
  undertone.frequency.setValueAtTime(270, start);
  undertone.frequency.exponentialRampToValueAtTime(470, start + 0.3);
  undertone.frequency.exponentialRampToValueAtTime(270, start + 0.62);
  undertone.frequency.exponentialRampToValueAtTime(470, start + 0.94);
  undertone.frequency.exponentialRampToValueAtTime(270, start + duration);
  undertoneGain.gain.setValueAtTime(0.035, start);
  undertoneGain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  undertone.connect(undertoneGain).connect(audio.destination);

  for (const oscillator of [siren, undertone]) {
    state.alarmNodes.add(oscillator);
    oscillator.onended = () => state.alarmNodes.delete(oscillator);
    oscillator.start(start);
    oscillator.stop(start + duration);
  }
}

function startAlarmAudio() {
  if (!alarmIsPending() || !state.soundEnabled || state.alarmTimer) return;
  if (audioIsReady()) {
    beginAlarmLoop();
    return;
  }
  unlockBrowserAudio();
}

function acknowledgeAlarm() {
  const currentResult = state.data?.last_result;
  state.acknowledgedAt = currentResult?.status !== "healthy"
    ? currentResult.finished_at
    : state.alarmIncidentAt;
  if (state.acknowledgedAt) {
    sessionStorage.setItem("heartbeat-acknowledged-at", state.acknowledgedAt);
  }
  state.alarmIncidentAt = null;
  stopAlarmAudio();
  api("/api/alarm/reset", { method: "POST", body: "{}" }).catch(() => {});
  updateAlarmUi();
  showToast("Alarm acknowledged. Monitoring remains active.");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function renderHero(data) {
  const { service, last_result: result } = data;
  const body = document.body;
  body.classList.toggle("degraded", service.status === "degraded" || service.status === "error");
  body.classList.toggle("checking", service.checking);

  let title = "Heartbeat is starting";
  let description = "The first Tradier health check will begin in a moment.";
  let eyebrow = "Connecting to monitor";

  if (service.checking) {
    title = "Running live diagnostics";
    description = "Checking account connectivity, market data, order preview, and balance safeguards.";
    eyebrow = "Heartbeat cycle in progress";
  } else if (service.status === "healthy") {
    title = "All systems are healthy";
    description = `The latest ${data.config.symbol} heartbeat completed without a detected failure.`;
    eyebrow = "Heartbeat verified";
  } else if (service.status === "degraded") {
    title = "Attention is required";
    description = `${result?.failures?.length || 1} issue${result?.failures?.length === 1 ? "" : "s"} detected in the latest heartbeat cycle.`;
    eyebrow = "Heartbeat degraded";
  } else if (service.status === "error") {
    title = "Monitor encountered an error";
    description = service.last_error || "The heartbeat cycle could not complete.";
    eyebrow = "Runtime error";
  }

  if (!service.monitoring && !service.checking) {
    eyebrow = "Automatic monitoring paused";
    description += " Automatic cycles are paused; manual checks are still available.";
  }
  if (alarmIsPending()) {
    description += " Alarm volume will continue rising until acknowledged.";
  }

  $("#heroTitle").textContent = title;
  $("#heroDescription").textContent = description;
  $("#heroEyebrow").textContent = eyebrow;
  $("#monitorState").textContent = service.checking ? "Checking now" : (service.monitoring ? "Monitoring" : "Paused");
  $("#lastCycle").textContent = result ? relativeTime(result.finished_at) : "Not run yet";
  $("#responseTime").textContent = result ? `${Number(result.duration_seconds).toFixed(2)} sec` : "—";
  $("#monitoringButton").textContent = service.monitoring ? "Pause monitor" : "Resume monitor";
  $("#runButton").disabled = service.checking;
  $("#runButton").lastChild.textContent = service.checking ? " Running checks" : " Run check now";
  updateAlarmUi();
}

function renderCountdown(data) {
  const target = data.service.next_run_at_epoch;
  if (!data.service.monitoring) {
    $("#nextCheck").textContent = "Paused";
    return;
  }
  if (data.service.checking || !target) {
    $("#nextCheck").textContent = data.service.checking ? "In progress" : "Soon";
    return;
  }
  const seconds = Math.max(0, Math.ceil(target - Date.now() / 1000));
  $("#nextCheck").textContent = seconds < 60 ? `${seconds} seconds` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function renderChecks(data) {
  const result = data.last_result;
  if (!result || !result.checks?.length) return;

  const failed = result.checks.filter((item) => item.status === "failed").length;
  const passed = result.checks.filter((item) => item.status === "passed").length;
  $("#checksSummary").textContent = failed ? `${failed} check${failed === 1 ? "" : "s"} need attention` : `${passed} checks passed`;

  $("#checkGrid").innerHTML = result.checks.map((check) => {
    const copy = checkCopy[check.id] || checkCopy.positions;
    const message = check.failures?.[0]?.message || copy[check.status] || "Check completed.";
    return `
      <article class="check-card ${escapeHtml(check.status)}">
        <div class="check-top">
          <span class="check-icon">${copy.icon}</span>
          <span class="status-badge"><i></i>${escapeHtml(check.status)}</span>
        </div>
        <h3>${escapeHtml(check.label)}</h3>
        <p>${escapeHtml(message)}</p>
        <div class="check-meta"><span>${escapeHtml(data.config.symbol)}</span><span>${formatTime(result.finished_at)}</span></div>
      </article>`;
  }).join("");
}

function renderChart(points) {
  const chart = $("#balanceChart");
  if (!points?.length) return;

  const values = points.map((point) => point.value);
  const latest = values.at(-1);
  const previous = values.length > 1 ? values.at(-2) : null;
  const change = previous ? ((latest - previous) / previous) * 100 : null;
  $("#latestEquity").textContent = formatMoney(latest);
  $("#periodChange").textContent = change === null ? "—" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
  $("#periodChange").className = change === null ? "" : (change >= 0 ? "positive" : "negative");

  const width = 700;
  const height = 210;
  const pad = { top: 12, right: 10, bottom: 26, left: 10 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const coords = values.map((value, index) => {
    const x = pad.left + (index / Math.max(1, values.length - 1)) * (width - pad.left - pad.right);
    const y = pad.top + ((max - value) / spread) * (height - pad.top - pad.bottom);
    return [x, y];
  });
  const line = coords.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L${coords.at(-1)[0].toFixed(1)},${height - pad.bottom} L${coords[0][0].toFixed(1)},${height - pad.bottom} Z`;
  const last = coords.at(-1);
  const labels = [0, Math.floor((points.length - 1) / 2), points.length - 1].map((index, labelIndex) => {
    const anchor = labelIndex === 0 ? "start" : labelIndex === 2 ? "end" : "middle";
    return `<text class="chart-label" x="${coords[index][0]}" y="205" text-anchor="${anchor}">${escapeHtml(points[index].date.slice(5))}</text>`;
  }).join("");

  chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Balance history">
    <defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5ce0a4" stop-opacity=".24"/><stop offset="1" stop-color="#5ce0a4" stop-opacity="0"/></linearGradient></defs>
    <line class="chart-grid" x1="0" x2="${width}" y1="${pad.top}" y2="${pad.top}"/>
    <line class="chart-grid" x1="0" x2="${width}" y1="${(height - pad.bottom + pad.top) / 2}" y2="${(height - pad.bottom + pad.top) / 2}"/>
    <line class="chart-grid" x1="0" x2="${width}" y1="${height - pad.bottom}" y2="${height - pad.bottom}"/>
    <path class="chart-area" d="${area}"/><path class="chart-line" d="${line}"/>
    <circle class="chart-dot" cx="${last[0]}" cy="${last[1]}" r="5"/>${labels}</svg>`;
}

function renderActivity(data) {
  $("#runCount").textContent = `${data.service.run_count} run${data.service.run_count === 1 ? "" : "s"}`;
  const history = [...(data.history || [])].reverse().slice(0, 4);
  if (!history.length) return;
  $("#activityList").innerHTML = history.map((item) => {
    const healthy = item.status === "healthy";
    const label = healthy ? "Heartbeat completed" : item.status === "error" ? "Cycle runtime error" : "Issue detected";
    const detail = healthy ? "All enabled checks passed" : `${item.failed_checks} failed check${item.failed_checks === 1 ? "" : "s"}`;
    return `<div class="activity-item ${escapeHtml(item.status)}">
      <span class="activity-symbol">${healthy ? "✓" : "!"}</span>
      <div><strong>${label}</strong><small>${detail} · ${Number(item.duration_seconds).toFixed(2)} sec</small></div>
      <span class="activity-time">${formatTime(item.finished_at)}</span>
    </div>`;
  }).join("");
}

function renderSettings(data, forceFormSync = false) {
  const config = data.config;
  const settingsAreBeingEdited = $("#settingsDialog").open && !forceFormSync;
  if (!settingsAreBeingEdited) {
    $("#intervalInput").value = config.interval_seconds;
    $("#priceRangeEnabled").checked = config.price_range_enabled;
    $("#rangeLowInput").value = config.price_range_low ?? "";
    $("#rangeHighInput").value = config.price_range_high ?? "";
    $("#rangeFields").classList.toggle("disabled", !config.price_range_enabled);
  }
  $("#credentialsState").textContent = config.credentials_configured ? "Ready" : "Missing";
  $("#credentialsDot").classList.toggle("ready", config.credentials_configured);
  $("#emailState").textContent = config.email_configured ? "Ready" : "Not configured";
  $("#emailDot").classList.toggle("ready", config.email_configured);
  $("#footerMeta").textContent = `${config.symbol} · ${config.interval_seconds}s cadence · ${config.sla_seconds}s SLA`;
}

function maybeAlert(data) {
  const result = data.last_result;
  if (
    result
    && result.status !== "healthy"
    && result.finished_at !== state.acknowledgedAt
    && !state.alarmIncidentAt
  ) {
    state.alarmIncidentAt = result.finished_at;
  }
  updateAlarmUi();
  startSystemVolumeEscalation();
  startAlarmAudio();
}

function render(data) {
  state.data = data;
  maybeAlert(data);
  renderHero(data);
  renderCountdown(data);
  renderChecks(data);
  renderChart(data.balance_history);
  renderActivity(data);
  renderSettings(data);
  updateSoundUi();
  if (state.soundEnabled && !audioIsReady() && !state.soundPromptShown) {
    state.soundPromptShown = true;
    showToast("Click the amber speaker icon once to enable browser alarm sound.");
  }
}

async function refresh() {
  try {
    render(await api("/api/status"));
  } catch (error) {
    $("#heroTitle").textContent = "Dashboard connection lost";
    $("#heroDescription").textContent = "The local heartbeat server is not responding. Check the terminal process.";
    document.body.classList.add("degraded");
  }
}

$("#runButton").addEventListener("click", async () => {
  try {
    await api("/api/run", { method: "POST", body: "{}" });
    showToast("Heartbeat cycle requested.");
    await refresh();
  } catch (error) { showToast(error.message, true); }
});

$("#monitoringButton").addEventListener("click", async () => {
  try {
    const enabled = !state.data.service.monitoring;
    await api("/api/monitoring", { method: "POST", body: JSON.stringify({ enabled }) });
    showToast(enabled ? "Automatic monitoring resumed." : "Automatic monitoring paused.");
    await refresh();
  } catch (error) { showToast(error.message, true); }
});

$("#acknowledgeButton").addEventListener("click", acknowledgeAlarm);

$("#soundButton").addEventListener("click", () => {
  if (!state.soundEnabled) {
    state.soundEnabled = true;
    localStorage.setItem("heartbeat-sound", "on");
    unlockBrowserAudio({ confirm: true });
    showToast(alarmIsPending() ? "Browser alarm sound enabled." : "Browser alarm sound armed.");
    return;
  }

  if (!audioIsReady()) {
    unlockBrowserAudio({ confirm: true });
    showToast(alarmIsPending() ? "Browser alarm sound enabled." : "Browser alarm sound armed.");
    return;
  }

  if (alarmIsPending()) {
    showToast("Acknowledge the active alarm before muting browser sound.", true);
    return;
  }

  state.soundEnabled = !state.soundEnabled;
  localStorage.setItem("heartbeat-sound", state.soundEnabled ? "on" : "off");
  updateSoundUi();
  showToast(`Browser alert sound ${state.soundEnabled ? "enabled" : "muted"}.`);
});

$("#settingsButton").addEventListener("click", () => {
  if (state.data) renderSettings(state.data, true);
  $("#settingsDialog").showModal();
});
$("#priceRangeEnabled").addEventListener("change", (event) => {
  $("#rangeFields").classList.toggle("disabled", !event.target.checked);
});

$("#settingsForm").addEventListener("submit", async (event) => {
  if (event.submitter?.value === "cancel") return;
  event.preventDefault();
  const enabled = $("#priceRangeEnabled").checked;
  try {
    await api("/api/config", {
      method: "POST",
      body: JSON.stringify({
        interval_seconds: Number($("#intervalInput").value),
        price_range_enabled: enabled,
        price_range_low: enabled ? Number($("#rangeLowInput").value) : null,
        price_range_high: enabled ? Number($("#rangeHighInput").value) : null,
      }),
    });
    $("#settingsDialog").close();
    showToast("Heartbeat settings saved.");
    await refresh();
  } catch (error) { showToast(error.message, true); }
});

updateSoundUi();
const armBrowserAudio = () => {
  if (state.soundEnabled && !audioIsReady()) unlockBrowserAudio();
};
document.addEventListener("pointerdown", armBrowserAudio, { capture: true, passive: true });
document.addEventListener("keydown", armBrowserAudio, { capture: true });
window.addEventListener("pagehide", () => {
  if (alarmIsPending()) {
    navigator.sendBeacon("/api/alarm/reset", new Blob(["{}"], { type: "application/json" }));
  }
});
refresh();
setInterval(refresh, 2500);
setInterval(() => state.data && renderCountdown(state.data), 1000);
