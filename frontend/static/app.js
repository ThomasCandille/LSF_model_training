const targetEl = document.getElementById("target");
const scoreEl = document.getElementById("score");
const predictionEl = document.getElementById("prediction");
const lastDetectedEl = document.getElementById("last-detected");
const statusEl = document.getElementById("status");
const feedbackEl = document.getElementById("feedback");
const historyBarEl = document.getElementById("history-bar");
const pauseBarEl = document.getElementById("pause-bar");

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const state = await response.json();
    targetEl.textContent = (state.target || "-").toUpperCase();
    scoreEl.textContent = String(state.score ?? 0);
    predictionEl.textContent = state.prediction || "-";
    lastDetectedEl.textContent = `Dernier : ${state.last_detected_label || "-"} (${(state.last_detected_conf ?? 0).toFixed(2)})`;
    statusEl.textContent = state.status || "En attente...";

    const historyRatio = clamp01(Number(state.history_ratio || 0));
    historyBarEl.style.width = `${historyRatio * 100}%`;

    const pauseRatio = clamp01(1 - (Number(state.analysis_pause_left || 0) / 1.2));
    pauseBarEl.style.width = `${pauseRatio * 100}%`;

    const feedback = state.feedback || "";
    feedbackEl.textContent = feedback;
    feedbackEl.classList.remove("ok", "warn");
    if (feedback.toLowerCase().includes("réussi")) {
      feedbackEl.classList.add("ok");
    } else if (feedback.length > 0) {
      feedbackEl.classList.add("warn");
    }
  } catch (error) {
    statusEl.textContent = "Connexion perdue. Nouvelle tentative...";
  }
}

setInterval(refreshState, 140);
refreshState();
