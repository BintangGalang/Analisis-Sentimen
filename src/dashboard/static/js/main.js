/**
 * src/dashboard/static/js/main.js
 * ===============================
 * Logika interaktif Dashboard Reputasi Undiksha:
 * - Dynamic Chart.js Rendering
 * - AJAX Filter & Real-Time KPI Update
 * - Live Sentiment Predictor
 * - Data Explorer & Search
 */

let sentimentChart = null;
let aspectChart = null;
let modelChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initSidebar();
  initTabs();
  initFilters();
  initPredictor();
  initSearch();
  loadDashboardData();
  loadModelComparison();
  loadCommentsTable();
});

/* ─────────────────────────────────────────────────────────────
   1. Sidebar Navigation & Mobile Drawer
   ───────────────────────────────────────────────────────────── */
function initSidebar() {
  const sidebar = document.getElementById("app-sidebar");
  const toggleBtn = document.getElementById("btn-sidebar-toggle");
  const closeBtn = document.getElementById("btn-sidebar-close");
  const backdrop = document.getElementById("sidebar-backdrop");

  const openSidebar = () => {
    if (sidebar) sidebar.classList.add("open");
    if (backdrop) backdrop.classList.add("active");
  };

  const closeSidebar = () => {
    if (sidebar) sidebar.classList.remove("open");
    if (backdrop) backdrop.classList.remove("active");
  };

  if (toggleBtn) toggleBtn.addEventListener("click", openSidebar);
  if (closeBtn) closeBtn.addEventListener("click", closeSidebar);
  if (backdrop) backdrop.addEventListener("click", closeSidebar);
}

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const crumbTitle = document.getElementById("active-crumb-title");
  const sidebar = document.getElementById("app-sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      // Update breadcrumb title
      const label = btn.querySelector(".nav-label")?.textContent?.trim() || btn.textContent.trim();
      if (crumbTitle && label) {
        crumbTitle.textContent = label;
      }

      // Close sidebar on mobile upon item selection
      if (window.innerWidth <= 992) {
        if (sidebar) sidebar.classList.remove("open");
        if (backdrop) backdrop.classList.remove("active");
      }
    });
  });
}

/* ─────────────────────────────────────────────────────────────
   2. Filter Event Listeners
   ───────────────────────────────────────────────────────────── */
function initFilters() {
  const platformSel = document.getElementById("filter-platform");
  const aspectSel = document.getElementById("filter-aspect");
  const sentimentSel = document.getElementById("filter-sentiment");

  const onFilterChange = () => {
    loadDashboardData();
    loadCommentsTable();
  };

  if (platformSel) platformSel.addEventListener("change", onFilterChange);
  if (aspectSel) aspectSel.addEventListener("change", onFilterChange);
  if (sentimentSel) sentimentSel.addEventListener("change", onFilterChange);
}

function getFilterParams() {
  const p = document.getElementById("filter-platform")?.value || "all";
  const a = document.getElementById("filter-aspect")?.value || "all";
  const s = document.getElementById("filter-sentiment")?.value || "all";
  return `platform=${encodeURIComponent(p)}&aspect=${encodeURIComponent(a)}&sentiment=${encodeURIComponent(s)}`;
}

/* ─────────────────────────────────────────────────────────────
   3. Muat Data Statistik & Update Chart.js
   ───────────────────────────────────────────────────────────── */
async function loadDashboardData() {
  const params = getFilterParams();
  try {
    const res = await fetch(`/api/stats?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Gagal memuat data statistik");

    // Update KPI Cards (Null-safe)
    const setText = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.innerText = val;
    };

    setText("kpi-total", Number(data.total).toLocaleString());
    setText("kpi-rep-idx", data.reputation_index_pct);
    setText("kpi-predikat", data.predikat);
    setText("kpi-pos", `${data.positif} (${data.positif_pct}%)`);
    setText("kpi-net", `${data.netral} (${data.netral_pct}%)`);
    setText("kpi-neg", `${data.negatif} (${data.negatif_pct}%)`);

    // Render / Update Donut Chart
    renderSentimentDonut(data);

    // Render / Update Aspect Reputation Bar Chart
    renderAspectChart(data.aspect_stats || []);
  } catch (err) {
    console.error("Error loading dashboard data:", err);
  }
}

function renderSentimentDonut(data) {
  const ctx = document.getElementById("chart-sentiment-donut")?.getContext("2d");
  if (!ctx) return;

  if (sentimentChart) {
    sentimentChart.destroy();
  }

  sentimentChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Positif", "Netral", "Negatif"],
      datasets: [
        {
          data: [data.positif, data.netral, data.negatif],
          backgroundColor: ["#10b981", "#64748b", "#ef4444"],
          borderWidth: 2,
          borderColor: "#ffffff",
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { family: "'Plus Jakarta Sans', sans-serif", weight: 600 } },
        },
        tooltip: {
          callbacks: {
            label: (item) => {
              const total = data.total || 1;
              const val = item.raw;
              const pct = ((val / total) * 100).toFixed(1);
              return ` ${item.label}: ${val} (${pct}%)`;
            },
          },
        },
      },
      cutout: "68%",
    },
  });
}

function renderAspectChart(aspectStats) {
  const ctx = document.getElementById("chart-aspect-bar")?.getContext("2d");
  if (!ctx) return;

  if (aspectChart) {
    aspectChart.destroy();
  }

  const sorted = [...aspectStats].sort((a, b) => a.index - b.index);
  const labels = sorted.map((s) => s.aspect);
  const values = sorted.map((s) => s.index);
  const colors = values.map((v) => (v >= 0.25 ? "#10b981" : v >= 0 ? "#64748b" : "#ef4444"));

  aspectChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Indeks Reputasi",
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderWidth: 1,
          borderColor: "#334155",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: -0.3,
          max: 0.8,
          ticks: {
            callback: (val) => `${(val * 100).toFixed(0)}%`,
          },
          grid: { color: "#e2e8f0" },
        },
        y: {
          grid: { display: false },
          ticks: { font: { weight: 600 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => ` Indeks: ${(item.raw * 100).toFixed(2)}%`,
          },
        },
      },
    },
  });
}

/* ─────────────────────────────────────────────────────────────
   4. Live Sentiment Predictor
   ───────────────────────────────────────────────────────────── */
function initPredictor() {
  const input = document.getElementById("predict-text");
  const btn = document.getElementById("btn-predict");
  const resultBox = document.getElementById("predict-result");

  if (!input || !btn) return;

  const runPrediction = async () => {
    const text = input.value.trim();
    if (!text) {
      alert("Silakan masukkan teks opini terlebih dahulu.");
      return;
    }

    btn.disabled = true;
    btn.innerHTML = "<span>⏳ Memproses Inferensi AI...</span>";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal memproses prediksi");

      // Tampilkan Box Hasil
      if (resultBox) {
        resultBox.style.display = "block";
      }
      const badge = document.getElementById("pred-badge");
      const aspectTag = document.getElementById("pred-aspect");
      const scoreTag = document.getElementById("pred-score");

      if (badge) {
        badge.innerText = data.sentiment.toUpperCase();
        badge.className = `result-badge pill-${data.sentiment}`;
      }
      if (aspectTag) aspectTag.innerText = data.aspect_label;
      if (scoreTag) scoreTag.innerText = `${data.net_score > 0 ? "+" : ""}${data.net_score}`;

      // Update Meter Bars
      const pNeg = data.probabilities.negatif;
      const pNet = data.probabilities.netral;
      const pPos = data.probabilities.positif;

      const meterNeg = document.getElementById("meter-neg");
      const pctNeg = document.getElementById("pct-neg");
      const meterNet = document.getElementById("meter-net");
      const pctNet = document.getElementById("pct-net");
      const meterPos = document.getElementById("meter-pos");
      const pctPos = document.getElementById("pct-pos");

      if (meterNeg) meterNeg.style.width = `${pNeg}%`;
      if (pctNeg) pctNeg.innerText = `${pNeg}%`;

      if (meterNet) meterNet.style.width = `${pNet}%`;
      if (pctNet) pctNet.innerText = `${pNet}%`;

      if (meterPos) meterPos.style.width = `${pPos}%`;
      if (pctPos) pctPos.innerText = `${pPos}%`;
    } catch (err) {
      alert(`Terjadi kesalahan: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = "<span>⚡ Jalankan Prediksi Sentimen</span>";
    }
  };

  btn.addEventListener("click", runPrediction);

  // Quick examples
  document.querySelectorAll(".quick-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const sampleText = pill.dataset.text;
      if (sampleText) {
        input.value = sampleText;
        runPrediction();
      }
    });
  });
}

/* ─────────────────────────────────────────────────────────────
   5. Model Comparison Data Loader
   ───────────────────────────────────────────────────────────── */
async function loadModelComparison() {
  try {
    const res = await fetch("/api/reports");
    const data = await res.json();
    if (!res.ok) return;

    // Render Tabel Perbandingan
    const tbodyComp = document.getElementById("tbody-model-comp");
    if (tbodyComp && data.model_comparison) {
      tbodyComp.innerHTML = data.model_comparison
        .map((row) => {
          const isBest = String(row.Model).includes("Weighted Soft Voting");
          return `
          <tr style="${isBest ? "background: #f0fdf4; font-weight: 700;" : ""}">
            <td>${row.Model} ${isBest ? "🏆" : ""}</td>
            <td>${row.Accuracy || "-"}</td>
            <td>${row["Precision-Macro"] || "-"}</td>
            <td>${row["Recall-Macro"] || "-"}</td>
            <td>${row["F1-Macro"] || "-"}</td>
          </tr>
        `;
        })
        .join("");
    }

    // Render Tabel Ablation Study
    const tbodyAbl = document.getElementById("tbody-ablation");
    if (tbodyAbl && data.ablation_study) {
      tbodyAbl.innerHTML = data.ablation_study
        .map((row) => `
          <tr>
            <td>${row["Konfigurasi Model"]}</td>
            <td>${row.Accuracy}</td>
            <td>${row["F1-Macro"]}</td>
            <td><strong>${row["Delta F1 vs Best Single"]}</strong></td>
          </tr>
        `)
        .join("");
    }
  } catch (err) {
    console.error("Gagal memuat laporan evaluasi model:", err);
  }
}

/* ─────────────────────────────────────────────────────────────
   6. Data Explorer Table & Search
   ───────────────────────────────────────────────────────────── */
async function loadCommentsTable() {
  const params = getFilterParams();
  const search = encodeURIComponent(document.getElementById("table-search")?.value || "");

  try {
    const res = await fetch(`/api/data?${params}&search=${search}&limit=60`);
    const result = await res.json();
    const tbody = document.getElementById("tbody-comments");
    const countSpan = document.getElementById("data-count-text");

    if (countSpan) {
      countSpan.innerText = `Menampilkan ${result.displayed} dari ${result.total} opini terfilter`;
    }

    if (tbody) {
      if (!result.data || result.data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #94a3b8; padding: 2rem;">Tidak ada opini yang sesuai kriteria filter.</td></tr>`;
        return;
      }

      tbody.innerHTML = result.data
        .map(
          (row) => `
        <tr>
          <td><span style="font-weight: 700; color: #475569;">${row.platform}</span></td>
          <td style="max-width: 450px;">${escapeHtml(row.text)}</td>
          <td><span class="sentiment-pill pill-${row.sentiment}">${row.sentiment}</span></td>
          <td><span style="font-weight: 600;">${row.aspect}</span></td>
          <td style="color: #64748b; font-size: 0.8rem;">${row.created_at}</td>
        </tr>
      `
        )
        .join("");
    }
  } catch (err) {
    console.error("Gagal memuat data explorer:", err);
  }
}

function initSearch() {
  const searchInput = document.getElementById("table-search");
  let timeout = null;
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        loadCommentsTable();
      }, 350);
    });
  }

  const exportBtn = document.getElementById("btn-export-csv");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      const params = getFilterParams();
      window.location.href = `/api/data?${params}&limit=2000`;
    });
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.innerText = text;
  return div.innerHTML;
}
