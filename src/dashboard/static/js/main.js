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
let aspectBreakdownChart = null;
let modelChart = null;
let timelineChart = null;

document.addEventListener("DOMContentLoaded", () => {
  try { initSidebar(); } catch (e) { console.error("initSidebar error:", e); }
  try { initTabs(); } catch (e) { console.error("initTabs error:", e); }
  try { initFilters(); } catch (e) { console.error("initFilters error:", e); }
  try { initExplorerFilters(); } catch (e) { console.error("initExplorerFilters error:", e); }
  try { initWeeklyPulseControls(); } catch (e) { console.error("initWeeklyPulseControls error:", e); }
  try { initPredictor(); } catch (e) { console.error("initPredictor error:", e); }
  try { initExplorerControls(); } catch (e) { console.error("initExplorerControls error:", e); }
  try { updateEfpActiveBadges(); } catch (e) { console.error("updateEfpActiveBadges error:", e); }

  loadWeeklyPulse().catch((e) => console.error("loadWeeklyPulse error:", e));
  loadDashboardData().catch((e) => console.error("loadDashboardData error:", e));
  loadTimelineChart().catch((e) => console.error("loadTimelineChart error:", e));
  loadModelComparison().catch((e) => console.error("loadModelComparison error:", e));
  loadCommentsTable(1).catch((e) => console.error("loadCommentsTable error:", e));

  // Inisialisasi Fitur Pemantauan Isu Mingguan (Langkah 5)
  try { initWeeklyMonitoring(); } catch (e) { console.error("initWeeklyMonitoring error:", e); }
  try { initFetchConfigModal(); } catch (e) { console.error("initFetchConfigModal error:", e); }
  try { initTopIssuesFilters(); } catch (e) { console.error("initTopIssuesFilters error:", e); }
  try { initMultiComponentPredictor(); } catch (e) { console.error("initMultiComponentPredictor error:", e); }
  loadWeeklyMonitoringData().catch((e) => console.error("loadWeeklyMonitoringData error:", e));
  loadWeeklyHistory().catch((e) => console.error("loadWeeklyHistory error:", e));
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
  const timeSel = document.getElementById("filter-time");

  const onFilterChange = () => {
    // Sinkronisasi otomatis ke panel filter Data Explorer
    const efpP = document.getElementById("efp-platform");
    const efpA = document.getElementById("efp-aspect");
    const efpS = document.getElementById("efp-sentiment");
    const efpT = document.getElementById("efp-time");
    if (efpP && platformSel) efpP.value = platformSel.value;
    if (efpA && aspectSel) efpA.value = aspectSel.value;
    if (efpS && sentimentSel) efpS.value = sentimentSel.value;
    if (efpT && timeSel) efpT.value = timeSel.value;
    updateEfpActiveBadges();

    loadDashboardData();
    loadCommentsTable(1);
  };

  if (platformSel) platformSel.addEventListener("change", onFilterChange);
  if (aspectSel) aspectSel.addEventListener("change", onFilterChange);
  if (sentimentSel) sentimentSel.addEventListener("change", onFilterChange);
  if (timeSel) timeSel.addEventListener("change", onFilterChange);
}

function getFilterParams() {
  const p = document.getElementById("filter-platform")?.value || "all";
  const a = document.getElementById("filter-aspect")?.value || "all";
  const s = document.getElementById("filter-sentiment")?.value || "all";
  const t = document.getElementById("filter-time")?.value || "all";
  return `platform=${encodeURIComponent(p)}&aspect=${encodeURIComponent(a)}&sentiment=${encodeURIComponent(s)}&time_range=${encodeURIComponent(t)}`;
}

/* ─────────────────────────────────────────────────────────────
   2b. Explorer Filter Panel (EFP) — Sinkron dengan Filter Global
   ───────────────────────────────────────────────────────────── */
function getExplorerFilterParams() {
  const p = document.getElementById("efp-platform")?.value || "all";
  const a = document.getElementById("efp-aspect")?.value || "all";
  const s = document.getElementById("efp-sentiment")?.value || "all";
  const t = document.getElementById("efp-time")?.value || "all";
  return `platform=${encodeURIComponent(p)}&aspect=${encodeURIComponent(a)}&sentiment=${encodeURIComponent(s)}&time_range=${encodeURIComponent(t)}`;
}

function updateEfpActiveBadges() {
  const container = document.getElementById("efp-active-badges");
  if (!container) return;
  const p = document.getElementById("efp-platform")?.value || "all";
  const a = document.getElementById("efp-aspect")?.value || "all";
  const s = document.getElementById("efp-sentiment")?.value || "all";
  const t = document.getElementById("efp-time")?.value || "all";

  const labels = [];
  if (t !== "all") {
    labels.push({ key: "efp-time", text: "⏱️ " + (t === "7d" ? "7 Hari Terakhir" : "30 Hari Terakhir") });
  }
  if (p !== "all") {
    const pNames = {
      twitter: "Twitter/X",
      youtube: "YouTube",
      tiktok: "TikTok",
      instagram: "Instagram",
      facebook: "Facebook",
      threads: "Threads"
    };
    const pLabel = pNames[p.toLowerCase()] || (p.charAt(0).toUpperCase() + p.slice(1));
    labels.push({ key: "efp-platform", text: "📺 " + pLabel });
  }
  if (a !== "all") {
    const aText = a.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    labels.push({ key: "efp-aspect", text: "🏷️ " + aText });
  }
  if (s !== "all") {
    const sMap = { positif: "🟢 Positif", netral: "🟠 Netral", negatif: "🔴 Negatif" };
    labels.push({ key: "efp-sentiment", text: sMap[s] || s });
  }

  if (labels.length === 0) {
    container.innerHTML = '<span class="efp-badge-none">Tidak ada filter aktif</span>';
  } else {
    container.innerHTML = labels.map(l =>
      `<span class="efp-badge">${l.text}
        <span class="efp-badge-remove" title="Hapus filter" onclick="resetEfpField('${l.key}')">✕</span>
      </span>`
    ).join("");
  }
}

function resetEfpField(id) {
  const el = document.getElementById(id);
  if (el) { el.value = "all"; }

  // Sinkronkan ke dropdown global
  if (id === "efp-platform") {
    const g = document.getElementById("filter-platform");
    if (g) g.value = "all";
  } else if (id === "efp-aspect") {
    const g = document.getElementById("filter-aspect");
    if (g) g.value = "all";
  } else if (id === "efp-sentiment") {
    const g = document.getElementById("filter-sentiment");
    if (g) g.value = "all";
  } else if (id === "efp-time") {
    const g = document.getElementById("filter-time");
    if (g) g.value = "all";
  }

  updateEfpActiveBadges();
  loadDashboardData();
  loadCommentsTable(1);
}

window.resetEfpField = resetEfpField;

function initExplorerFilters() {
  const efpPlatform = document.getElementById("efp-platform");
  const efpAspect = document.getElementById("efp-aspect");
  const efpSentiment = document.getElementById("efp-sentiment");
  const efpTime = document.getElementById("efp-time");
  const efpReset = document.getElementById("btn-reset-explorer-filters");

  const onEfpChange = () => {
    // Sinkronkan balik ke dropdown global
    const gP = document.getElementById("filter-platform");
    const gA = document.getElementById("filter-aspect");
    const gS = document.getElementById("filter-sentiment");
    const gT = document.getElementById("filter-time");
    if (gP && efpPlatform) gP.value = efpPlatform.value;
    if (gA && efpAspect) gA.value = efpAspect.value;
    if (gS && efpSentiment) gS.value = efpSentiment.value;
    if (gT && efpTime) gT.value = efpTime.value;

    updateEfpActiveBadges();
    loadDashboardData();
    loadCommentsTable(1);
  };

  if (efpPlatform) efpPlatform.addEventListener("change", onEfpChange);
  if (efpAspect) efpAspect.addEventListener("change", onEfpChange);
  if (efpSentiment) efpSentiment.addEventListener("change", onEfpChange);
  if (efpTime) efpTime.addEventListener("change", onEfpChange);

  if (efpReset) {
    efpReset.addEventListener("click", () => {
      if (efpPlatform) efpPlatform.value = "all";
      if (efpAspect) efpAspect.value = "all";
      if (efpSentiment) efpSentiment.value = "all";
      if (efpTime) efpTime.value = "all";

      const gP = document.getElementById("filter-platform");
      const gA = document.getElementById("filter-aspect");
      const gS = document.getElementById("filter-sentiment");
      const gT = document.getElementById("filter-time");
      if (gP) gP.value = "all";
      if (gA) gA.value = "all";
      if (gS) gS.value = "all";
      if (gT) gT.value = "all";

      updateEfpActiveBadges();
      loadDashboardData();
      loadCommentsTable(1);
    });
  }
}

/* ─────────────────────────────────────────────────────────────
   2c. Weekly Sentiment Pulse (7 Hari Terakhir)
   ───────────────────────────────────────────────────────────── */
async function loadWeeklyPulse() {
  const dateRangeEl = document.getElementById("wp-date-range");
  const totalEl = document.getElementById("wp-total");
  const totalSubEl = document.getElementById("wp-total-sub");
  const repIdxEl = document.getElementById("wp-rep-idx");
  const deltaEl = document.getElementById("wp-delta");
  const barPos = document.getElementById("wp-bar-pos");
  const barNet = document.getElementById("wp-bar-net");
  const barNeg = document.getElementById("wp-bar-neg");
  const sentText = document.getElementById("wp-sentiment-text");
  const topAspContainer = document.getElementById("wp-top-aspects");
  const alertBar = document.getElementById("wp-alert-bar");
  const alertText = document.getElementById("wp-alert-text");

  try {
    const res = await fetch(`/api/weekly_pulse?_t=${Date.now()}`, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok || !data.available) {
      if (dateRangeEl) dateRangeEl.innerText = data.message || "Data mingguan belum tersedia.";
      return;
    }

    if (dateRangeEl) dateRangeEl.innerText = `Periode: ${data.date_range}`;
    if (totalEl) totalEl.innerText = Number(data.total).toLocaleString("id-ID");
    if (totalSubEl) {
      const prevDiff = data.total - data.total_prev;
      const diffSign = prevDiff >= 0 ? "+" : "";
      totalSubEl.innerText = `${diffSign}${prevDiff} dibanding minggu lalu (${data.total_prev})`;
    }

    if (repIdxEl) {
      repIdxEl.innerText = data.reputation_index_pct;
      repIdxEl.className = `wp-item-val ${data.reputation_index >= 0 ? "text-green" : "text-red"}`;
    }

    if (deltaEl) {
      const isUp = data.trend === "up";
      const isDown = data.trend === "down";
      const icon = isUp ? "▲" : (isDown ? "▼" : "●");
      const colorClass = isUp ? "text-green" : (isDown ? "text-red" : "text-slate");
      deltaEl.innerHTML = `<span class="${colorClass}">${icon} ${data.delta_pct}</span> vs minggu sebelumnya`;
    }

    if (barPos) barPos.style.width = `${data.positif_pct}%`;
    if (barNet) barNet.style.width = `${data.netral_pct}%`;
    if (barNeg) barNeg.style.width = `${data.negatif_pct}%`;

    if (sentText) {
      sentText.innerText = `Positif: ${data.positif_pct}% | Netral: ${data.netral_pct}% | Negatif: ${data.negatif_pct}%`;
    }

    if (topAspContainer && data.top_aspects) {
      topAspContainer.innerHTML = data.top_aspects
        .map(asp => `<span class="wp-asp-pill" title="Indeks: ${asp.index_pct}">🏷️ ${asp.aspect} (${asp.count})</span>`)
        .join("");
    }

    if (alertBar && alertText) {
      alertText.innerText = data.alert_message;
      if (data.alert) {
        alertBar.className = "wp-alert-bar alert-warn";
      } else {
        alertBar.className = "wp-alert-bar";
      }
    }

    if (data.platform_breakdown && data.platform_breakdown.length > 0) {
      renderPlatformBreakdown(data.platform_breakdown);
    }
  } catch (err) {
    console.error("Gagal memuat weekly pulse:", err);
  }
}

function initWeeklyPulseControls() {
  const btn = document.getElementById("btn-fetch-weekly");
  if (!btn) return;
  btn.addEventListener("click", () => {
    openFetchConfigModal();
  });
}

function renderPlatformBreakdown(platforms) {
  const wrap = document.getElementById("wp-platform-table-wrap");
  const tbody = document.getElementById("wp-platform-tbody");
  if (!wrap || !tbody) return;

  const pNames = {
    twitter: "Twitter/X", instagram: "Instagram", youtube: "YouTube",
    threads: "Threads", tiktok: "TikTok", facebook: "Facebook"
  };

  tbody.innerHTML = platforms
    .sort((a, b) => b.total - a.total)
    .map(p => {
      const name = pNames[p.platform] || p.platform;
      const idxVal = parseFloat(p.reputation_index_pct);
      const idxClass = isNaN(idxVal) ? "" : (idxVal >= 0 ? "text-green" : "text-red");
      return `<tr>
        <td><span class="platform-badge platform-${p.platform}">${name}</span></td>
        <td>${Number(p.total).toLocaleString("id-ID")}</td>
        <td class="text-green">${p.positif_pct}%</td>
        <td class="text-slate">${p.netral_pct}%</td>
        <td class="text-red">${p.negatif_pct}%</td>
        <td class="${idxClass} fw-600">${p.reputation_index_pct}</td>
      </tr>`;
    }).join("");

  wrap.style.display = "block";
}

/* ─────────────────────────────────────────────────────────────
   3. Muat Data Statistik & Update Chart.js
   ───────────────────────────────────────────────────────────── */
async function loadDashboardData() {
  const params = getFilterParams();
  try {
    const res = await fetch(`/api/stats?${params}&_t=${Date.now()}`, { cache: "no-store" });
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

    // Render / Update 7 Aspect Sentiment Breakdown (Stacked Bar, Cards & Table)
    renderAspectBreakdownChart(data.aspect_stats || []);
    renderAspectCards(data.aspect_stats || []);
    renderAspectTable(data.aspect_stats || []);
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

const ASPECT_META = {
  "Akademik": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10M6 10h10"/></svg>',
    label: "Akademik & Kurikulum",
    desc: "Dosen, kurikulum, akreditasi, perkuliahan, dan skripsi"
  },
  "Penerimaan Biaya": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" x2="10"/></svg>',
    label: "Penerimaan & Biaya (UKT)",
    desc: "SNBP, SNBT, mandiri, UKT, beasiswa, dan KIP-Kuliah"
  },
  "Isu Citra": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18M3 10h18M5 10v11M19 10v11M9 10v11M15 10v11M12 2l9 5H3l9-5z"/></svg>',
    label: "Isu & Citra Kampus",
    desc: "Berita, viral, reputasi publik, almamater, dan kebijakan kampus"
  },
  "Fasilitas": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="20" x="4" y="2" rx="2"/><line x1="9" x2="9.01" y1="6" y2="6"/><line x1="15" x2="15.01" y1="6" y2="6"/><line x1="9" x2="9.01" y1="10" y2="10"/><line x1="15" x2="15.01" y1="10" y2="10"/><line x1="9" x2="9.01" y1="14" y2="14"/><line x1="15" x2="15.01" y1="14" y2="14"/></svg>',
    label: "Fasilitas & Sarana",
    desc: "Gedung kuliah, laboratorium, ruang kelas, WiFi, dan perpustakaan"
  },
  "Lulusan Karier": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    label: "Lulusan & Karier",
    desc: "Wisuda, alumni, prospek kerja, dan tracer study"
  },
  "Kemahasiswaan": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    label: "Kemahasiswaan & Prestasi",
    desc: "PKKMB, BEM, UKM, organisasi mahasiswa, lomba, dan KKN"
  },
  "Layanan Administrasi": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    label: "Layanan & SIAKAD",
    desc: "SIAKAD, web server, administrasi birokrasi, surat, dan KRS"
  },
  "Lainnya": {
    svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>',
    label: "Opini Umum / Lainnya",
    desc: "Komentar umum dan sapaan tanpa pilar spesifik"
  }
};
// Alias mapping untuk raw keys dan nama alternatif
ASPECT_META["akademik"] = ASPECT_META["Akademik"];
ASPECT_META["penerimaan_biaya"] = ASPECT_META["Penerimaan Biaya"];
ASPECT_META["Biaya"] = ASPECT_META["Penerimaan Biaya"];
ASPECT_META["isu_citra"] = ASPECT_META["Isu Citra"];
ASPECT_META["Citra Umum"] = ASPECT_META["Isu Citra"];
ASPECT_META["fasilitas"] = ASPECT_META["Fasilitas"];
ASPECT_META["lulusan_karier"] = ASPECT_META["Lulusan Karier"];
ASPECT_META["Karier"] = ASPECT_META["Lulusan Karier"];
ASPECT_META["kemahasiswaan"] = ASPECT_META["Kemahasiswaan"];
ASPECT_META["layanan_administrasi"] = ASPECT_META["Layanan Administrasi"];
ASPECT_META["Layanan"] = ASPECT_META["Layanan Administrasi"];
ASPECT_META["lainnya"] = ASPECT_META["Lainnya"];

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
          backgroundColor: "#0f172a",
          titleFont: { family: "'Plus Jakarta Sans', sans-serif", weight: 700, size: 13 },
          bodyFont: { family: "'Plus Jakarta Sans', sans-serif", size: 12 },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            title: (items) => {
              const item = items[0];
              const stat = sorted[item.dataIndex];
              return `${stat.aspect} (Total ${stat.total.toLocaleString()} Opini)`;
            },
            label: (item) => {
              const stat = sorted[item.dataIndex];
              const posPct = stat.positif_pct ?? ((stat.positif / (stat.total || 1)) * 100).toFixed(1);
              const netPct = stat.netral_pct ?? ((stat.netral / (stat.total || 1)) * 100).toFixed(1);
              const negPct = stat.negatif_pct ?? ((stat.negatif / (stat.total || 1)) * 100).toFixed(1);
              return [
                ` Indeks: ${(stat.index * 100).toFixed(2)}% [${stat.predikat}]`,
                ` Positif: ${stat.positif.toLocaleString()} (${posPct}%)`,
                ` Netral:  ${stat.netral.toLocaleString()} (${netPct}%)`,
                ` Negatif: ${stat.negatif.toLocaleString()} (${negPct}%)`,
              ];
            },
          },
        },
      },
    },
  });
}

function renderAspectBreakdownChart(aspectStats) {
  const ctx = document.getElementById("chart-aspect-breakdown")?.getContext("2d");
  if (!ctx) return;

  if (aspectBreakdownChart) {
    aspectBreakdownChart.destroy();
  }

  // Sort by total opini descending
  const sorted = [...aspectStats].sort((a, b) => b.total - a.total);
  const labels = sorted.map((s) => s.aspect);

  const posData = sorted.map((s) => {
    const tot = s.total || 1;
    return Number(((s.positif / tot) * 100).toFixed(1));
  });
  const netData = sorted.map((s) => {
    const tot = s.total || 1;
    return Number(((s.netral / tot) * 100).toFixed(1));
  });
  const negData = sorted.map((s) => {
    const tot = s.total || 1;
    return Number(((s.negatif / tot) * 100).toFixed(1));
  });

  aspectBreakdownChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Positif (Apresiasi)",
          data: posData,
          backgroundColor: "#10b981",
          borderWidth: 1,
          borderColor: "#059669",
        },
        {
          label: "Netral (Informasi)",
          data: netData,
          backgroundColor: "#64748b",
          borderWidth: 1,
          borderColor: "#475569",
        },
        {
          label: "Negatif (Keluhan)",
          data: negData,
          backgroundColor: "#ef4444",
          borderWidth: 1,
          borderColor: "#dc2626",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          stacked: true,
          max: 100,
          ticks: {
            callback: (val) => `${val}%`,
          },
          grid: { color: "#e2e8f0" },
        },
        y: {
          stacked: true,
          grid: { display: false },
          ticks: { font: { weight: 600 } },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { family: "'Plus Jakarta Sans', sans-serif", weight: 600 } },
        },
        tooltip: {
          backgroundColor: "#0f172a",
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            title: (items) => {
              const stat = sorted[items[0].dataIndex];
              return `${stat.aspect} (Total ${stat.total.toLocaleString()} Opini)`;
            },
            label: (item) => {
              const stat = sorted[item.dataIndex];
              const val = item.raw;
              let rawCount = 0;
              if (item.datasetIndex === 0) rawCount = stat.positif;
              else if (item.datasetIndex === 1) rawCount = stat.netral;
              else if (item.datasetIndex === 2) rawCount = stat.negatif;
              return ` ${item.dataset.label}: ${val}% (${rawCount.toLocaleString()} opini)`;
            },
            afterBody: (items) => {
              const stat = sorted[items[0].dataIndex];
              return `\nIndeks Reputasi: ${(stat.index * 100).toFixed(2)}% [${stat.predikat}]`;
            },
          },
        },
      },
    },
  });
}

function renderAspectCards(aspectStats) {
  const container = document.getElementById("aspect-cards-container");
  if (!container) return;

  if (!aspectStats || aspectStats.length === 0) {
    container.innerHTML = '<div class="empty-state">Tidak ada data aspek untuk filter saat ini.</div>';
    return;
  }

  const sorted = [...aspectStats].sort((a, b) => b.total - a.total);

  let html = "";
  sorted.forEach((stat) => {
    const meta = ASPECT_META[stat.aspect] || ASPECT_META[stat.aspect_raw] || {
      svg: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18M3 10h18M5 10v11M19 10v11M9 10v11M15 10v11M12 2l9 5H3l9-5z"/></svg>',
      label: stat.aspect,
      desc: "Pilar reputasi Undiksha"
    };
    const tot = stat.total || 1;
    const posPct = stat.positif_pct ?? ((stat.positif / tot) * 100).toFixed(1);
    const netPct = stat.netral_pct ?? ((stat.netral / tot) * 100).toFixed(1);
    const negPct = stat.negatif_pct ?? ((stat.negatif / tot) * 100).toFixed(1);

    const isPos = stat.index >= 0.25;
    const isMod = stat.index >= 0 && stat.index < 0.25;
    const badgeClass = isPos ? "badge-pos" : isMod ? "badge-net" : "badge-neg";
    const rawKey = stat.aspect_raw || stat.aspect.toLowerCase().replace(/ /g, "_");

    html += `
      <div class="aspect-detail-card">
        <div class="aspect-card-header">
          <div class="aspect-card-title-group">
            <span class="aspect-card-icon">${meta.svg}</span>
            <div>
              <h4 class="aspect-card-name">${stat.aspect}</h4>
              <span class="aspect-card-sub">${meta.desc}</span>
            </div>
          </div>
          <span class="aspect-status-pill ${badgeClass}">${stat.predikat}</span>
        </div>

        <div class="aspect-sentiment-stats-grid">
          <div class="aspect-stat-cell cell-pos">
            <div class="stat-top">
              <span class="stat-bullet bg-green"></span>
              <span class="stat-label">Positif</span>
            </div>
            <div class="stat-number text-green">${stat.positif.toLocaleString()}</div>
            <div class="stat-pct">${posPct}%</div>
          </div>

          <div class="aspect-stat-cell cell-net">
            <div class="stat-top">
              <span class="stat-bullet bg-slate"></span>
              <span class="stat-label">Netral</span>
            </div>
            <div class="stat-number text-slate">${stat.netral.toLocaleString()}</div>
            <div class="stat-pct">${netPct}%</div>
          </div>

          <div class="aspect-stat-cell cell-neg">
            <div class="stat-top">
              <span class="stat-bullet bg-red"></span>
              <span class="stat-label">Negatif</span>
            </div>
            <div class="stat-number text-red">${stat.negatif.toLocaleString()}</div>
            <div class="stat-pct">${negPct}%</div>
          </div>
        </div>

        <!-- Stacked Visual Bar -->
        <div class="aspect-stacked-meter" title="Positif: ${posPct}% | Netral: ${netPct}% | Negatif: ${negPct}%">
          <div class="seg-fill fill-green" style="width: ${posPct}%;"></div>
          <div class="seg-fill fill-slate" style="width: ${netPct}%;"></div>
          <div class="seg-fill fill-red" style="width: ${negPct}%;"></div>
        </div>

        <div class="aspect-card-footer">
          <div class="aspect-footer-index">
            <span class="foot-label">Indeks:</span>
            <strong class="foot-val ${stat.index >= 0 ? "text-green" : "text-red"}">${stat.index_pct}</strong>
            <span class="foot-total">(${stat.total.toLocaleString()} opini)</span>
          </div>
          <button type="button" class="btn-filter-aspect" onclick="filterAspectExplorer('${rawKey}')">
            <span>Telusuri Opini</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </button>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function renderAspectTable(aspectStats) {
  const tbody = document.getElementById("tbody-aspect-summary");
  if (!tbody) return;

  if (!aspectStats || aspectStats.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align: center;">Tidak ada data aspek.</td></tr>';
    return;
  }

  const sorted = [...aspectStats].sort((a, b) => b.total - a.total);
  let html = "";

  sorted.forEach((stat) => {
    const tot = stat.total || 1;
    const posPct = stat.positif_pct ?? ((stat.positif / tot) * 100).toFixed(1);
    const netPct = stat.netral_pct ?? ((stat.netral / tot) * 100).toFixed(1);
    const negPct = stat.negatif_pct ?? ((stat.negatif / tot) * 100).toFixed(1);

    const isPos = stat.index >= 0.25;
    const isMod = stat.index >= 0 && stat.index < 0.25;
    const badgeClass = isPos ? "badge-pos" : isMod ? "badge-net" : "badge-neg";
    const rawKey = stat.aspect_raw || stat.aspect.toLowerCase().replace(/ /g, "_");

    html += `
      <tr>
        <td><strong>${stat.aspect}</strong></td>
        <td style="text-align: right; font-weight: 700;">${stat.total.toLocaleString()}</td>
        <td style="text-align: right; color: #10b981; font-weight: 600;">
          ${stat.positif.toLocaleString()} <span style="font-size: 0.75rem; color: #64748b;">(${posPct}%)</span>
        </td>
        <td style="text-align: right; color: #64748b; font-weight: 600;">
          ${stat.netral.toLocaleString()} <span style="font-size: 0.75rem; color: #64748b;">(${netPct}%)</span>
        </td>
        <td style="text-align: right; color: #ef4444; font-weight: 600;">
          ${stat.negatif.toLocaleString()} <span style="font-size: 0.75rem; color: #64748b;">(${negPct}%)</span>
        </td>
        <td>
          <div class="aspect-stacked-meter" style="height: 10px; margin: 0;">
            <div class="seg-fill fill-green" style="width: ${posPct}%;"></div>
            <div class="seg-fill fill-slate" style="width: ${netPct}%;"></div>
            <div class="seg-fill fill-red" style="width: ${negPct}%;"></div>
          </div>
        </td>
        <td style="text-align: right; font-weight: 800; color: ${stat.index >= 0 ? "#10b981" : "#ef4444"};">
          ${stat.index_pct}
        </td>
        <td><span class="kpi-status-badge ${badgeClass}" style="font-size: 0.75rem; padding: 0.25rem 0.6rem;">${stat.predikat}</span></td>
        <td style="text-align: center;">
          <button type="button" class="btn-filter-aspect-sm" onclick="filterAspectExplorer('${rawKey}')" title="Buka di Explorer">
            Buka
          </button>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

window.filterAspectExplorer = function(aspectKey) {
  // 1. Sinkronkan filter di Overview tab
  const aspectSel = document.getElementById("filter-aspect");
  if (aspectSel) {
    let found = false;
    for (let opt of aspectSel.options) {
      if (opt.value.toLowerCase() === aspectKey.toLowerCase()) {
        aspectSel.value = opt.value;
        found = true;
        break;
      }
    }
    if (!found) {
      aspectSel.value = aspectKey;
    }
  }

  // 2. Sinkronkan filter mandiri di Data Explorer tab
  const efpAspect = document.getElementById("efp-aspect");
  if (efpAspect) {
    let found = false;
    for (let opt of efpAspect.options) {
      if (opt.value.toLowerCase() === aspectKey.toLowerCase()) {
        efpAspect.value = opt.value;
        found = true;
        break;
      }
    }
    if (!found) {
      efpAspect.value = aspectKey;
    }
    updateEfpActiveBadges();
  }

  // 3. Berpindah ke Tab Data Explorer
  const btnExplorer = document.getElementById("btn-tab-explorer");
  if (btnExplorer) {
    btnExplorer.click();
  }

  // 4. Muat ulang data
  loadDashboardData();
  loadCommentsTable(1);
};

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
    btn.innerHTML = "<span>Memproses analisis...</span>";

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
      btn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Jalankan Prediksi</span>
      `;
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
   5. Model Comparison Data Loader & Timeline Chart Loader
   ───────────────────────────────────────────────────────────── */
async function loadTimelineChart() {
  const canvas = document.getElementById("chart-timeline-sentiment");
  if (!canvas) return;

  try {
    const res = await fetch("/api/timeline");
    const data = await res.json();
    if (!res.ok || !data.timeline || data.timeline.length === 0) return;

    // Ambil data bulanan (maks 30 periode terbaru agar grafik proporsional dan tidak padat)
    const rawList = data.timeline;
    const items = rawList.length > 30 ? rawList.slice(-30) : rawList;

    const labels = items.map((d) => d.periode);
    const posCounts = items.map((d) => d.positif);
    const netCounts = items.map((d) => d.netral);
    const negCounts = items.map((d) => d.negatif);
    const repIndices = items.map((d) => (d.index * 100));

    if (timelineChart) {
      timelineChart.destroy();
    }

    const ctx = canvas.getContext("2d");
    timelineChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            type: "line",
            label: "Indeks Reputasi (%)",
            data: repIndices,
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.12)",
            borderWidth: 2.5,
            pointBackgroundColor: "#1d4ed8",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.3,
            yAxisID: "y1",
            order: 1,
          },
          {
            label: "Positif",
            data: posCounts,
            backgroundColor: "#22c55e",
            borderRadius: 4,
            stack: "volume",
            yAxisID: "y",
            order: 2,
          },
          {
            label: "Netral",
            data: netCounts,
            backgroundColor: "#94a3b8",
            borderRadius: 4,
            stack: "volume",
            yAxisID: "y",
            order: 3,
          },
          {
            label: "Negatif",
            data: negCounts,
            backgroundColor: "#ef4444",
            borderRadius: 4,
            stack: "volume",
            yAxisID: "y",
            order: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        scales: {
          x: {
            stacked: true,
            grid: { display: false },
            ticks: {
              maxRotation: 45,
              minRotation: 0,
              font: { size: 10, family: "'Plus Jakarta Sans', sans-serif" },
            },
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            stacked: true,
            title: {
              display: true,
              text: "Volume Opini",
              font: { weight: 700, family: "'Plus Jakarta Sans', sans-serif" },
            },
            grid: { color: "#f1f5f9" },
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            title: {
              display: true,
              text: "Indeks Reputasi (%)",
              font: { weight: 700, family: "'Plus Jakarta Sans', sans-serif" },
            },
            min: -100,
            max: 100,
            grid: { drawOnChartArea: false },
            ticks: {
              callback: (val) => `${val}%`,
            },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              usePointStyle: true,
              font: { family: "'Plus Jakarta Sans', sans-serif", weight: 600 },
            },
          },
          tooltip: {
            backgroundColor: "#0f172a",
            padding: 12,
            cornerRadius: 8,
            callbacks: {
              afterBody: (items) => {
                const idxItem = items.find((it) => it.datasetIndex === 0);
                if (idxItem) {
                  return `\nIndeks Reputasi: ${idxItem.raw > 0 ? "+" : ""}${idxItem.raw.toFixed(2)}%`;
                }
                return "";
              },
            },
          },
        },
      },
    });
  } catch (err) {
    console.error("Gagal memuat timeline chart:", err);
  }
}

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
          const modelName = String(row.Model || "");
          const isStacking = modelName.includes("Stacking");
          const isVoting = modelName.includes("Soft Voting");
          
          let rowClass = "";
          let badgeHtml = "";
          
          if (isStacking) {
            rowClass = "model-winner-row";
            badgeHtml = '<span class="gold-badge">🏆 Pemenang Utama (F1 93.73%)</span>';
          } else if (isVoting) {
            badgeHtml = '<span class="best-model-tag" style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;">Ensemble Soft Voting</span>';
          }

          return `
          <tr class="${rowClass}">
            <td><strong>${row.Model}</strong> ${badgeHtml}</td>
            <td><strong>${row.Accuracy || "-"}</strong></td>
            <td>${row["Precision-Macro"] || "-"}</td>
            <td>${row["Recall-Macro"] || "-"}</td>
            <td><strong style="${isStacking ? 'color:#15803d; font-size:1.02rem;' : ''}">${row["F1-Macro"] || "-"}</strong></td>
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
   6. Data Explorer Table, Pagination & Scroll Controls
   ───────────────────────────────────────────────────────────── */
let currentExplorerPage = 1;
let currentExplorerPageSize = 50;
let totalExplorerPages = 1;
let currentExplorerRows = {};
let activeEditRow = null;

async function loadCommentsTable(page = currentExplorerPage) {
  currentExplorerPage = page;
  // Gunakan EFP filter jika tersedia, fallback ke global filter
  const efpPlatform = document.getElementById("efp-platform");
  const params = efpPlatform ? getExplorerFilterParams() : getFilterParams();
  const search = encodeURIComponent(document.getElementById("table-search")?.value || "");

  try {
    const res = await fetch(`/api/data?${params}&search=${search}&page=${page}&limit=${currentExplorerPageSize}&_t=${Date.now()}`, { cache: "no-store" });
    const result = await res.json();
    const tbody = document.getElementById("tbody-comments");
    const countSpan = document.getElementById("data-count-text");
    const rangeSpan = document.getElementById("pagination-range-text");

    totalExplorerPages = result.total_pages || 1;

    if (countSpan) {
      countSpan.innerText = `Total ${Number(result.total).toLocaleString("id-ID")} opini terfilter`;
    }

    if (rangeSpan) {
      if (result.total === 0) {
        rangeSpan.innerText = "Menampilkan 0 dari 0 data";
      } else {
        rangeSpan.innerText = `Menampilkan ${Number(result.start).toLocaleString("id-ID")} - ${Number(result.end).toLocaleString("id-ID")} dari ${Number(result.total).toLocaleString("id-ID")} opini`;
      }
    }

    if (tbody) {
      if (!result.data || result.data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #94a3b8; padding: 2.5rem 1rem;">Tidak ada opini yang sesuai kriteria filter.</td></tr>`;
        renderPaginationButtons(result.page, result.total_pages);
        return;
      }

      currentExplorerRows = {};
      result.data.forEach((r) => {
        const key = r.full_id || r.id;
        currentExplorerRows[key] = r;
      });

      tbody.innerHTML = result.data
        .map((row) => {
          const rowKey = row.full_id || row.id;
          const pLower = (row.platform || "").toLowerCase();
          let platformBadgeClass = "badge-tw";
          if (pLower.includes("instagram")) platformBadgeClass = "badge-ig";
          else if (pLower.includes("youtube")) platformBadgeClass = "badge-yt";
          else if (pLower.includes("facebook") || pLower.includes("fb")) platformBadgeClass = "badge-fb";
          else if (pLower.includes("threads")) platformBadgeClass = "badge-threads";
          else if (pLower.includes("tiktok")) platformBadgeClass = "badge-tiktok";

          const editBtn = `<button type="button" class="btn-table-edit" onclick="openEditModalById('${rowKey}')" title="Koreksi sentimen atau aspek">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            <span>Edit</span>
          </button>`;

          const linkBtn =
            row.url && row.url.trim()
              ? `<a href="${escapeHtml(row.url)}" target="_blank" rel="noopener noreferrer" class="btn-table-link" title="Buka tautan postingan asli">
                  <span>Buka</span>
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3"/></svg>
                 </a>`
              : `<span style="color: #cbd5e1; font-size: 0.75rem;">-</span>`;

          const dateDisplay = (!row.created_at || row.created_at === "nan" || row.created_at === "None") ? "-" : row.created_at;

          return `
            <tr>
              <td><span class="platform-badge ${platformBadgeClass}">${row.platform}</span></td>
              <td style="max-width: 480px; word-break: break-word;">${escapeHtml(row.text)}</td>
              <td><span class="sentiment-pill pill-${row.sentiment}">${row.sentiment}</span></td>
              <td><span class="aspect-tag">${row.aspect}</span></td>
              <td style="color: #64748b; font-size: 0.8rem; white-space: nowrap;">${dateDisplay}</td>
              <td style="text-align: center; white-space: nowrap;">
                <div class="table-action-cell">
                  ${editBtn}
                  ${linkBtn}
                </div>
              </td>
            </tr>
          `;
        })
        .join("");

      renderPaginationButtons(result.page, result.total_pages);
    }
  } catch (err) {
    console.error("Gagal memuat data explorer:", err);
  }
}

function renderPaginationButtons(currentPage, totalPages) {
  const container = document.getElementById("pagination-pages-container");
  const prevBtn = document.getElementById("btn-page-prev");
  const nextBtn = document.getElementById("btn-page-next");

  if (prevBtn) {
    prevBtn.disabled = currentPage <= 1;
  }
  if (nextBtn) {
    nextBtn.disabled = currentPage >= totalPages;
  }

  if (!container) return;
  container.innerHTML = "";

  if (totalPages <= 1) return;

  const pages = [];
  const delta = 2;

  let left = Math.max(1, currentPage - delta);
  let right = Math.min(totalPages, currentPage + delta);

  if (left > 1) {
    pages.push(1);
    if (left > 2) pages.push("...");
  }

  for (let i = left; i <= right; i++) {
    pages.push(i);
  }

  if (right < totalPages) {
    if (right < totalPages - 1) pages.push("...");
    pages.push(totalPages);
  }

  pages.forEach((p) => {
    if (p === "...") {
      const span = document.createElement("span");
      span.className = "page-num-dots";
      span.innerText = "...";
      container.appendChild(span);
    } else {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `page-num-btn ${p === currentPage ? "active" : ""}`;
      btn.innerText = p;
      btn.title = `Geser ke halaman ${p}`;
      btn.addEventListener("click", () => {
        loadCommentsTable(p);
      });
      container.appendChild(btn);
    }
  });
}

function initExplorerControls() {
  // 1. Tombol Geser Tabel Horizontal (Smooth Scroll Left & Right)
  const tableContainer = document.getElementById("explorer-table-responsive");
  const scrollLeftBtn = document.getElementById("btn-scroll-table-left");
  const scrollRightBtn = document.getElementById("btn-scroll-table-right");

  if (tableContainer) {
    if (scrollLeftBtn) {
      scrollLeftBtn.addEventListener("click", () => {
        tableContainer.scrollBy({ left: -320, behavior: "smooth" });
      });
    }
    if (scrollRightBtn) {
      scrollRightBtn.addEventListener("click", () => {
        tableContainer.scrollBy({ left: 320, behavior: "smooth" });
      });
    }
  }

  // 2. Tombol Geser Halaman Data (Prev & Next)
  const prevBtn = document.getElementById("btn-page-prev");
  const nextBtn = document.getElementById("btn-page-next");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (currentExplorerPage > 1) {
        loadCommentsTable(currentExplorerPage - 1);
      }
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      if (currentExplorerPage < totalExplorerPages) {
        loadCommentsTable(currentExplorerPage + 1);
      }
    });
  }

  // 3. Dropdown Jumlah Baris per Halaman
  const sizeSelect = document.getElementById("select-page-size");
  if (sizeSelect) {
    sizeSelect.addEventListener("change", (e) => {
      currentExplorerPageSize = parseInt(e.target.value, 10) || 50;
      loadCommentsTable(1);
    });
  }

  // 4. Pencarian Teks Opini
  const searchInput = document.getElementById("table-search");
  let timeout = null;
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        loadCommentsTable(1);
      }, 300);
    });
  }

  // 5. Ekspor Data
  const exportBtn = document.getElementById("btn-export-csv");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      const efpPlatform = document.getElementById("efp-platform");
      const params = efpPlatform ? getExplorerFilterParams() : getFilterParams();
      window.location.href = `/api/data?${params}&limit=2000`;
    });
  }

  // 6. Tombol Aksi Modal Koreksi Label (Human-in-the-Loop)
  const btnCloseModal = document.getElementById("btn-close-edit-modal");
  const btnCancelModal = document.getElementById("btn-cancel-edit");
  const btnSaveModal = document.getElementById("btn-save-edit");
  const btnSpamModal = document.getElementById("btn-mark-spam");
  const modalBackdrop = document.getElementById("edit-modal-backdrop");

  if (btnCloseModal) btnCloseModal.addEventListener("click", closeEditModal);
  if (btnCancelModal) btnCancelModal.addEventListener("click", closeEditModal);
  if (btnSaveModal) btnSaveModal.addEventListener("click", saveEditRow);
  if (btnSpamModal) btnSpamModal.addEventListener("click", markRowAsSpam);
  if (modalBackdrop) {
    modalBackdrop.addEventListener("click", (e) => {
      if (e.target === modalBackdrop) closeEditModal();
    });
  }
}

/* ─────────────────────────────────────────────────────────────
   7. Logika Modal Koreksi Label (Human-in-the-Loop)
   ───────────────────────────────────────────────────────────── */
window.openEditModalById = function(rowKey) {
  const row = currentExplorerRows[rowKey];
  if (!row) return;
  activeEditRow = row;

  const modal = document.getElementById("edit-modal-backdrop");
  const textEl = document.getElementById("edit-modal-text");
  const sentEl = document.getElementById("edit-select-sentiment");
  const aspEl = document.getElementById("edit-select-aspect");

  if (textEl) textEl.innerText = row.text || "-";
  if (sentEl) {
    const sLower = (row.sentiment || "").toLowerCase();
    for (let opt of sentEl.options) {
      if (opt.value.toLowerCase() === sLower) {
        sentEl.value = opt.value;
        break;
      }
    }
  }
  if (aspEl) {
    const aLower = (row.aspect_raw || row.aspect || "").toLowerCase().replace(/ /g, "_");
    for (let opt of aspEl.options) {
      if (opt.value.toLowerCase() === aLower) {
        aspEl.value = opt.value;
        break;
      }
    }
  }

  if (modal) modal.style.display = "flex";
};

function closeEditModal() {
  const modal = document.getElementById("edit-modal-backdrop");
  if (modal) modal.style.display = "none";
  activeEditRow = null;
}

async function saveEditRow() {
  if (!activeEditRow) return;
  const sentEl = document.getElementById("edit-select-sentiment");
  const aspEl = document.getElementById("edit-select-aspect");
  const saveBtn = document.getElementById("btn-save-edit");

  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.innerText = "Menyimpan...";
  }

  try {
    const res = await fetch("/api/update_row", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_id: activeEditRow.full_id || activeEditRow.id,
        text: activeEditRow.text,
        sentiment: sentEl?.value,
        aspect: aspEl?.value,
        is_spam: false
      })
    });
    const result = await res.json();
    if (result.success) {
      closeEditModal();
      await loadDashboardData();
      await loadCommentsTable(currentExplorerPage);
      await loadWeeklyMonitoringData(currentWiPage);
    } else {
      alert("Gagal menyimpan koreksi: " + (result.error || "Kesalahan server"));
    }
  } catch (err) {
    alert("Koneksi gagal saat menyimpan koreksi: " + err.message);
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
        <span>Simpan Koreksi</span>
      `;
    }
  }
}

async function markRowAsSpam() {
  if (!activeEditRow) return;
  if (!confirm("Apakah Anda yakin ingin menandai postingan ini sebagai Iklan/Spam? Postingan ini akan dipindahkan ke arsip spam dan dikeluarkan dari dataset utama.")) {
    return;
  }

  const markBtn = document.getElementById("btn-mark-spam");
  if (markBtn) {
    markBtn.disabled = true;
    markBtn.innerText = "Memproses...";
  }

  try {
    const res = await fetch("/api/update_row", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_id: activeEditRow.full_id || activeEditRow.id,
        text: activeEditRow.text,
        is_spam: true
      })
    });
    const result = await res.json();
    if (result.success) {
      closeEditModal();
      await loadDashboardData();
      await loadCommentsTable(currentExplorerPage);
      await loadWeeklyMonitoringData(currentWiPage);
    } else {
      alert("Gagal menandai spam: " + (result.error || "Kesalahan server"));
    }
  } catch (err) {
    alert("Koneksi gagal: " + err.message);
  } finally {
    if (markBtn) {
      markBtn.disabled = false;
      markBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        <span>Tandai Iklan/Spam</span>
      `;
    }
  }
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.innerText = text;
  return div.innerHTML;
}

/* ─────────────────────────────────────────────────────────────
   FITUR PEMANTAUAN ISU MINGGUAN (LANGKAH 5)
   ───────────────────────────────────────────────────────────── */

let wiDailyChart = null;
let wiPlatformChart = null;
let wiMultiWeekChart = null;
let currentWiPage = 1;
let currentWiPageSize = 15;
let totalWiPages = 1;
let currentWiRows = {};
let wiDebounceTimer = null;

function updateWiActiveBadges() {
  const container = document.getElementById("wi-active-badges");
  if (!container) return;
  const p = document.getElementById("wi-filter-platform")?.value || "all";
  const a = document.getElementById("wi-filter-aspect")?.value || "all";
  const s = document.getElementById("wi-filter-sentiment")?.value || "all";
  const o = document.getElementById("wi-filter-official")?.value || "all";

  const labels = [];
  if (p !== "all") {
    const pNames = {
      twitter: "Twitter/X",
      youtube: "YouTube",
      tiktok: "TikTok",
      instagram: "Instagram",
      facebook: "Facebook",
      threads: "Threads"
    };
    const pLabel = pNames[p.toLowerCase()] || (p.charAt(0).toUpperCase() + p.slice(1));
    labels.push({ key: "wi-filter-platform", text: "📺 " + pLabel });
  }
  if (a !== "all") {
    const aText = a.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    labels.push({ key: "wi-filter-aspect", text: "🏷️ " + aText });
  }
  if (s !== "all") {
    const sMap = { positif: "🟢 Positif", netral: "🟠 Netral", negatif: "🔴 Negatif" };
    labels.push({ key: "wi-filter-sentiment", text: sMap[s] || s });
  }
  if (o !== "all") {
    const oMap = { public: "👤 Akun Publik", official: "🏛️ Akun Resmi" };
    labels.push({ key: "wi-filter-official", text: oMap[o] || o });
  }

  if (labels.length === 0) {
    container.innerHTML = '<span class="efp-badge-none">Tidak ada filter aktif</span>';
  } else {
    container.innerHTML = labels.map(l =>
      `<span class="efp-badge">${l.text}
        <span class="efp-badge-remove" title="Hapus filter" onclick="resetWiField('${l.key}')">✕</span>
      </span>`
    ).join("");
  }
}

function resetWiField(id) {
  const el = document.getElementById(id);
  if (el) el.value = "all";
  updateWiActiveBadges();
  currentWiPage = 1;
  loadWeeklyMonitoringData(1);
}
window.resetWiField = resetWiField;

window.openWiEditModalById = function(rowKey) {
  const row = currentWiRows[rowKey];
  if (!row) return;
  activeEditRow = {
    full_id: row.full_id || row.id_anonim,
    id: row.full_id || row.id_anonim,
    text: row.teks_asli || "",
    sentiment: (row.sentimen || "").toLowerCase(),
    aspect_raw: row.aspek_raw || (row.aspek || "").toLowerCase().replace(/ /g, "_"),
    aspect: row.aspek || ""
  };

  const modal = document.getElementById("edit-modal-backdrop");
  const textEl = document.getElementById("edit-modal-text");
  const sentEl = document.getElementById("edit-select-sentiment");
  const aspEl = document.getElementById("edit-select-aspect");

  if (textEl) textEl.innerText = activeEditRow.text || "-";
  if (sentEl) {
    for (let opt of sentEl.options) {
      if (opt.value.toLowerCase() === activeEditRow.sentiment) {
        sentEl.value = opt.value;
        break;
      }
    }
  }
  if (aspEl) {
    for (let opt of aspEl.options) {
      if (opt.value.toLowerCase() === activeEditRow.aspect_raw) {
        aspEl.value = opt.value;
        break;
      }
    }
  }

  if (modal) modal.style.display = "flex";
};

function initWeeklyMonitoring() {
  const weekSel = document.getElementById("wi-select-week");
  const platSel = document.getElementById("wi-filter-platform");
  const aspSel = document.getElementById("wi-filter-aspect");
  const sentSel = document.getElementById("wi-filter-sentiment");
  const offSel = document.getElementById("wi-filter-official");
  const searchInput = document.getElementById("wi-search-post");
  const btnReset = document.getElementById("btn-reset-wi-filters");
  const btnPrev = document.getElementById("wi-btn-prev-page");
  const btnNext = document.getElementById("wi-btn-next-page");
  const sizeSelect = document.getElementById("wi-select-page-size");
  const tableContainer = document.getElementById("wi-table-responsive");
  const scrollLeftBtn = document.getElementById("btn-wi-scroll-table-left");
  const scrollRightBtn = document.getElementById("btn-wi-scroll-table-right");
  const btnRun = document.getElementById("wi-btn-run-pipeline");

  const onFilterChange = () => {
    updateWiActiveBadges();
    currentWiPage = 1;
    loadWeeklyMonitoringData(1);
  };

  if (weekSel) weekSel.addEventListener("change", onFilterChange);
  if (platSel) platSel.addEventListener("change", onFilterChange);
  if (aspSel) aspSel.addEventListener("change", onFilterChange);
  if (sentSel) sentSel.addEventListener("change", onFilterChange);
  if (offSel) offSel.addEventListener("change", onFilterChange);

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      if (platSel) platSel.value = "all";
      if (aspSel) aspSel.value = "all";
      if (sentSel) sentSel.value = "all";
      if (offSel) offSel.value = "all";
      if (searchInput) searchInput.value = "";
      updateWiActiveBadges();
      currentWiPage = 1;
      loadWeeklyMonitoringData(1);
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(wiDebounceTimer);
      wiDebounceTimer = setTimeout(() => {
        currentWiPage = 1;
        loadWeeklyMonitoringData(1);
      }, 350);
    });
  }

  if (tableContainer) {
    if (scrollLeftBtn) {
      scrollLeftBtn.addEventListener("click", () => {
        tableContainer.scrollBy({ left: -320, behavior: "smooth" });
      });
    }
    if (scrollRightBtn) {
      scrollRightBtn.addEventListener("click", () => {
        tableContainer.scrollBy({ left: 320, behavior: "smooth" });
      });
    }
  }

  if (sizeSelect) {
    sizeSelect.addEventListener("change", (e) => {
      currentWiPageSize = parseInt(e.target.value, 10) || 15;
      currentWiPage = 1;
      loadWeeklyMonitoringData(1);
    });
  }

  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (currentWiPage > 1) {
        currentWiPage--;
        loadWeeklyMonitoringData(currentWiPage);
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (currentWiPage < totalWiPages) {
        currentWiPage++;
        loadWeeklyMonitoringData(currentWiPage);
      }
    });
  }

  if (btnRun) {
    btnRun.addEventListener("click", () => {
      openFetchConfigModal();
    });
  }

  updateWiActiveBadges();
}

/* ─────────────────────────────────────────────────────────────
   MODAL KONFIGURASI PENARIKAN DATA & KATA KUNCI
   ───────────────────────────────────────────────────────────── */

function openFetchConfigModal() {
  const modal = document.getElementById("modal-fetch-config");
  if (modal) {
    modal.classList.add("active");
    modal.style.setProperty("display", "flex", "important");
    updateFetchSummary();
  }
}

function closeFetchConfigModal() {
  const modal = document.getElementById("modal-fetch-config");
  if (modal) {
    modal.classList.remove("active");
    modal.style.setProperty("display", "none", "important");
  }
}

// Make accessible globally
window.openFetchConfigModal = openFetchConfigModal;
window.closeFetchConfigModal = closeFetchConfigModal;

function updateFetchSummary() {
  const days = document.getElementById("cfg-fetch-days")?.value || 7;
  const posts = document.getElementById("cfg-fetch-posts")?.value || 10;
  const comments = document.getElementById("cfg-fetch-comments")?.value || 50;
  const keywords = document.getElementById("cfg-fetch-keywords")?.value.trim() || "undiksha";
  const prioritizeNeg = document.getElementById("cfg-fetch-prioritize-neg")?.checked;
  const summaryEl = document.getElementById("cfg-summary-text");

  if (summaryEl) {
    const negText = prioritizeNeg 
      ? `<strong style="color: #ea580c;">🔥 Prioritas Isu Negatif: AKTIF</strong> (bobot keluhan dan deteksi risiko kampus dimaksimalkan).` 
      : `<span>Fokus Sentimen: Seimbang & Proporsional.</span>`;

    summaryEl.innerHTML = `
      <div>
        <strong>Ringkasan Permintaan:</strong> Mengambil data <strong>${days} hari terakhir</strong> dengan kata kunci <code>${escapeHtml(keywords)}</code>. Batas maks <strong>${posts} konten</strong> per platform dan <strong>${comments} komentar</strong> per konten.
      </div>
      <div style="margin-top: 0.25rem;">
        ${negText} Estimasi durasi: ~${Math.max(12, Math.round(days * 1.2 + posts * 1.5))} detik.
      </div>
    `;
  }
}

function initFetchConfigModal() {
  const modal = document.getElementById("modal-fetch-config");
  const btnClose = document.getElementById("btn-close-fetch-modal");
  const btnCancel = document.getElementById("btn-cancel-fetch-modal");
  const btnSubmit = document.getElementById("btn-submit-fetch-modal");

  if (!modal) return;
  modal.classList.remove("active");
  modal.style.setProperty("display", "none", "important");

  // Close handlers
  if (btnClose) btnClose.addEventListener("click", closeFetchConfigModal);
  if (btnCancel) btnCancel.addEventListener("click", closeFetchConfigModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeFetchConfigModal();
  });

  // Presets
  const presetCards = document.querySelectorAll(".preset-card");
  presetCards.forEach((card) => {
    card.addEventListener("click", () => {
      presetCards.forEach((c) => c.classList.remove("active"));
      card.classList.add("active");

      const days = card.dataset.days;
      const posts = card.dataset.posts || card.dataset.videos;
      const comments = card.dataset.comments;
      const keywords = card.dataset.keywords;
      const isNeg = card.dataset.preset === "negative";

      if (days) {
        document.getElementById("cfg-fetch-days").value = days;
        document.getElementById("cfg-fetch-days-slider").value = days;
        document.getElementById("cfg-days-display").textContent = days;
      }
      if (posts) {
        document.getElementById("cfg-fetch-posts").value = posts;
        document.getElementById("cfg-fetch-posts-slider").value = posts;
        document.getElementById("cfg-posts-display").textContent = posts;
      }
      if (comments) {
        document.getElementById("cfg-fetch-comments").value = comments;
        document.getElementById("cfg-fetch-comments-slider").value = comments;
        document.getElementById("cfg-comments-display").textContent = comments;
      }
      if (keywords) {
        document.getElementById("cfg-fetch-keywords").value = keywords;
      }

      const chkNeg = document.getElementById("cfg-fetch-prioritize-neg");
      if (chkNeg) chkNeg.checked = isNeg;

      updateChipActiveState();
      updateFetchSummary();
    });
  });

  // Days slider & number input sync
  const sDays = document.getElementById("cfg-fetch-days-slider");
  const nDays = document.getElementById("cfg-fetch-days");
  const dDays = document.getElementById("cfg-days-display");
  if (sDays && nDays && dDays) {
    sDays.addEventListener("input", (e) => {
      nDays.value = e.target.value;
      dDays.textContent = e.target.value;
      updateChipActiveState();
      updateFetchSummary();
    });
    nDays.addEventListener("input", (e) => {
      sDays.value = e.target.value;
      dDays.textContent = e.target.value;
      updateChipActiveState();
      updateFetchSummary();
    });
  }

  // Posts slider & number input sync
  const sPosts = document.getElementById("cfg-fetch-posts-slider");
  const nPosts = document.getElementById("cfg-fetch-posts");
  const dPosts = document.getElementById("cfg-posts-display");
  if (sPosts && nPosts && dPosts) {
    sPosts.addEventListener("input", (e) => {
      nPosts.value = e.target.value;
      dPosts.textContent = e.target.value;
      updateChipActiveState();
      updateFetchSummary();
    });
    nPosts.addEventListener("input", (e) => {
      sPosts.value = e.target.value;
      dPosts.textContent = e.target.value;
      updateChipActiveState();
      updateFetchSummary();
    });
  }

  // Comments slider & number input sync
  const sComm = document.getElementById("cfg-fetch-comments-slider");
  const nComm = document.getElementById("cfg-fetch-comments");
  const dComm = document.getElementById("cfg-comments-display");
  if (sComm && nComm && dComm) {
    sComm.addEventListener("input", (e) => {
      nComm.value = e.target.value;
      dComm.textContent = e.target.value;
      updateChipActiveState();
      updateFetchSummary();
    });
    nComm.addEventListener("input", (e) => {
      sComm.value = e.target.value;
      dComm.textContent = e.target.value;
      updateChipActiveState();
      updateFetchSummary();
    });
  }

  // Quick chips
  document.querySelectorAll(".cfg-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const v = chip.dataset.val;
      if (sDays && nDays && dDays) {
        sDays.value = v;
        nDays.value = v;
        dDays.textContent = v;
        updateChipActiveState();
        updateFetchSummary();
      }
    });
  });

  document.querySelectorAll(".cfg-chip-post").forEach((chip) => {
    chip.addEventListener("click", () => {
      const v = chip.dataset.val;
      if (sPosts && nPosts && dPosts) {
        sPosts.value = v;
        nPosts.value = v;
        dPosts.textContent = v;
        updateChipActiveState();
        updateFetchSummary();
      }
    });
  });

  document.querySelectorAll(".cfg-chip-comm").forEach((chip) => {
    chip.addEventListener("click", () => {
      const v = chip.dataset.val;
      if (sComm && nComm && dComm) {
        sComm.value = v;
        nComm.value = v;
        dComm.textContent = v;
        updateChipActiveState();
        updateFetchSummary();
      }
    });
  });

  // Keywords quick tags
  const kwInput = document.getElementById("cfg-fetch-keywords");
  if (kwInput) {
    kwInput.addEventListener("input", updateFetchSummary);
    document.querySelectorAll(".kw-tag-btn").forEach((tagBtn) => {
      tagBtn.addEventListener("click", () => {
        const tag = tagBtn.dataset.tag;
        const current = kwInput.value.trim();
        const parts = current.split(",").map(p => p.trim()).filter(Boolean);
        if (!parts.includes(tag)) {
          parts.push(tag);
          kwInput.value = parts.join(", ");
        }
        updateFetchSummary();
      });
    });
  }

  const chkNeg = document.getElementById("cfg-fetch-prioritize-neg");
  if (chkNeg) chkNeg.addEventListener("change", updateFetchSummary);

  function updateChipActiveState() {
    const curDays = nDays ? nDays.value : "";
    document.querySelectorAll(".cfg-chip").forEach((c) => {
      c.classList.toggle("active", c.dataset.val === curDays);
    });
    const curPosts = nPosts ? nPosts.value : "";
    document.querySelectorAll(".cfg-chip-post").forEach((c) => {
      c.classList.toggle("active", c.dataset.val === curPosts);
    });
    const curComm = nComm ? nComm.value : "";
    document.querySelectorAll(".cfg-chip-comm").forEach((c) => {
      c.classList.toggle("active", c.dataset.val === curComm);
    });
  }

  // Submit button
  if (btnSubmit) {
    btnSubmit.addEventListener("click", () => {
      const config = {
        days: parseInt(document.getElementById("cfg-fetch-days")?.value || 7, 10),
        maxVideos: parseInt(document.getElementById("cfg-fetch-posts")?.value || 10, 10),
        maxComments: parseInt(document.getElementById("cfg-fetch-comments")?.value || 50, 10),
        maxPosts: parseInt(document.getElementById("cfg-fetch-posts")?.value || 10, 10),
        keywords: document.getElementById("cfg-fetch-keywords")?.value.trim() || "undiksha",
        prioritizeNeg: !!document.getElementById("cfg-fetch-prioritize-neg")?.checked,
      };
      executeDataFetch(config);
    });
  }

  updateFetchSummary();
}

async function executeDataFetch(config) {
  closeFetchConfigModal();

  const btnWi = document.getElementById("wi-btn-run-pipeline");
  const btnPulse = document.getElementById("btn-fetch-weekly");
  const btnPulseText = document.getElementById("btn-fetch-text");

  if (btnWi) {
    btnWi.disabled = true;
    btnWi.innerHTML = `<span class="pulse-dot"></span> Sedang Menarik Data (${config.days} Hari)...`;
  }
  if (btnPulse) {
    btnPulse.classList.add("loading");
    btnPulse.disabled = true;
    if (btnPulseText) btnPulseText.innerText = `Menarik Data (${config.days} Hari)...`;
  }

  const progressPanels = [
    document.getElementById("wi-progress-panel"),
    document.getElementById("wp-progress-panel")
  ].filter(Boolean);

  const progressLists = [
    document.getElementById("wi-progress-list"),
    document.getElementById("wp-progress-list")
  ].filter(Boolean);

  const progressTimes = [
    document.getElementById("wi-progress-time"),
    document.getElementById("wp-progress-time")
  ].filter(Boolean);

  const platformWrap = document.getElementById("wp-platform-table-wrap");

  progressPanels.forEach(p => p.style.display = "block");
  progressLists.forEach(l => l.innerHTML = "");
  if (platformWrap) platformWrap.style.display = "none";
  const summaryBoxReset = document.getElementById("wi-fetch-summary-box");
  if (summaryBoxReset) summaryBoxReset.style.display = "none";
  const doneBannerReset = document.getElementById("wi-progress-done-banner");
  if (doneBannerReset) doneBannerReset.style.display = "none";

  const tStart = Date.now();

  function addProgressStep(step) {
    if (progressLists.length === 0) return;
    if (step.step === null || step.step === undefined) return;
    const icon = step.status === "done" ? "✓" : (step.status === "error" ? "✗" : (step.status === "warn" ? "⚠️" : "⟳"));
    const cls = step.status === "done" ? "step-done" : (step.status === "error" ? "step-error" : "step-running");
    
    progressLists.forEach(pl => {
      const stepId = `step-item-${step.step}`;
      const existing = pl.querySelector(`#${stepId}`);
      const li = document.createElement("li");
      li.className = `wp-step ${cls}`;
      li.id = stepId;
      li.innerHTML = `<span class="wp-step-icon">${icon}</span><span class="wp-step-label">${step.label}</span>`;
      if (existing) existing.replaceWith(li);
      else pl.appendChild(li);
    });
  }

  addProgressStep({
    step: 1,
    label: `Memulai pengambilan data (${config.days} hari terakhir, kata kunci: ${config.keywords}, maks ${config.maxVideos} video & ${config.maxPosts} post)...`,
    status: "running",
  });

  let fetchSummaryData = null;

  try {
    let sseSuccess = false;
    const sseUrl = `/api/weekly_progress?days=${config.days}&max_videos=${config.maxVideos}&max_comments=${config.maxComments}&max_posts=${config.maxPosts}&keywords=${encodeURIComponent(config.keywords)}`;

    await new Promise((resolve) => {
      let evtSrc = null;
      try {
        evtSrc = new EventSource(sseUrl);
        evtSrc.onmessage = async (e) => {
          if (e.data === "__DONE__") {
            if (evtSrc) evtSrc.close();
            sseSuccess = true;
            resolve();
            return;
          }
          try {
            const step = JSON.parse(e.data);
            if (step.status === "summary" && step.summary) {
              fetchSummaryData = step.summary;
            } else {
              addProgressStep(step);
            }
          } catch (_) {}
        };
        evtSrc.onerror = async () => {
          if (evtSrc) evtSrc.close();
          resolve();
        };
      } catch (_) {
        resolve();
      }

      setTimeout(() => {
        if (evtSrc) evtSrc.close();
        resolve();
      }, 600000); // 10 menit max, sinkron dengan backend timeout
    });

    // Fallback POST jika SSE gagal atau tidak selesai
    if (!sseSuccess) {
      try {
        const postRes = await fetch("/api/fetch_weekly", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            days: config.days,
            max_videos: config.maxVideos,
            max_comments: config.maxComments,
            max_posts: config.maxPosts,
            keywords: config.keywords,
          }),
        });
        const postJson = await postRes.json();
        if (postJson.success && postJson.steps) {
          postJson.steps.forEach((s) => addProgressStep(s));
        }
      } catch (_) {}
    }

    // Pastikan seluruh indikator running yang telah selesai berubah menjadi centang hijau
    progressLists.forEach(pl => {
      pl.querySelectorAll(".step-running").forEach(el => {
        el.classList.remove("step-running");
        el.classList.add("step-done");
        const icon = el.querySelector(".wp-step-icon");
        if (icon) icon.textContent = "✓";
      });
    });

    const elapsed = ((Date.now() - tStart) / 1000).toFixed(2);
    progressTimes.forEach(t => { t.innerText = `${elapsed} dtk`; });

    // Tampilkan rincian per platform & statistik pemrosesan jika tersedia
    const summaryBox = document.getElementById("wi-fetch-summary-box");
    const platGrid = document.getElementById("wi-fetch-platform-grid");
    const metaStrip = document.getElementById("wi-fetch-meta-strip");
    if (summaryBox && platGrid && metaStrip && fetchSummaryData) {
      const plats = fetchSummaryData.platforms || [];
      platGrid.innerHTML = plats.map(p => `
        <div class="fetch-plat-card ${p.total === 0 ? "is-zero" : ""}">
          <div class="fetch-plat-head">
            <span>${escapeHtml(p.platform)}</span>
            <span class="fetch-plat-count">${Number(p.total || 0).toLocaleString("id-ID")}</span>
          </div>
          <div class="fetch-plat-note">${escapeHtml(p.note || "-")}</div>
        </div>
      `).join("");

      const totalSpam = (fetchSummaryData.spam_cleaning || 0) + (fetchSummaryData.spam_llm || 0);
      metaStrip.innerHTML = `
        <span class="fetch-meta-chip">📥 Total Ditarik: ${Number(fetchSummaryData.total_fetched || 0).toLocaleString("id-ID")}</span>
        <span class="fetch-meta-chip">🧹 Spam/Noise Dibuang: ${totalSpam} (${fetchSummaryData.spam_cleaning || 0} cleaning + ${fetchSummaryData.spam_llm || 0} LLM)</span>
        <span class="fetch-meta-chip">✨ Data Baru Masuk: +${Number(fetchSummaryData.added || 0).toLocaleString("id-ID")} (${fetchSummaryData.duplicates || 0} duplikat)</span>
        <span class="fetch-meta-chip">🏷️ Pelabelan: ${escapeHtml(fetchSummaryData.labeling_method || "-")}</span>
        <span class="fetch-meta-chip">💬 Sentimen: ${fetchSummaryData.positif || 0} Pos | ${fetchSummaryData.netral || 0} Net | ${fetchSummaryData.negatif || 0} Neg</span>
        <span class="fetch-meta-chip">⏱️ Waktu: ${fetchSummaryData.elapsed_sec || elapsed} dtk | Total Dataset: ${Number(fetchSummaryData.total_dataset || 0).toLocaleString("id-ID")}</span>
      `;
      summaryBox.style.display = "block";
    }

    // Tampilkan kartu sukses sinkronisasi di dalam panel
    const doneBanner = document.getElementById("wi-progress-done-banner");
    const doneDesc = document.getElementById("wi-success-desc");
    if (doneBanner) {
      if (doneDesc) {
        doneDesc.innerText = `Data baru (${config.days} hari terakhir, kata kunci: ${config.keywords}) telah berhasil ditarik dan dianalisis oleh Model AI Ensemble. Metrik dashboard telah disegarkan.`;
      }
      doneBanner.style.display = "flex";
    }

    // Refresh semua data dashboard secara menyeluruh
    await loadWeeklyMonitoringData(1);
    await loadWeeklyHistory();
    await loadWeeklyPulse();
    await loadDashboardData();
    await loadCommentsTable(1);

    if (btnPulseText) btnPulseText.innerText = "Tersinkronisasi ✓";

    // Auto-scroll halus ke ringkasan KPI agar pengguna langsung melihat data baru
    const kpiSection = document.querySelector(".kpi-cards-grid");
    if (kpiSection) {
      setTimeout(() => {
        kpiSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 500);
    }
  } catch (err) {
    console.error("Gagal menjalankan pipeline:", err);
    alert("Koneksi gagal: " + err.message);
  } finally {
    if (btnWi) {
      btnWi.disabled = false;
      btnWi.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
        <span>⚙️ Tarik Data & Analisis</span>
      `;
    }
    if (btnPulse) {
      btnPulse.disabled = false;
      btnPulse.classList.remove("loading");
      setTimeout(() => {
        if (btnPulseText && btnPulseText.innerText === "Tersinkronisasi ✓") {
          btnPulseText.innerText = "Tarik Data 7 Hari Terakhir";
        }
      }, 4000);
    }
  }
}

function dismissProgressPanel() {
  const p1 = document.getElementById("wi-progress-panel");
  const p2 = document.getElementById("wp-progress-panel");
  if (p1) p1.style.display = "none";
  if (p2) p2.style.display = "none";
}
window.dismissProgressPanel = dismissProgressPanel;

let rawWeeklyTopIssues = [];
let isFilterNegativeOnly = true;

function renderFilteredTopIssues() {
  let issues = [...rawWeeklyTopIssues];
  if (isFilterNegativeOnly) {
    // Hanya tampilkan isu yang benar-benar memiliki sentimen negatif (> 0%)
    issues = issues.filter(i => Number(i.persen_negatif || 0) > 0);
    issues.sort((a, b) => (b.persen_negatif - a.persen_negatif) || (b.skor_isu - a.skor_isu) || (b.sebutan - a.sebutan));
  }
  renderTopIssues(issues);
}

function cleanIssueTitle(rawTitle, aspectLabel) {
  let t = String(rawTitle || "").trim();
  // Hilangkan duplikasi "Isu Isu Citra" -> "Isu Citra Kampus"
  t = t.replace(/^Isu\s+Isu\s+Citra/i, "Isu Citra Kampus");
  // Pisahkan prefix aspek dan kata kunci jika ada ":"
  const parts = t.split(":");
  if (parts.length === 2) {
    const emojiNoise = new Set(["face", "heart", "loudly", "crying", "hands", "smiling", "folded", "sparkles", "fire", "eyes", "tears", "laughing", "thumbs", "pengikut", "lihat", "selengkapnya"]);
    const cleanWords = parts[1]
      .split(",")
      .map(w => w.trim())
      .filter(w => w && !emojiNoise.has(w.toLowerCase()));
    if (cleanWords.length > 0) {
      return `${parts[0].trim()}: ${cleanWords.join(", ")}`;
    }
    return parts[0].trim();
  }
  return t || (aspectLabel ? `Isu ${aspectLabel}` : "Isu Kampus");
}

function initTopIssuesFilters() {
  const btnAll = document.getElementById("btn-issue-filter-all");
  const btnNeg = document.getElementById("btn-issue-filter-neg");

  if (btnAll) {
    btnAll.addEventListener("click", () => {
      isFilterNegativeOnly = false;
      btnAll.classList.add("active");
      if (btnNeg) btnNeg.classList.remove("active");
      renderFilteredTopIssues();
    });
  }

  if (btnNeg) {
    btnNeg.addEventListener("click", () => {
      isFilterNegativeOnly = true;
      btnNeg.classList.add("active");
      if (btnAll) btnAll.classList.remove("active");
      renderFilteredTopIssues();
    });
  }
}

async function loadWeeklyMonitoringData(page = 1) {
  const weekSel = document.getElementById("wi-select-week");
  const platSel = document.getElementById("wi-filter-platform");
  const aspSel = document.getElementById("wi-filter-aspect");
  const sentSel = document.getElementById("wi-filter-sentiment");
  const offSel = document.getElementById("wi-filter-official");
  const searchInput = document.getElementById("wi-search-post");

  const params = new URLSearchParams();
  if (weekSel && weekSel.value) params.append("week", weekSel.value);
  if (platSel && platSel.value) params.append("platform", platSel.value);
  if (aspSel && aspSel.value) params.append("aspect", aspSel.value);
  if (sentSel && sentSel.value) params.append("sentiment", sentSel.value);
  if (offSel && offSel.value) params.append("official", offSel.value);
  if (searchInput && searchInput.value.trim()) params.append("search", searchInput.value.trim());
  params.append("page", page);
  params.append("limit", currentWiPageSize);
  params.append("_t", Date.now());

  try {
    const res = await fetch(`/api/weekly_monitoring_data?${params.toString()}`, { cache: "no-store" });
    const data = await res.json();

    if (!data.available) {
      const issuesContainer = document.getElementById("wi-top-issues-container");
      if (issuesContainer) {
        issuesContainer.innerHTML = `<div class="issue-loading-card">${escapeHtml(data.message || "Data belum tersedia.")}</div>`;
      }
      return;
    }

function getIsoWeekStartDate(weekStr) {
  const m = /^(\d{4})-W(\d{1,2})$/i.exec(String(weekStr || "").trim());
  if (!m) return null;
  const year = parseInt(m[1], 10);
  const week = parseInt(m[2], 10);
  const jan4 = new Date(Date.UTC(year, 0, 4));
  const jan4Day = jan4.getUTCDay() || 7;
  const mondayW1 = new Date(jan4.getTime() - (jan4Day - 1) * 86400000);
  return new Date(mondayW1.getTime() + (week - 1) * 7 * 86400000);
}

function formatIsoWeekLabel(weekStr) {
  const m = /^(\d{4})-W(\d{1,2})$/i.exec(String(weekStr || "").trim());
  const start = getIsoWeekStartDate(weekStr);
  if (!m || !start) return `Minggu ${weekStr}`;
  const week = parseInt(m[2], 10);
  const end = new Date(start.getTime() + 6 * 86400000);

  const months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"];
  const sD = start.getUTCDate();
  const sM = months[start.getUTCMonth()];
  const sY = start.getUTCFullYear();
  const eD = end.getUTCDate();
  const eM = months[end.getUTCMonth()];
  const eY = end.getUTCFullYear();

  let rangeText = "";
  if (sY !== eY) {
    rangeText = `${sD} ${sM} ${sY} – ${eD} ${eM} ${eY}`;
  } else if (sM !== eM) {
    rangeText = `${sD} ${sM} – ${eD} ${eM} ${eY}`;
  } else {
    rangeText = `${sD} – ${eD} ${eM} ${eY}`;
  }
  return `${rangeText} (Minggu ke-${week})`;
}

    // 1. Update Dropdown Minggu Secara Dinamis (urutkan sesuai tahun & bulan, dikelompokkan per Bulan Tahun)
    if (weekSel && data.available_weeks && data.available_weeks.length > 0) {
      const prevVal = weekSel.value;
      const existingCount = weekSel.querySelectorAll("option").length;
      if (existingCount !== data.available_weeks.length || existingCount === 0) {
        weekSel.innerHTML = "";
        const fullMonths = [
          "Januari", "Februari", "Maret", "April", "Mei", "Juni",
          "Juli", "Agustus", "September", "Oktober", "November", "Desember"
        ];
        const sortedWeeks = [...data.available_weeks].sort((a, b) => {
          const da = getIsoWeekStartDate(a);
          const db = getIsoWeekStartDate(b);
          return (db ? db.getTime() : 0) - (da ? da.getTime() : 0);
        });

        let currentGroupKey = null;
        let currentOptGroup = null;
        sortedWeeks.forEach((w) => {
          const start = getIsoWeekStartDate(w);
          const groupKey = start
            ? `${fullMonths[start.getUTCMonth()]} ${start.getUTCFullYear()}`
            : "Lainnya";
          if (groupKey !== currentGroupKey) {
            currentGroupKey = groupKey;
            currentOptGroup = document.createElement("optgroup");
            currentOptGroup.label = groupKey;
            weekSel.appendChild(currentOptGroup);
          }
          const opt = document.createElement("option");
          opt.value = w;
          opt.textContent = formatIsoWeekLabel(w);
          if (w === (prevVal || data.selected_week)) opt.selected = true;
          currentOptGroup.appendChild(opt);
        });
      }
      if (!weekSel.value && data.selected_week) {
        weekSel.value = data.selected_week;
      }
    }

    // 2. Update Ringkasan KPI
    const s = data.summary || {};
    const periodEl = document.getElementById("wi-period-text");
    if (periodEl) periodEl.textContent = `Rentang Tanggal: ${s.period || "-"}`;

    const kpiMentions = document.getElementById("wi-kpi-mentions");
    if (kpiMentions) kpiMentions.textContent = (s.total_mentions || 0).toLocaleString("id-ID");

    const kpiRepIdx = document.getElementById("wi-kpi-rep-index");
    if (kpiRepIdx) kpiRepIdx.textContent = s.reputation_index_pct || "0.00%";

    const trendIcon = document.getElementById("wi-trend-icon");
    if (trendIcon) {
      trendIcon.textContent = s.trend_arrow || "→";
      trendIcon.className = s.trend_type === "up" ? "text-green" : (s.trend_type === "down" ? "text-red" : "text-slate");
    }

    const deltaVal = document.getElementById("wi-delta-val");
    if (deltaVal) deltaVal.textContent = s.reputation_delta_pct || "0.00%";

    const kpiNeg = document.getElementById("wi-kpi-neg-pct");
    if (kpiNeg) kpiNeg.textContent = `${s.negatif_pct || 0}%`;

    const kpiSentSub = document.getElementById("wi-kpi-sent-breakdown");
    if (kpiSentSub) kpiSentSub.textContent = `Positif: ${s.positif_pct || 0}% • Netral: ${s.netral_pct || 0}%`;

    const kpiAlert = document.getElementById("wi-kpi-alert-status");
    if (kpiAlert) {
      kpiAlert.textContent = s.alert_status || "Aman";
      kpiAlert.className = "kpi-stat-value " + (s.alert_status === "Siaga" ? "text-red" : (s.alert_status === "Waspada" ? "text-orange" : "text-green"));
    }

    // Banner Data Terbatas
    const limitedBanner = document.getElementById("wi-limited-data-alert");
    if (limitedBanner) {
      limitedBanner.style.display = s.is_limited_data ? "flex" : "none";
    }

    // 3. Render Top 5 Isu Saat Ini (disimpan untuk filtering negatif)
    rawWeeklyTopIssues = data.top_5_issues || [];
    renderFilteredTopIssues();

    // 4. Render Grafik
    renderDailySentimentTrendChart(data.daily_trend);
    renderPlatformDistChart(data.platform_dist);
    renderHeatmapMatrix(data.heatmap);
    renderTopWordsPills(data.top_words);
    renderMultiWeekReputationChart(data.multi_week_trend);

    // 5. Render Tabel Postingan & Pagination
    renderWeeklyPostsTable(data.posts);

  } catch (err) {
    console.error("Gagal memuat monitoring mingguan:", err);
  }
}

function renderTopIssues(issues) {
  const container = document.getElementById("wi-top-issues-container");
  if (!container) return;

  if (issues.length === 0) {
    container.innerHTML = `
      <div class="issue-empty-safe-card">
        <div class="safe-icon">✅</div>
        <div>
          <h4>Tidak Ada Isu Negatif Terdeteksi Minggu Ini</h4>
          <p>Seluruh opini publik yang terkumpul pada periode ini bersifat positif atau netral (0% keluhan negatif).</p>
        </div>
      </div>
    `;
    return;
  }

  container.innerHTML = issues.map((issue, idx) => {
    const negPct = Number(issue.persen_negatif || 0);
    const alertLevel = negPct >= 30 ? "tinggi" : (negPct >= 10 ? "sedang" : (issue.tingkat_perhatian || "Rendah").toLowerCase());
    const badgeClass = `badge-${alertLevel}`;
    const cardClass = `alert-${alertLevel}`;
    const rankLabel = `#${idx + 1}`;
    const estNegCount = Math.max(negPct > 0 ? 1 : 0, Math.round((negPct / 100) * Number(issue.sebutan || 0)));
    const displayTitle = cleanIssueTitle(issue.judul_isu, issue.aspek_label || issue.aspek);

    const commentsHtml = (issue.contoh_komentar || []).filter(c => c && c !== "-").map((c, cIdx) => `
      <li class="issue-comment-quote">
        <div class="comment-author-chip">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          <span>Opini Sampel ${cIdx + 1}</span>
        </div>
        <div class="comment-quote-text">"${escapeHtml(c)}"</div>
      </li>
    `).join("") || `<li class="issue-comment-quote"><div class="comment-quote-text">Tidak ada contoh komentar spesifik.</div></li>`;

    const safeAspect = escapeHtml(issue.aspek || "").replace(/'/g, "\\'");
    const safeTitle = escapeHtml(displayTitle).replace(/'/g, "\\'");
    const barWidth = Math.min(100, Math.max(negPct > 0 ? 4 : 0, negPct));

    return `
      <div class="issue-card issue-card-modern clickable-issue ${cardClass}" onclick="if(!event.target.closest('details')) openWeeklyDrilldown('aspect', '${safeAspect}', '${safeTitle}', 'negatif')">
        <div>
          <div class="issue-card-top">
            <div class="issue-badge-group">
              <span class="issue-rank-badge">${rankLabel}</span>
              <span class="issue-aspect-pill">${escapeHtml(issue.aspek_label || issue.aspek)}</span>
            </div>
            <span class="issue-badge-alert ${badgeClass}">
              <span class="alert-indicator-dot"></span>
              ${escapeHtml(issue.tingkat_perhatian)}
            </span>
          </div>
          <h3 class="issue-title">${escapeHtml(displayTitle)}</h3>

          <!-- Visual Bar Tingkat Negatif -->
          <div class="issue-neg-meter">
            <div class="issue-neg-meter-head">
              <span class="neg-meter-label">Proporsi Keluhan / Negatif</span>
              <span class="neg-meter-val">${negPct}% (~${estNegCount} dari ${issue.sebutan} opini)</span>
            </div>
            <div class="issue-neg-bar-track">
              <div class="issue-neg-bar-fill" style="width: ${barWidth}%;"></div>
            </div>
          </div>

          <!-- 3 Metrik Ringkas Tanpa Berdesakan -->
          <div class="issue-clean-metrics">
            <div class="icm-box">
              <span class="icm-val">${issue.sebutan}</span>
              <span class="icm-lbl">Total Sebutan</span>
            </div>
            <div class="icm-box">
              <span class="icm-val">${escapeHtml(issue.perubahan_minggu_lalu)}</span>
              <span class="icm-lbl">Lonjakan</span>
            </div>
            <div class="icm-box">
              <span class="icm-val">${issue.skor_isu}</span>
              <span class="icm-lbl">Skor Urgensi</span>
            </div>
          </div>
        </div>

        <button type="button" class="btn-issue-drilldown" onclick="event.stopPropagation(); openWeeklyDrilldown('aspect', '${safeAspect}', '${safeTitle}', 'negatif')">
          📊 Bedah Keluhan &amp; Sentimen &rarr;
        </button>

        <div class="issue-comments-details">
          <details>
            <summary class="issue-comments-toggle">
              <span>💬 Contoh Komentar</span>
              <span class="toggle-arrow">▾</span>
            </summary>
            <ul class="issue-comments-list">
              ${commentsHtml}
            </ul>
          </details>
        </div>
      </div>
    `;
  }).join("");
}

function renderDailySentimentTrendChart(trendData) {
  const ctx = document.getElementById("chartDailySentimentTrend");
  if (!ctx || !trendData) return;

  if (wiDailyChart) wiDailyChart.destroy();

  wiDailyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: trendData.labels || [],
      datasets: [
        {
          label: "Positif",
          data: trendData.positif || [],
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          tension: 0.3,
          fill: true,
          pointRadius: 4,
        },
        {
          label: "Netral",
          data: trendData.netral || [],
          borderColor: "#94a3b8",
          backgroundColor: "rgba(148, 163, 184, 0.15)",
          tension: 0.3,
          fill: true,
          pointRadius: 4,
        },
        {
          label: "Negatif",
          data: trendData.negatif || [],
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.15)",
          tension: 0.3,
          fill: true,
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
}

function renderPlatformDistChart(platData) {
  const ctx = document.getElementById("chartPlatformSentimentDist");
  if (!ctx || !platData) return;

  if (wiPlatformChart) wiPlatformChart.destroy();

  wiPlatformChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: platData.platforms || [],
      datasets: [
        {
          label: "Positif",
          data: platData.positif || [],
          backgroundColor: "#10b981",
        },
        {
          label: "Netral",
          data: platData.netral || [],
          backgroundColor: "#94a3b8",
        },
        {
          label: "Negatif",
          data: platData.negatif || [],
          backgroundColor: "#ef4444",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
}

function renderHeatmapMatrix(heatmapData) {
  const container = document.getElementById("wi-heatmap-container");
  if (!container || !heatmapData) return;

  const aspects = heatmapData.aspects || [];
  const platforms = heatmapData.platforms || [];
  const matrix = heatmapData.matrix || [];

  if (aspects.length === 0 || platforms.length === 0) {
    container.innerHTML = `<div class="p-3 text-center text-muted">Data sebaran aspek belum cukup.</div>`;
    return;
  }

  // Hitung max value untuk densitas warna
  let maxVal = 1;
  matrix.forEach(row => {
    row.forEach(v => { if (v > maxVal) maxVal = v; });
  });

  let thHeaders = `<th>Aspek Reputasi</th>` + platforms.map(p => `<th>${escapeHtml(p)}</th>`).join("");
  let rowsHtml = aspects.map((asp, i) => {
    let cellsHtml = platforms.map((p, j) => {
      const val = matrix[i] ? matrix[i][j] || 0 : 0;
      let densityClass = "cell-density-0";
      if (val > 0) {
        const ratio = val / maxVal;
        densityClass = ratio > 0.75 ? "cell-density-4" : (ratio > 0.5 ? "cell-density-3" : (ratio > 0.25 ? "cell-density-2" : "cell-density-1"));
      }
      return `<td class="${densityClass}">${val}</td>`;
    }).join("");
    return `<tr><td class="aspect-name-th">${escapeHtml(asp)}</td>${cellsHtml}</tr>`;
  }).join("");

  container.innerHTML = `
    <table class="heatmap-table">
      <thead><tr>${thHeaders}</tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
  `;
}

function renderTopWordsPills(wordsList) {
  const container = document.getElementById("wi-top-words-container");
  if (!container) return;

  if (!wordsList || wordsList.length === 0) {
    container.innerHTML = `<div class="text-muted">Tidak ada kata kunci dominan.</div>`;
    return;
  }

  container.innerHTML = wordsList.map(w => {
    const safeWord = escapeHtml(w.word).replace(/'/g, "\\'");
    return `
      <span class="word-pill clickable-word" title="Klik untuk melihat analisis sentimen kata '${escapeHtml(w.word)}'" onclick="openWeeklyDrilldown('keyword', '${safeWord}', 'Kata Kunci: ${safeWord}')">
        ${escapeHtml(w.word)}
        <span class="word-pill-count">${w.count}</span>
      </span>
    `;
  }).join("");
}

/* ─────────────────────────────────────────────────────────────
   MODAL DRILL-DOWN SENTIMEN ISU & KATA KUNCI
   ───────────────────────────────────────────────────────────── */
let ddCurrentOpinions = [];
let ddCurrentSentFilter = "all";

function renderDrilldownOpinions() {
  const listEl = document.getElementById("dd-opinions-list");
  if (!listEl) return;

  const filtered = ddCurrentSentFilter === "all"
    ? ddCurrentOpinions
    : ddCurrentOpinions.filter(o => (o.sentimen || "").toLowerCase() === ddCurrentSentFilter);

  if (filtered.length === 0) {
    listEl.innerHTML = `<div class="text-center text-muted" style="padding: 2rem;">Tidak ada opini dengan sentimen ini.</div>`;
    return;
  }

  listEl.innerHTML = filtered.map(o => {
    const pLower = (o.platform || "").toLowerCase();
    let pBadge = "badge-tw";
    if (pLower.includes("instagram")) pBadge = "badge-ig";
    else if (pLower.includes("youtube")) pBadge = "badge-yt";
    else if (pLower.includes("facebook") || pLower.includes("fb")) pBadge = "badge-fb";
    else if (pLower.includes("threads")) pBadge = "badge-threads";
    else if (pLower.includes("tiktok")) pBadge = "badge-tiktok";

    const s = (o.sentimen || "netral").toLowerCase();
    const sLabel = s.charAt(0).toUpperCase() + s.slice(1);
    const linkBtn = o.url && o.url.trim()
      ? `<a href="${escapeHtml(o.url)}" target="_blank" rel="noopener noreferrer" class="btn-table-link" title="Buka sumber asli"><span>Buka</span><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3"/></svg></a>`
      : "";

    return `
      <div class="dd-opinion-item">
        <div class="dd-opinion-meta">
          <div class="dd-opinion-badges">
            <span class="platform-badge ${pBadge}">${escapeHtml(o.platform)}</span>
            <span class="sentiment-pill pill-${s}">${escapeHtml(sLabel)} (${o.probabilitas}%)</span>
            <span class="aspect-tag">${escapeHtml(o.aspek)}</span>
          </div>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size:0.75rem; color:#64748b;">${escapeHtml(o.created_at)}</span>
            ${linkBtn}
          </div>
        </div>
        <div class="dd-opinion-text">${escapeHtml(o.teks_asli)}</div>
      </div>
    `;
  }).join("");
}

window.openWeeklyDrilldown = async function(type, value, title, defaultSent = "all") {
  const modal = document.getElementById("wi-sentiment-drilldown-modal");
  const titleEl = document.getElementById("drilldown-modal-title");
  const listEl = document.getElementById("dd-opinions-list");
  if (!modal) return;

  if (titleEl) titleEl.innerText = `📊 Sentimen: ${title || value}`;
  if (listEl) listEl.innerHTML = `<div class="text-center text-muted" style="padding: 2rem;">Memuat analisis sentimen...</div>`;

  ddCurrentSentFilter = defaultSent || "all";
  document.querySelectorAll("#dd-filter-tabs .dd-tab-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.sent === ddCurrentSentFilter);
  });

  modal.style.setProperty("display", "flex", "important");

  const weekSel = document.getElementById("wi-select-week");
  const params = new URLSearchParams();
  if (weekSel && weekSel.value) params.append("week", weekSel.value);
  if (type === "aspect") params.append("aspect", value);
  if (type === "keyword") params.append("keyword", value);
  params.append("limit", 120);
  params.append("_t", Date.now());

  try {
    let data = null;
    const res = await fetch(`/api/weekly_drilldown?${params.toString()}`, { cache: "no-store" });
    const contentType = res.headers.get("content-type") || "";
    if (res.ok && contentType.includes("application/json")) {
      data = await res.json();
    } else {
      // Fallback jika server Flask belum di-restart setelah penambahan route /api/weekly_drilldown
      const fbParams = new URLSearchParams();
      if (weekSel && weekSel.value) fbParams.append("week", weekSel.value);
      if (type === "aspect") fbParams.append("aspect", value);
      if (type === "keyword") fbParams.append("search", value);
      fbParams.append("limit", 200);
      fbParams.append("_t", Date.now());
      const fbRes = await fetch(`/api/weekly_monitoring_data?${fbParams.toString()}`, { cache: "no-store" });
      const fbJson = await fbRes.json();
      const rows = fbJson?.posts?.data || [];
      const tot = fbJson?.posts?.total ?? rows.length;
      const posC = rows.filter(r => (r.sentimen || "").toLowerCase() === "positif").length;
      const neuC = rows.filter(r => (r.sentimen || "").toLowerCase() === "netral").length;
      const negC = rows.filter(r => (r.sentimen || "").toLowerCase() === "negatif").length;
      const denom = rows.length || 1;
      data = {
        total: tot,
        positif: posC,
        netral: neuC,
        negatif: negC,
        positif_pct: Number(((posC / denom) * 100).toFixed(1)),
        netral_pct: Number(((neuC / denom) * 100).toFixed(1)),
        negatif_pct: Number(((negC / denom) * 100).toFixed(1)),
        opinions: rows.map(r => ({
          platform: r.platform,
          teks_asli: r.teks_asli,
          sentimen: (r.sentimen || "").toLowerCase(),
          aspek: r.aspek,
          probabilitas: r.probabilitas,
          created_at: r.created_at,
          url: r.url || ""
        }))
      };
    }

    document.getElementById("dd-total-count").innerText = (data.total || 0).toLocaleString("id-ID");
    document.getElementById("dd-pos-count").innerText = `${data.positif || 0} (${data.positif_pct || 0}%)`;
    document.getElementById("dd-neu-count").innerText = `${data.netral || 0} (${data.netral_pct || 0}%)`;
    document.getElementById("dd-neg-count").innerText = `${data.negatif || 0} (${data.negatif_pct || 0}%)`;

    document.getElementById("dd-bar-pos").style.width = `${data.positif_pct || 0}%`;
    document.getElementById("dd-bar-neu").style.width = `${data.netral_pct || 0}%`;
    document.getElementById("dd-bar-neg").style.width = `${data.negatif_pct || 0}%`;

    ddCurrentOpinions = data.opinions || [];
    renderDrilldownOpinions();
  } catch (err) {
    if (listEl) listEl.innerHTML = `<div class="text-center text-danger" style="padding: 2rem;">Gagal memuat data sentimen: ${escapeHtml(err.message)}</div>`;
  }
};

function closeWeeklyDrilldownModal() {
  const modal = document.getElementById("wi-sentiment-drilldown-modal");
  if (modal) modal.style.setProperty("display", "none", "important");
}

document.addEventListener("DOMContentLoaded", () => {
  const ddModal = document.getElementById("wi-sentiment-drilldown-modal");
  const ddClose = document.getElementById("btn-close-drilldown-modal");
  if (ddClose) ddClose.addEventListener("click", closeWeeklyDrilldownModal);
  if (ddModal) {
    ddModal.addEventListener("click", (e) => {
      if (e.target === ddModal) closeWeeklyDrilldownModal();
    });
  }
  document.querySelectorAll("#dd-filter-tabs .dd-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      ddCurrentSentFilter = btn.dataset.sent || "all";
      document.querySelectorAll("#dd-filter-tabs .dd-tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderDrilldownOpinions();
    });
  });
});

function renderMultiWeekReputationChart(trend) {
  const ctx = document.getElementById("chartMultiWeekReputationTrend");
  if (!ctx || !trend) return;

  if (wiMultiWeekChart) wiMultiWeekChart.destroy();

  wiMultiWeekChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: trend.weeks || [],
      datasets: [
        {
          label: "Indeks Reputasi Kampus (%)",
          data: trend.reputation_index || [],
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.1)",
          borderWidth: 3,
          tension: 0.3,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
          yAxisID: "y",
        },
        {
          type: "bar",
          label: "Total Sebutan (Volume)",
          data: trend.volume || [],
          backgroundColor: "rgba(148, 163, 184, 0.35)",
          borderRadius: 4,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        y: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Indeks Reputasi (%)" },
        },
        y1: {
          type: "linear",
          position: "right",
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Volume Sebutan" },
          beginAtZero: true,
        },
      },
    },
  });
}

function renderWiPaginationButtons(currentPage, totalPages) {
  const container = document.getElementById("wi-pagination-pages-container");
  const prevBtn = document.getElementById("wi-btn-prev-page");
  const nextBtn = document.getElementById("wi-btn-next-page");

  if (prevBtn) prevBtn.disabled = currentPage <= 1;
  if (nextBtn) nextBtn.disabled = currentPage >= totalPages;

  if (!container) return;
  container.innerHTML = "";

  if (totalPages <= 1) return;

  const pages = [];
  const delta = 2;
  let left = Math.max(1, currentPage - delta);
  let right = Math.min(totalPages, currentPage + delta);

  if (left > 1) {
    pages.push(1);
    if (left > 2) pages.push("...");
  }
  for (let i = left; i <= right; i++) {
    pages.push(i);
  }
  if (right < totalPages) {
    if (right < totalPages - 1) pages.push("...");
    pages.push(totalPages);
  }

  pages.forEach((p) => {
    if (p === "...") {
      const span = document.createElement("span");
      span.className = "page-num-dots";
      span.innerText = "...";
      container.appendChild(span);
    } else {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `page-num-btn ${p === currentPage ? "active" : ""}`;
      btn.innerText = p;
      btn.title = `Geser ke halaman ${p}`;
      btn.addEventListener("click", () => {
        currentWiPage = p;
        loadWeeklyMonitoringData(p);
      });
      container.appendChild(btn);
    }
  });
}

function renderWeeklyPostsTable(postsData) {
  const tbody = document.getElementById("wi-posts-tbody");
  const countSpan = document.getElementById("wi-data-count-text");
  const rangeSpan = document.getElementById("wi-pagination-range-text");

  if (!tbody || !postsData) return;

  totalWiPages = postsData.total_pages || 1;
  currentWiPage = postsData.page || 1;
  const totalCount = postsData.total || 0;
  const pageSize = postsData.page_size || currentWiPageSize;

  if (countSpan) {
    countSpan.innerText = `Total ${Number(totalCount).toLocaleString("id-ID")} opini terfilter`;
  }

  if (rangeSpan) {
    if (totalCount === 0) {
      rangeSpan.innerText = "Menampilkan 0 dari 0 data";
    } else {
      const startNum = (currentWiPage - 1) * pageSize + 1;
      const endNum = Math.min(currentWiPage * pageSize, totalCount);
      rangeSpan.innerText = `Menampilkan ${Number(startNum).toLocaleString("id-ID")} - ${Number(endNum).toLocaleString("id-ID")} dari ${Number(totalCount).toLocaleString("id-ID")} opini`;
    }
  }

  const records = postsData.data || [];
  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #94a3b8; padding: 2.5rem 1rem;">Tidak ada postingan yang sesuai dengan filter yang dipilih.</td></tr>`;
    renderWiPaginationButtons(currentWiPage, totalWiPages);
    return;
  }

  currentWiRows = {};
  records.forEach((r) => {
    const key = r.full_id || r.id_anonim;
    currentWiRows[key] = r;
  });

  tbody.innerHTML = records.map(r => {
    const rowKey = r.full_id || r.id_anonim;
    const pLower = (r.platform || "").toLowerCase();
    let platformBadgeClass = "badge-tw";
    if (pLower.includes("instagram")) platformBadgeClass = "badge-ig";
    else if (pLower.includes("youtube")) platformBadgeClass = "badge-yt";
    else if (pLower.includes("facebook") || pLower.includes("fb")) platformBadgeClass = "badge-fb";
    else if (pLower.includes("threads")) platformBadgeClass = "badge-threads";
    else if (pLower.includes("tiktok")) platformBadgeClass = "badge-tiktok";

    const s = (r.sentimen || "").toLowerCase();
    const sentLabel = s ? (s.charAt(0).toUpperCase() + s.slice(1)) : "Netral";

    const editBtn = `<button type="button" class="btn-table-edit" onclick="openWiEditModalById('${rowKey}')" title="Koreksi sentimen atau aspek">
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
      <span>Edit</span>
    </button>`;

    const linkBtn =
      r.url && r.url.trim()
        ? `<a href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer" class="btn-table-link" title="Buka tautan postingan asli">
            <span>Buka</span>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3"/></svg>
           </a>`
        : `<span style="color: #cbd5e1; font-size: 0.75rem;">-</span>`;

    const dateDisplay = (!r.created_at || r.created_at === "nan" || r.created_at === "None") ? "-" : r.created_at;

    return `
      <tr>
        <td><span class="platform-badge ${platformBadgeClass}">${escapeHtml(r.platform)}</span></td>
        <td style="max-width: 480px; word-break: break-word;">${escapeHtml(r.teks_asli)}</td>
        <td><span class="sentiment-pill pill-${s}">${escapeHtml(sentLabel)}</span></td>
        <td><span class="aspect-tag">${escapeHtml(r.aspek)}</span></td>
        <td style="color: #64748b; font-size: 0.8rem; white-space: nowrap;">${escapeHtml(dateDisplay)}</td>
        <td style="text-align: center; white-space: nowrap;">
          <div class="table-action-cell">
            ${editBtn}
            ${linkBtn}
          </div>
        </td>
      </tr>
    `;
  }).join("");

  renderWiPaginationButtons(currentWiPage, totalWiPages);
}

/* ─────────────────────────────────────────────────────────────
   RIWAYAT ISU HISTORIS
   ───────────────────────────────────────────────────────────── */

async function loadWeeklyHistory() {
  const containerRec = document.getElementById("wi-recurring-container");
  const tbody = document.getElementById("wi-history-tbody");

  try {
    const res = await fetch(`/api/weekly_issues_history?_t=${Date.now()}`, { cache: "no-store" });
    const data = await res.json();

    if (!data.available) {
      if (containerRec) containerRec.innerHTML = `<div class="p-3 text-muted">Belum ada riwayat isu.</div>`;
      if (tbody) tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted">Belum ada arsip isu.</td></tr>`;
      return;
    }

    // 1. Recurring Issues Analysis
    if (containerRec) {
      const recList = data.recurring_analysis || [];
      if (recList.length === 0) {
        containerRec.innerHTML = `<div class="p-3 text-muted">Belum terdeteksi isu berulang kronis.</div>`;
      } else {
        containerRec.innerHTML = recList.map(item => `
          <div class="recurring-card">
            <h4 class="recurring-card-aspect">${escapeHtml(item.aspek)}</h4>
            <div class="recurring-card-meta">Muncul dalam <strong>${item.frekuensi_minggu} minggu</strong> berbeda</div>
            <div class="recurring-card-meta">Total Sebutan Terakumulasi: <strong>${item.total_sebutan} sebutan</strong></div>
            <div class="recurring-card-meta">Skor Rata-Rata: <strong>${item.skor_rata_rata}</strong> &bull; <span class="badge-tinggi" style="font-size: 0.7rem; padding: 2px 6px;">${escapeHtml(item.status)}</span></div>
          </div>
        `).join("");
      }
    }

    // 2. History Table
    if (tbody) {
      const issues = data.issues || [];
      tbody.innerHTML = issues.map(ir => {
        const alertClass = (ir.tingkat_perhatian || "Rendah").toLowerCase();
        const badgeClass = `badge-${alertClass}`;
        return `
          <tr>
            <td><strong>${escapeHtml(ir.minggu)}</strong></td>
            <td><strong>${escapeHtml(ir.judul_isu)}</strong></td>
            <td>${escapeHtml(ir.aspek_label || ir.aspek)}</td>
            <td>${ir.sebutan}</td>
            <td class="text-red"><strong>${ir.persen_negatif}%</strong></td>
            <td>${escapeHtml(ir.perubahan_minggu_lalu)}</td>
            <td><strong>${ir.skor_isu}</strong></td>
            <td><span class="issue-badge-alert ${badgeClass}">${escapeHtml(ir.tingkat_perhatian)}</span></td>
            <td style="max-width: 280px; font-style: italic; font-size: 0.78rem;">"${escapeHtml(ir.contoh_komentar_1)}"</td>
          </tr>
        `;
      }).join("");
    }
  } catch (err) {
    console.error("Gagal memuat riwayat isu:", err);
  }
}

/* ─────────────────────────────────────────────────────────────
   UJI PREDIKSI MULTI-KOMPONEN & ENSEMBLE
   ───────────────────────────────────────────────────────────── */

function initMultiComponentPredictor() {
  const btn = document.getElementById("btn-predict");
  const textarea = document.getElementById("predict-text");
  const quickPills = document.querySelectorAll(".quick-pill");

  quickPills.forEach(pill => {
    pill.addEventListener("click", () => {
      if (textarea) textarea.value = pill.dataset.text || "";
    });
  });

  if (btn) {
    btn.addEventListener("click", runMultiComponentPrediction);
  }
}

async function runMultiComponentPrediction() {
  const textarea = document.getElementById("predict-text");
  const resultBox = document.getElementById("predict-result");
  const btn = document.getElementById("btn-predict");

  const text = (textarea?.value || "").trim();
  if (!text) {
    alert("Masukkan teks opini terlebih dahulu.");
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="pulse-dot"></span> Menganalisis...`;
  }

  try {
    const res = await fetch("/api/predict_multicomponent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (resultBox) resultBox.style.display = "block";

    // 1. Verdict Banner (Ensemble Stacking)
    const predBadge = document.getElementById("pred-badge");
    const predConf = document.getElementById("pred-confidence");
    const predAspect = document.getElementById("pred-aspect");

    const verdict = data.final_verdict || {};
    const sent = (verdict.sentiment || "netral").toLowerCase();
    const sentClass = sent === "positif" ? "pill-positif" : (sent === "negatif" ? "pill-negatif" : "pill-netral");

    if (predBadge) {
      predBadge.className = `result-badge ${sentClass}`;
      predBadge.textContent = sent.toUpperCase();
    }
    if (predConf) predConf.textContent = `${verdict.confidence || 0}%`;
    if (predAspect) predAspect.textContent = (data.detected_aspect_label || data.detected_aspect || "-").toUpperCase();

    // 2. Comparison Table
    const tbody = document.getElementById("tbody-component-comparison");
    if (tbody) {
      const rows = [
        {
          model: "Komponen A: Lexicon InSet",
          sent: data.components?.lexicon?.sentiment,
          probs: data.components?.lexicon?.probabilities,
        },
        {
          model: "Komponen B: TF-IDF + SVM",
          sent: data.components?.svm?.sentiment,
          probs: data.components?.svm?.probabilities,
        },
        {
          model: "Komponen C: IndoBERT Pretrained",
          sent: data.components?.indobert?.sentiment,
          probs: data.components?.indobert?.probabilities,
        },
        {
          model: "Ensemble: Weighted Soft Voting",
          sent: data.ensemble?.soft_voting?.sentiment,
          probs: data.ensemble?.soft_voting?.probabilities,
        },
        {
          model: "Ensemble: Stacking (Meta-Learner LR)",
          sent: data.ensemble?.stacking?.sentiment,
          probs: data.ensemble?.stacking?.probabilities,
        },
      ];

      tbody.innerHTML = rows.map(r => {
        const rSent = (r.sent || "netral").toLowerCase();
        const rBadge = rSent === "positif" ? "pill-positif" : (rSent === "negatif" ? "pill-negatif" : "pill-netral");
        const pNeg = r.probs?.negatif || 0;
        const pNet = r.probs?.netral || 0;
        const pPos = r.probs?.positif || 0;

        return `
          <tr>
            <td><strong>${escapeHtml(r.model)}</strong></td>
            <td><span class="result-badge ${rBadge}">${escapeHtml(rSent.toUpperCase())}</span></td>
            <td class="text-red">${pNeg}%</td>
            <td class="text-slate">${pNet}%</td>
            <td class="text-green">${pPos}%</td>
            <td>
              <div class="prob-bar-stacked" title="Negatif: ${pNeg}% | Netral: ${pNet}% | Positif: ${pPos}%">
                <div class="prob-seg neg" style="width: ${pNeg}%;"></div>
                <div class="prob-seg net" style="width: ${pNet}%;"></div>
                <div class="prob-seg pos" style="width: ${pPos}%;"></div>
              </div>
            </td>
          </tr>
        `;
      }).join("");
    }

  } catch (err) {
    alert("Gagal melakukan prediksi: " + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Jalankan Prediksi Lengkap</span>
      `;
    }
  }
}


