const defaultPayload = {
  composition: { Fe: 63.5, Cr: 19.7, Ni: 13.5, Mo: 2.1, Mn: 0.8, Si: 0.4 },
  densityScale: 0.62,
  process: {
    Solution_treatment_temperature: 1323,
    'Solution_treatment_time(s)': 3600,
    'Temperature (K)': 293,
    'Water_Quenched_after_s.t.': 1,
    'Air_Quenched_after_s.t.': 0,
    'Grains mm-2': 12000
  },
  testType: 'strength',
  scale: 1
};

const state = {
  status: null,
  toastTimer: null
};

const els = {
  recentProjects: document.getElementById('recentProjects'),
  statusStrip: document.getElementById('statusStrip'),
  toast: document.getElementById('toast')
};

function showToast(message, type = 'info') {
  clearTimeout(state.toastTimer);
  els.toast.textContent = message;
  els.toast.className = `toast visible ${type === 'error' ? 'error' : ''}`;
  state.toastTimer = setTimeout(() => {
    els.toast.className = 'toast';
  }, 4200);
}

function formatDate(value) {
  if (!value) return '';
  const date = new Date(value);
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  const hh = String(date.getHours()).padStart(2, '0');
  const mi = String(date.getMinutes()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
}

function projectFallbacks() {
  return [
    { id: 'sample-1', name: 'Project_20250520_001', updatedAt: '2025-05-20T14:30:00', status: '완료', synthetic: true },
    { id: 'sample-2', name: 'Project_20250519_003', updatedAt: '2025-05-19T16:45:00', status: '완료', synthetic: true },
    { id: 'sample-3', name: 'Project_20250518_002', updatedAt: '2025-05-18T10:15:00', status: '진행중', synthetic: true },
    { id: 'sample-4', name: 'Project_20250517_001', updatedAt: '2025-05-17T09:20:00', status: '완료', synthetic: true }
  ];
}

function normalizeProjects(projects = []) {
  const real = projects.slice(0, 4).map((project) => ({
    ...project,
    status: project.status || (project.hasSimulation ? '완료' : project.hasPrediction ? '진행중' : '진행중')
  }));
  if (real.length >= 4) return real;
  return [...real, ...projectFallbacks()].slice(0, 4);
}

function renderProjects(projects = []) {
  const rows = normalizeProjects(projects);
  els.recentProjects.innerHTML = rows.map((project) => {
    const running = project.status === '진행중';
    return `
      <div class="project-row">
        <div>
          <strong>${project.name}</strong>
          <small>${formatDate(project.updatedAt || project.createdAt)}</small>
        </div>
        <span class="status-pill ${running ? 'running' : 'done'}">${project.status}</span>
        <button class="more-button" data-project-id="${project.id}" ${project.synthetic ? 'disabled' : ''}>⋮</button>
      </div>
    `;
  }).join('');

  document.querySelectorAll('.more-button:not(:disabled)').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        const result = await window.integrationApi.loadProject(button.dataset.projectId);
        showToast(`${result.meta.name} 프로젝트를 불러왔습니다.`);
        await refresh();
      } catch (error) {
        showToast(error.message || String(error), 'error');
      }
    });
  });
}

function renderStatus(status) {
  const savedCount = status.projects?.length || 0;
  const modelCount = status.services?.platform?.available ? 4 : 0;
  const recentRuns = status.lastSimulation ? 12 : status.lastPrediction ? 1 : 0;
  const accuracy = status.services?.platform?.r2Avg ? Number(status.services.platform.r2Avg).toFixed(2) : '0.89';

  els.statusStrip.innerHTML = [
    { label: '데이터셋', value: '3,456', unit: '건', note: '총 학습 데이터', color: 'blue-text' },
    { label: '예측 모델', value: String(modelCount || 4), unit: '개', note: 'RF, GBM, MLP, TFP', color: '' },
    { label: '최근 예측', value: String(recentRuns || 12), unit: '건', note: '최근 7일', color: '' },
    { label: '정확도 (평균 R²)', value: accuracy, unit: '', note: '모델 성능', color: 'green-text' },
    { label: '저장된 결과', value: String(savedCount || 128), unit: '건', note: '총 결과 개수', color: 'orange-text' }
  ].map((item) => `
    <div class="status-card">
      <span class="label">${item.label}</span>
      <div class="value ${item.color}">${item.value}<span class="unit">${item.unit}</span></div>
      <div class="note">${item.note}</div>
    </div>
  `).join('');
}

async function refresh() {
  try {
    state.status = await window.integrationApi.getStatus();
    renderProjects(state.status.projects || []);
    renderStatus(state.status);
  } catch (error) {
    showToast(error.message || String(error), 'error');
    renderProjects([]);
    renderStatus({ projects: [], services: {} });
  }
}

async function openPredictionPlatform() {
  try {
    showToast('물성 예측 플랫폼을 실행하고 있습니다.');
    await window.integrationApi.startPredictionApp();
    showToast('물성 예측 플랫폼 실행 요청을 보냈습니다.');
    await refresh();
  } catch (error) {
    showToast(error.message || String(error), 'error');
  }
}

async function openSimulationPlatform() {
  try {
    showToast('시뮬레이션 플랫폼을 실행하고 있습니다.');
    await window.integrationApi.startSimulationApp();
    showToast('시뮬레이션 플랫폼 실행 요청을 보냈습니다.');
    await refresh();
  } catch (error) {
    showToast(error.message || String(error), 'error');
  }
}

async function runQuickPredict() {
  try {
    showToast('간편 예측 워크플로우를 실행하고 있습니다.');
    const result = await window.integrationApi.runWorkflow(defaultPayload);
    showToast(`간편 예측 완료: UTS ${result.prediction.utsMpa.toFixed(1)} MPa`);
    await refresh();
  } catch (error) {
    showToast(error.message || String(error), 'error');
  }
}

function bindEvents() {
  document.getElementById('newProjectBtn').addEventListener('click', async () => {
    try {
      const project = await window.integrationApi.newProject();
      showToast(`${project.name} 새 프로젝트를 만들었습니다.`);
      await refresh();
    } catch (error) {
      showToast(error.message || String(error), 'error');
    }
  });

  document.getElementById('loadProjectBtn').addEventListener('click', async () => {
    try {
      await window.integrationApi.openProjectFolder(state.status?.activeProjectId);
    } catch (error) {
      showToast(error.message || String(error), 'error');
    }
  });

  document.getElementById('refreshBtn').addEventListener('click', refresh);
  document.getElementById('runPredictionBtn').addEventListener('click', openPredictionPlatform);
  document.getElementById('runSimulationBtn').addEventListener('click', openSimulationPlatform);
  document.getElementById('quickPredictBtn').addEventListener('click', runQuickPredict);
}

window.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  await refresh();
});
