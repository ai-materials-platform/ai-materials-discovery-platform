const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const fsp = fs.promises;
const http = require('http');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const projectsDir = path.join(rootDir, 'projects');
const predictionRepoDir = process.env.AI_MATERIALS_PLATFORM_DIR || rootDir;
const simulationRepoDir = process.env.AI_MATERIALS_SIMULATION_DIR || path.resolve(rootDir, '..', 'ai-materials-discovery-platform-simulation');
const simulationApiBase = 'http://127.0.0.1:8765';

function readVenvExecutable(repoDir) {
  const cfgPath = path.join(repoDir, '.venv', 'pyvenv.cfg');
  try {
    const cfg = fs.readFileSync(cfgPath, 'utf8');
    const line = cfg.split(/\r?\n/).find((row) => row.toLowerCase().startsWith('executable ='));
    if (!line) return null;
    const executable = line.split('=').slice(1).join('=').trim();
    return fs.existsSync(executable) ? executable : null;
  } catch (_) {
    return null;
  }
}

function venvSitePackages(repoDir) {
  const sitePackages = path.join(repoDir, '.venv', 'Lib', 'site-packages');
  return fs.existsSync(sitePackages) ? sitePackages : null;
}

function pythonEnv(extra = {}) {
  const paths = [
    venvSitePackages(predictionRepoDir),
    venvSitePackages(simulationRepoDir),
    process.env.PYTHONPATH
  ].filter(Boolean);
  return { ...extra, PYTHONPATH: paths.join(path.delimiter), PATH: runtimePath() };
}

function resolvePythonExe() {
  const candidates = [
    process.env.AI_MATERIALS_PYTHON,
    readVenvExecutable(predictionRepoDir),
    readVenvExecutable(simulationRepoDir),
    path.join(predictionRepoDir, '.venv', 'Scripts', 'python.exe'),
    path.join(simulationRepoDir, '.venv', 'Scripts', 'python.exe'),
    'python',
    'py'
  ].filter(Boolean);
  return candidates.find((candidate) => candidate === 'python' || candidate === 'py' || fs.existsSync(candidate)) || 'python';
}
let mainWindow;
let predictionAppProcess = null;
let simulationAppProcess = null;
let simulationViteProcess = null;
let simulationApiProcess = null;
let activeProjectId = null;
let lastPrediction = null;
let lastSimulation = null;
let serviceLogs = [];

function bundledRuntimeDir(...parts) {
  return path.join(app.getPath('home'), '.cache', 'codex-runtimes', 'codex-primary-runtime', 'dependencies', ...parts);
}

function resolvePackageRunner() {
  const candidates = [
    process.env.AI_MATERIALS_PNPM,
    bundledRuntimeDir('bin', 'pnpm.cmd'),
    process.env.AI_MATERIALS_NPM,
    'C:\\Program Files\\nodejs\\npm.cmd',
    path.join(process.env.APPDATA || '', 'npm', 'npm.cmd'),
    'pnpm',
    'npm'
  ].filter(Boolean);
  return candidates.find((candidate) => candidate === 'npm' || candidate === 'pnpm' || fs.existsSync(candidate)) || 'npm';
}

function runtimePath(extra = []) {
  return [
    path.join(simulationRepoDir, 'node_modules', '.bin'),
    path.join(rootDir, 'node_modules', '.bin'),
    bundledRuntimeDir('node', 'bin'),
    bundledRuntimeDir('bin'),
    ...extra,
    process.env.PATH
  ].filter(Boolean).join(path.delimiter);
}

function resolveSimulationBin(name) {
  const suffix = process.platform === 'win32' ? '.cmd' : '';
  return path.join(simulationRepoDir, 'node_modules', '.bin', `${name}${suffix}`);
}

function requestOk(url) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const req = http.get({ hostname: parsed.hostname, port: parsed.port, path: parsed.pathname }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on('error', reject);
    req.setTimeout(3000, () => req.destroy(new Error(`Timeout: ${url}`)));
  });
}

async function waitForViteServer() {
  let lastError;
  for (let i = 0; i < 80; i += 1) {
    try {
      if (await requestOk('http://127.0.0.1:5173')) return true;
    } catch (err) {
      lastError = err;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Vite dev server did not become ready: ${lastError ? lastError.message : 'unknown error'}`);
}

function waitForProcessBoot(child, label, timeout = 2500) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      settled = true;
      resolve({ running: true });
    }, timeout);
    child.once('exit', (code) => {
      if (settled) return;
      clearTimeout(timer);
      settled = true;
      reject(new Error(`${label} failed to start. Exit code: ${code}. Check service log in the dashboard.`));
    });
    child.once('error', (err) => {
      if (settled) return;
      clearTimeout(timer);
      settled = true;
      reject(err);
    });
  });
}

function logService(source, message) {
  const line = `[${new Date().toISOString()}] ${source}: ${String(message).trim()}`;
  serviceLogs = [line, ...serviceLogs].slice(0, 80);
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('service-log', line);
}

function repoExists(dir, requiredFile) {
  return fs.existsSync(dir) && fs.existsSync(path.join(dir, requiredFile));
}

function isProcessRunning(child) {
  return !!child && !child.killed && child.exitCode === null && child.signalCode === null;
}

function clearProcessRef(label, child) {
  if (label === 'prediction-app' && predictionAppProcess === child) predictionAppProcess = null;
  if (label === 'simulation-app' && simulationAppProcess === child) simulationAppProcess = null;
  if (label === 'simulation-vite' && simulationViteProcess === child) simulationViteProcess = null;
  if (label === 'simulation-api' && simulationApiProcess === child) simulationApiProcess = null;
}
function spawnManaged(label, command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: { ...process.env, ...(options.env || {}) },
    shell: process.platform === 'win32',
    windowsHide: !!options.hidden,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  child.stdout.on('data', (data) => logService(label, data));
  child.stderr.on('data', (data) => logService(label, data));
  child.on('error', (err) => logService(label, err.message));
  child.on('exit', (code, signal) => {
    const suffix = signal ? ` signal ${signal}` : '';
    logService(label, `exited with code ${code}${suffix}`);
    clearProcessRef(label, child);
  });
  return child;
}

function requestJson(method, url, body) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const data = body ? Buffer.from(JSON.stringify(body)) : null;
    const req = http.request({
      method,
      hostname: parsed.hostname,
      port: parsed.port,
      path: `${parsed.pathname}${parsed.search}`,
      headers: data ? { 'Content-Type': 'application/json', 'Content-Length': data.length } : {}
    }, (res) => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { raw += chunk; });
      res.on('end', () => {
        let json = {};
        try { json = raw ? JSON.parse(raw) : {}; } catch (err) { return reject(new Error(`Invalid JSON from ${url}: ${raw}`)); }
        if (res.statusCode >= 400) return reject(new Error(json.error || `HTTP ${res.statusCode}`));
        resolve(json);
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => req.destroy(new Error(`Timeout: ${url}`)));
    if (data) req.write(data);
    req.end();
  });
}

async function waitForSimulationApi() {
  const started = Date.now();
  let lastError;
  while (Date.now() - started < 45000) {
    try { return await requestJson('GET', `${simulationApiBase}/health`); }
    catch (err) { lastError = err; await new Promise((resolve) => setTimeout(resolve, 700)); }
  }
  throw new Error(`Simulation API did not become ready: ${lastError ? lastError.message : 'unknown error'}`);
}

async function ensureSimulationApi() {
  try { return await requestJson('GET', `${simulationApiBase}/health`); } catch (_) {}
  if (!repoExists(simulationRepoDir, path.join('backend', 'simulation_server.py'))) {
    throw new Error(`Simulation repository not found: ${simulationRepoDir}`);
  }
  if (!repoExists(predictionRepoDir, path.join('models', 'pretrained_material_model.pkl'))) {
    throw new Error(`Prediction model repository not found: ${predictionRepoDir}`);
  }
  if (!isProcessRunning(simulationApiProcess)) {
    simulationApiProcess = spawnManaged('simulation-api', resolvePythonExe(), ['backend/simulation_server.py'], {
      cwd: simulationRepoDir,
      env: pythonEnv({ AI_MATERIALS_PLATFORM_DIR: predictionRepoDir }),
      hidden: true
    });
  }
  return waitForSimulationApi();
}

function projectId() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `Project_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

async function listProjects() {
  await fsp.mkdir(projectsDir, { recursive: true });
  const entries = await fsp.readdir(projectsDir, { withFileTypes: true });
  const rows = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const file = path.join(projectsDir, entry.name, 'project.json');
    try {
      const meta = JSON.parse(await fsp.readFile(file, 'utf8'));
      rows.push(meta);
    } catch (_) {}
  }
  rows.sort((a, b) => String(b.updatedAt || b.createdAt).localeCompare(String(a.updatedAt || a.createdAt)));
  return rows;
}

function normalizePrediction(payload) {
  const p = payload.platformPrediction || payload.prediction || payload;
  return {
    yieldStrengthMpa: Number(p.yieldStressMpa || p.yieldStrengthMpa || 0),
    utsMpa: Number(p.utsMpa || p.ultimateTensileStrengthMpa || 0),
    elongationPercent: Number(p.elongationPercent || 0),
    areaReductionPercent: Number(p.areaReductionPercent || 0),
    uncertainty: p.uncertainty || {},
    raw: payload
  };
}

function buildStressStrainCurve(prediction) {
  const ys = Math.max(prediction.yieldStrengthMpa, 1);
  const uts = Math.max(prediction.utsMpa, ys);
  const elongation = Math.max(prediction.elongationPercent, 1);
  const points = [];
  const elasticStrain = Math.min(0.012, Math.max(0.004, ys / 210000));
  for (let i = 0; i <= 60; i += 1) {
    const ratio = i / 60;
    const strain = (elongation / 100) * ratio;
    let stress;
    if (strain <= elasticStrain) stress = ys * (strain / elasticStrain);
    else if (ratio <= 0.75) stress = ys + (uts - ys) * ((ratio - elasticStrain / (elongation / 100)) / 0.75);
    else stress = uts - (uts * 0.16) * ((ratio - 0.75) / 0.25);
    points.push({ strain: Number(strain.toFixed(5)), stressMpa: Number(Math.max(0, stress).toFixed(2)) });
  }
  return points;
}

function defaultInputs() {
  return {
    composition: { Fe: 63.5, Cr: 19.7, Ni: 13.5, Mo: 2.1, Mn: 0.8, Si: 0.4 },
    densityScale: 0.62,
    process: {
      Solution_treatment_temperature: 1323,
      'Solution_treatment_time(s)': 3600,
      'Water_Quenched_after_s.t.': 1,
      'Air_Quenched_after_s.t.': 0,
      'Temperature (K)': 293,
      'Grains mm-2': 12000
    },
    testType: 'strength',
    scale: 1
  };
}
async function writeProjectFiles(status = '진행중') {
  if (!activeProjectId) activeProjectId = projectId();
  const dir = path.join(projectsDir, activeProjectId);
  await fsp.mkdir(dir, { recursive: true });
  const now = new Date().toISOString();
  let previous = {};
  try { previous = JSON.parse(await fsp.readFile(path.join(dir, 'project.json'), 'utf8')); } catch (_) {}
  const meta = {
    id: activeProjectId,
    name: activeProjectId,
    createdAt: previous.createdAt || now,
    updatedAt: now,
    status,
    hasPrediction: !!lastPrediction,
    hasSimulation: !!lastSimulation
  };
  await fsp.writeFile(path.join(dir, 'project.json'), JSON.stringify(meta, null, 2), 'utf8');
  if (lastPrediction) {
    await fsp.writeFile(path.join(dir, 'prediction-result.json'), JSON.stringify(lastPrediction, null, 2), 'utf8');
    await fsp.writeFile(path.join(dir, 'stress-strain-curve.json'), JSON.stringify(lastPrediction.stressStrainCurve || [], null, 2), 'utf8');
  }
  if (lastSimulation) await fsp.writeFile(path.join(dir, 'simulation-result.json'), JSON.stringify(lastSimulation, null, 2), 'utf8');
  return meta;
}

async function runPrediction(payload) {
  const inputs = { ...defaultInputs(), ...(payload || {}) };
  await ensureSimulationApi();
  const platformStatus = await requestJson('GET', `${simulationApiBase}/platform/status`);
  if (!platformStatus.available) throw new Error(platformStatus.error || 'Prediction model is not available');
  const apiResult = await requestJson('POST', `${simulationApiBase}/platform/predict`, inputs);
  const prediction = normalizePrediction(apiResult);
  prediction.input = inputs;
  prediction.model = platformStatus;
  prediction.stressStrainCurve = buildStressStrainCurve(prediction);
  prediction.completedAt = new Date().toISOString();
  lastPrediction = prediction;
  await writeProjectFiles('예측 완료');
  return prediction;
}

async function runSimulation(payload) {
  const inputs = { ...defaultInputs(), ...(payload || {}) };
  if (!lastPrediction) throw new Error('Prediction result is required before simulation. Run material prediction first.');
  await ensureSimulationApi();
  const simulation = await requestJson('POST', `${simulationApiBase}/simulate`, inputs);
  lastSimulation = {
    input: inputs,
    transferredPrediction: {
      yieldStrengthMpa: lastPrediction.yieldStrengthMpa,
      utsMpa: lastPrediction.utsMpa,
      elongationPercent: lastPrediction.elongationPercent,
      areaReductionPercent: lastPrediction.areaReductionPercent
    },
    result: simulation,
    completedAt: new Date().toISOString()
  };
  await writeProjectFiles('완료');
  return lastSimulation;
}

ipcMain.handle('integration:getStatus', async () => {
  let api = { available: false };
  let platform = { available: false };
  try { api = await requestJson('GET', `${simulationApiBase}/health`); } catch (_) {}
  try { platform = await requestJson('GET', `${simulationApiBase}/platform/status`); } catch (_) {}
  return {
    repositories: {
      prediction: { path: predictionRepoDir, available: repoExists(predictionRepoDir, 'main.py') },
      simulation: { path: simulationRepoDir, available: repoExists(simulationRepoDir, 'package.json') }
    },
    services: {
      predictionAppRunning: isProcessRunning(predictionAppProcess),
      simulationAppRunning: isProcessRunning(simulationAppProcess),
      simulationApi: api,
      platform
    },
    projects: await listProjects(),
    activeProjectId,
    lastPrediction,
    lastSimulation,
    logs: serviceLogs
  };
});

ipcMain.handle('integration:newProject', async () => {
  activeProjectId = projectId();
  lastPrediction = null;
  lastSimulation = null;
  return writeProjectFiles('새 프로젝트');
});

ipcMain.handle('integration:saveProject', async () => writeProjectFiles(lastSimulation ? '완료' : lastPrediction ? '예측 완료' : '진행중'));

ipcMain.handle('integration:loadProject', async (_event, id) => {
  if (!id || !/^[\w.-]+$/.test(id)) throw new Error('Invalid project id');
  const dir = path.join(projectsDir, id);
  const meta = JSON.parse(await fsp.readFile(path.join(dir, 'project.json'), 'utf8'));
  activeProjectId = id;
  try { lastPrediction = JSON.parse(await fsp.readFile(path.join(dir, 'prediction-result.json'), 'utf8')); } catch (_) { lastPrediction = null; }
  try { lastSimulation = JSON.parse(await fsp.readFile(path.join(dir, 'simulation-result.json'), 'utf8')); } catch (_) { lastSimulation = null; }
  return { meta, prediction: lastPrediction, simulation: lastSimulation };
});

ipcMain.handle('integration:openProjectFolder', async (_event, id) => {
  const target = id ? path.join(projectsDir, id) : projectsDir;
  await fsp.mkdir(target, { recursive: true });
  return shell.openPath(target);
});

ipcMain.handle('integration:startPredictionApp', async () => {
  if (!repoExists(predictionRepoDir, 'main.py')) throw new Error(`Prediction repository not found: ${predictionRepoDir}`);
  if (!isProcessRunning(predictionAppProcess)) {
    const pythonExe = resolvePythonExe();
    logService('prediction-app', `starting with ${pythonExe}`);
    predictionAppProcess = spawnManaged('prediction-app', pythonExe, ['main.py'], {
      cwd: predictionRepoDir,
      env: pythonEnv(),
      hidden: false
    });
    await waitForProcessBoot(predictionAppProcess, 'Prediction app');
  }
  return { started: true, path: predictionRepoDir };
});

ipcMain.handle('integration:startSimulationApp', async () => {
  if (!repoExists(simulationRepoDir, 'package.json')) throw new Error(`Simulation repository not found: ${simulationRepoDir}`);
  const viteCmd = resolveSimulationBin('vite');
  const electronCmd = resolveSimulationBin('electron');
  if (!fs.existsSync(viteCmd)) throw new Error(`Vite executable not found: ${viteCmd}`);
  if (!fs.existsSync(electronCmd)) throw new Error(`Electron executable not found: ${electronCmd}`);

  if (!isProcessRunning(simulationViteProcess)) {
    logService('simulation-vite', `starting ${viteCmd}`);
    simulationViteProcess = spawnManaged('simulation-vite', viteCmd, ['--host', '127.0.0.1', '--port', '5173'], {
      cwd: simulationRepoDir,
      env: pythonEnv({ AI_MATERIALS_PLATFORM_DIR: predictionRepoDir }),
      hidden: false
    });
  }

  await waitForViteServer();

  if (!isProcessRunning(simulationAppProcess)) {
    const env = pythonEnv({
      AI_MATERIALS_PLATFORM_DIR: predictionRepoDir,
      VITE_DEV_SERVER_URL: 'http://127.0.0.1:5173'
    });
    delete env.ELECTRON_RUN_AS_NODE;
    logService('simulation-app', `starting ${electronCmd}`);
    simulationAppProcess = spawnManaged('simulation-app', electronCmd, [simulationRepoDir], {
      cwd: simulationRepoDir,
      env,
      hidden: false
    });
    await waitForProcessBoot(simulationAppProcess, 'Simulation app');
  }
  return { started: true, path: simulationRepoDir };
});

ipcMain.handle('integration:runPrediction', async (_event, payload) => runPrediction(payload));
ipcMain.handle('integration:runSimulation', async (_event, payload) => runSimulation(payload));
ipcMain.handle('integration:runWorkflow', async (_event, payload) => {
  const prediction = await runPrediction(payload);
  const simulation = await runSimulation(payload);
  return { prediction, simulation };
});

ipcMain.handle('integration:stopServices', async () => {
  for (const child of [predictionAppProcess, simulationAppProcess, simulationViteProcess, simulationApiProcess]) {
    if (isProcessRunning(child)) child.kill();
  }
  predictionAppProcess = null;
  simulationAppProcess = null;
  simulationViteProcess = null;
  simulationApiProcess = null;
  return { stopped: true };
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1540,
    height: 980,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#f8fafc',
    title: 'Material Property Prediction & Simulation System',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow.loadFile(path.join(rootDir, 'renderer', 'index.html'));
}

app.whenReady().then(async () => {
  await fsp.mkdir(projectsDir, { recursive: true });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (isProcessRunning(simulationApiProcess)) simulationApiProcess.kill();
});
