const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('integrationApi', {
  getStatus: () => ipcRenderer.invoke('integration:getStatus'),
  newProject: () => ipcRenderer.invoke('integration:newProject'),
  saveProject: () => ipcRenderer.invoke('integration:saveProject'),
  loadProject: (projectId) => ipcRenderer.invoke('integration:loadProject', projectId),
  openProjectFolder: (projectId) => ipcRenderer.invoke('integration:openProjectFolder', projectId),
  startPredictionApp: () => ipcRenderer.invoke('integration:startPredictionApp'),
  startSimulationApp: () => ipcRenderer.invoke('integration:startSimulationApp'),
  runPrediction: (payload) => ipcRenderer.invoke('integration:runPrediction', payload),
  runSimulation: (payload) => ipcRenderer.invoke('integration:runSimulation', payload),
  runWorkflow: (payload) => ipcRenderer.invoke('integration:runWorkflow', payload),
  stopServices: () => ipcRenderer.invoke('integration:stopServices')
});
