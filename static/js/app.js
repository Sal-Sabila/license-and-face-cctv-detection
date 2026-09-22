/**
 * PlateVision - AI CCTV Monitoring & Detection System
 * Frontend Application Controller
 */

// Helper pembersih string untuk pencegahan XSS
const esc = value => String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

// Helper: normalize image URL paths from DB to browser-friendly URLs
function buildCaptureUrl(path) {
    if (!path) return null;
    // Convert backslashes to forward slashes and trim spaces
    let clean = String(path).trim().replace(/\\/g, '/');
    // Remove leading slashes
    clean = clean.replace(/^\/+/, '');
    // Ensure it starts with "static/"
    if (!clean.startsWith('static/')) {
        clean = 'static/' + clean;
    }
    // Return as absolute URL path
    return '/' + clean;
}

function getDateRange(period) {
    if (period === 'all') return { start_date: '', end_date: '' };

    const end = new Date();
    const start = new Date(end);
    const days = { today: 1, '2d': 2, '7d': 7, '30d': 30 }[period] || 1;
    start.setDate(start.getDate() - days + 1);
    const formatDate = date => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    };

    return { start_date: formatDate(start), end_date: formatDate(end) };
}

function getStateDateRange(state) {
    if (state.start_date || state.end_date) {
        return { start_date: state.start_date, end_date: state.end_date };
    }
    return getDateRange(state.period);
}

function applyCustomDateRange(state, startInputId, endInputId) {
    const startDate = document.getElementById(startInputId)?.value || '';
    const endDate = document.getElementById(endInputId)?.value || '';
    const today = new Date();
    const todayValue = [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, '0'),
        String(today.getDate()).padStart(2, '0')
    ].join('-');

    if (!startDate && !endDate) {
        state.start_date = '';
        state.end_date = '';
        state.period = 'today';
        return true;
    }
    if ((startDate && endDate) && startDate > endDate) {
        showNotification('Tanggal mulai tidak boleh lebih besar dari tanggal akhir.', 'warning');
        return false;
    }
    if (startDate > todayValue || endDate > todayValue) {
        showNotification('Tanggal tidak boleh melebihi tanggal hari ini.', 'warning');
        return false;
    }

    state.start_date = startDate;
    state.end_date = endDate;
    state.period = 'custom';
    return true;
}

function setDateFilterLimits() {
    const today = new Date();
    const todayValue = [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, '0'),
        String(today.getDate()).padStart(2, '0')
    ].join('-');

    ['detStartDate', 'detEndDate', 'plateStartDate', 'plateEndDate', 'recapStart', 'recapEnd'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.max = todayValue;
    });
}

// Fetch JSON helper
async function json(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok && response.status >= 500) {
        throw new Error(`Server error (${response.status})`);
    }
    return response.json();
}

function showNotification(message, type = 'info', title = '') {
    const toastEl = document.getElementById('appToast');
    if (!toastEl || !window.bootstrap) return;

    const presets = {
        success: { title: 'Berhasil', icon: 'bi-check-circle-fill text-success' },
        danger: { title: 'Terjadi Kesalahan', icon: 'bi-x-circle-fill text-danger' },
        warning: { title: 'Perhatian', icon: 'bi-exclamation-triangle-fill text-warning' },
        info: { title: 'Informasi', icon: 'bi-info-circle-fill text-primary' }
    };
    const preset = presets[type] || presets.info;
    const icon = document.getElementById('toastIcon');
    const titleEl = document.getElementById('toastTitle');
    const messageEl = document.getElementById('toastMessage');
    if (icon) icon.className = `bi ${preset.icon} me-2`;
    if (titleEl) titleEl.textContent = title || preset.title;
    if (messageEl) messageEl.textContent = message || '';
    bootstrap.Toast.getOrCreateInstance(toastEl, { delay: 4200 }).show();
}
window.showNotification = showNotification;

function askConfirmation(message, title = 'Konfirmasi') {
    return new Promise(resolve => {
        const modalEl = document.getElementById('appConfirmModal');
        const acceptButton = document.getElementById('confirmAccept');
        if (!modalEl || !acceptButton || !window.bootstrap) {
            resolve(window.confirm(message));
            return;
        }

        document.getElementById('confirmTitle').textContent = title;
        document.getElementById('confirmMessage').textContent = message;
        const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        let settled = false;
        const finish = value => {
            if (settled) return;
            settled = true;
            resolve(value);
        };
        const onAccept = () => {
            finish(true);
            modal.hide();
        };
        const onHidden = () => {
            finish(false);
            acceptButton.removeEventListener('click', onAccept);
            modalEl.removeEventListener('hidden.bs.modal', onHidden);
        };
        acceptButton.addEventListener('click', onAccept);
        modalEl.addEventListener('hidden.bs.modal', onHidden);
        modal.show();
    });
}
window.askConfirmation = askConfirmation;

// ============================================================
// MODAL GLOBAL: PRATINJAU GAMBAR & DETAIL DETEKSI
// ============================================================

function openImageModal(imgSrc, title = 'Detail Foto Tangkapan CCTV', meta = '', details = '', objType = null) {
    const modalEl = document.getElementById('imagePreviewModal');
    if (!modalEl) return;

    const titleEl = document.getElementById('imagePreviewTitle');
    const metaEl = document.getElementById('imagePreviewMeta');
    const srcEl = document.getElementById('imagePreviewSrc');
    const detailsEl = document.getElementById('imagePreviewDetails');

    // Determine dynamic title based on object type
    if (objType) {
        title = objType === 'vehicle' ? 'Foto Kendaraan' : 'Foto Orang';
    }

    if (titleEl) titleEl.textContent = title;
    if (metaEl) metaEl.textContent = meta;
    if (detailsEl) detailsEl.innerHTML = details;

// Compute image URL from the provided path (already prioritized)
    const imageUrl = buildCaptureUrl(imgSrc);
    console.log('[IMAGE SRC]', imgSrc);
    console.log('[OBJECT TYPE]', objType);
    console.log('[IMAGE URL]', imageUrl);

    if (srcEl) {
        if (imageUrl) {
            srcEl.src = imageUrl;
            srcEl.style.display = 'block';
        } else {
            srcEl.src = 'https://placehold.co/600x400/1e293b/94a3b8?text=Foto+Tidak+Tersedia';
            srcEl.style.display = 'block';
            if (detailsEl) detailsEl.innerHTML = 'Foto Tidak Tersedia';
        }
        // Error handling for load failure
        srcEl.onerror = function () {
            srcEl.style.display = 'none';
            if (detailsEl) detailsEl.innerHTML = 'Foto tidak dapat dimuat.';
            console.error('Failed to load image:', imageUrl);
        };
    }

    bootstrap.Modal.getOrCreateInstance(modalEl).show();
}
window.openImageModal = openImageModal;


// ============================================================
// MODUL: MONITORING CCTV
// ============================================================

async function loadCameras() {
    try {
        const result = await json('/api/cameras');
        const cameras = result.data || [];
        const count = document.getElementById('cameraCountTitle');
        if (count) count.textContent = cameras.length;

        const select = document.getElementById('cameraSelect');
        if (select) {
            select.innerHTML = cameras.map(item => `<option value="${item.id}">${esc(item.name)}</option>`).join('') || '<option value="">Belum ada kamera</option>';
        }

        const table = document.getElementById('cameraTable');
        if (table) {
            table.innerHTML = cameras.map(camera => `
                <tr>
                    <td><strong>${esc(camera.name)}</strong></td>
                    <td class="camera-url">${esc(camera.url)}</td>
                    <td><span class="status ${camera.active ? 'success' : 'warning'}">${camera.active ? 'Aktif' : 'Nonaktif'}</span></td>
                    <td>
                        <div class="table-actions">
                            <button class="action-toggle ${camera.active ? 'deactivate' : 'activate'}" data-action="toggle" data-id="${camera.id}">
                                ${camera.active ? 'Nonaktifkan' : 'Aktifkan'}
                            </button>
                            <button class="action-edit" data-action="edit" data-id="${camera.id}">Edit</button>
                            <button class="action-delete" data-action="delete" data-id="${camera.id}">Hapus</button>
                        </div>
                    </td>
                </tr>
            `).join('') || '<tr><td colspan="4" class="text-center text-muted py-5">Belum ada kamera.</td></tr>';
        }
        return cameras;
    } catch (err) {
        console.error('Error loadCameras:', err);
        return [];
    }
}

function openCameraModal(camera = null) {
    const form = document.getElementById('cameraForm');
    if (!form) return;
    form.reset();
    document.getElementById('cameraId').value = camera?.id || '';
    if (camera) {
        document.getElementById('cameraNameInput').value = camera.name || '';
        document.getElementById('cameraUrl').value = camera.url || '';
        document.getElementById('cameraActive').checked = camera.active;
    }
    bootstrap.Modal.getOrCreateInstance(document.getElementById('cameraModal')).show();
}

async function exportCameras(format) {
    try {
        // Excel (.xlsx) dan PDF dibuat di backend (openpyxl / ReportLab) agar
        // hasilnya rapi dan konsisten dengan data kamera di database.
        // Backend tetap menghasilkan file valid walau daftar kamera kosong.
        if (format === 'excel') {
            window.location.href = '/api/export/cameras';
            return;
        }
        if (format === 'pdf') {
            window.location.href = '/api/export/cameras/pdf';
            return;
        }

        const result = await json('/api/cameras');
        const cameras = result.data || [];
        if (!cameras.length) {
            showNotification('Belum ada kamera untuk diekspor.', 'info');
            return;
        }

        if (format === 'xspf') {
            let xspf = `<?xml version="1.0" encoding="UTF-8"?>\n<playlist version="1" xmlns="http://xspf.org/ns/0/">\n  <title>CCTV Playlist</title>\n  <trackList>\n`;
            cameras.forEach(cam => {
                xspf += `    <track>\n      <title>${esc(cam.name)}</title>\n      <location>${esc(cam.url)}</location>\n    </track>\n`;
            });
            xspf += `  </trackList>\n</playlist>`;

            const blob = new Blob([xspf], { type: 'application/xspf+xml;charset=utf-8' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'cctv_playlist.xspf';
            a.click();
            URL.revokeObjectURL(a.href);
        }
    } catch (err) {
        console.error('Gagal mengekspor kamera:', err);
        showNotification('Gagal mengekspor kamera.', 'danger');
    }
}
window.exportCameras = exportCameras;

async function initMonitoring() {
    let cameras = await loadCameras();
    document.getElementById('cameraForm')?.addEventListener('submit', async event => {
        event.preventDefault();
        const id = document.getElementById('cameraId').value;
        const data = {
            name: document.getElementById('cameraNameInput').value,
            url: document.getElementById('cameraUrl').value,
            active: document.getElementById('cameraActive').checked
        };
        await json(id ? `/api/cameras/${id}` : '/api/cameras', {
            method: id ? 'PUT' : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        bootstrap.Modal.getInstance(document.getElementById('cameraModal')).hide();
        cameras = await loadCameras();
    });

    document.getElementById('cameraTable')?.addEventListener('click', async event => {
        const button = event.target.closest('button');
        if (!button) return;
        const id = Number(button.dataset.id);
        const camera = cameras.find(item => item.id === id);
        if (button.dataset.action === 'edit') openCameraModal(camera);
        if (button.dataset.action === 'delete' && await askConfirmation('Hapus kamera ini?', 'Hapus Kamera')) {
            await json(`/api/cameras/${id}`, { method: 'DELETE' });
            cameras = await loadCameras();
            showNotification('Kamera berhasil dihapus.', 'success');
        }
        if (button.dataset.action === 'toggle') {
            await json(`/api/cameras/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active: !camera.active })
            });
            cameras = await loadCameras();
        }
    });
}


// ============================================================
// MODUL: DASHBOARD UTAMA
// ============================================================

async function initDashboard() {
    async function updateDashboardMetrics() {
        try {
            const summaryRes = await json('/api/analytics?period=today');
            const data = summaryRes.data?.summary || {};

            const elTotalVeh = document.getElementById('totalVehicle');
            const elPlateRead = document.getElementById('plateRead');
            const elNeedCheck = document.getElementById('needCheck');
            const elCamStatus = document.getElementById('cameraStatus');

            if (elTotalVeh) elTotalVeh.textContent = data.vehicles || 0;
            if (elPlateRead) elPlateRead.textContent = data.plates || 0;
            if (elNeedCheck) elNeedCheck.textContent = data.people || 0;
            if (elCamStatus) elCamStatus.textContent = `${data.active_cameras || 0} / ${data.total_cameras || 0}`;
            const unique = document.getElementById('uniquePlateCount');
            const people = document.getElementById('peopleTotal');
            const vehicleLabel = document.getElementById('vehicleFlowLabel');
            const peopleLabel = document.getElementById('peopleFlowLabel');
            const vehicleTotal = document.getElementById('vehicleFlowTotal');
            const peopleTotal = document.getElementById('peopleFlowTotal');
            const vehicleDetail = document.getElementById('vehicleFlowDetail');
            const peopleDetail = document.getElementById('peopleFlowDetail');
            if (unique) unique.textContent = data.unique_plates || 0;
            if (people) people.textContent = data.people || 0;
            if (vehicleLabel) vehicleLabel.textContent = `Masuk ${data.vehicle_entry || 0} · Keluar ${data.vehicle_exit || 0}`;
            if (peopleLabel) peopleLabel.textContent = `Masuk ${data.people_entry || 0} · Keluar ${data.people_exit || 0}`;
            if (vehicleTotal) vehicleTotal.textContent = data.vehicles || 0;
            if (peopleTotal) peopleTotal.textContent = data.people || 0;
            if (vehicleDetail) vehicleDetail.textContent = `Masuk ${data.vehicle_entry || 0} · Keluar ${data.vehicle_exit || 0}`;
            if (peopleDetail) peopleDetail.textContent = `Masuk ${data.people_entry || 0} · Keluar ${data.people_exit || 0}`;
            const insight = document.getElementById('dashboardInsights');
            if (insight) {
                const insightData = summaryRes.data?.insights || [];
                insight.innerHTML = insightData.map(item => `<li>${esc(item)}</li>`).join('') || '<li>Belum ada data cukup untuk insight.</li>';
            }
        } catch (e) {
            console.error('Update dashboard metrics error:', e);
        }
    }

    const camRes = await json('/api/cameras').catch(() => ({ data: [] }));
    const cameras = camRes.data || [];

    const select = document.getElementById('cameraSelect');
    if (select) {
        select.innerHTML = cameras.map(item => `<option value="${item.id}" ${item.active ? 'selected' : ''}>${esc(item.name)}</option>`).join('') || '<option value="">Belum ada kamera</option>';
    }

    // Video Stream Control
    const streamImg = document.getElementById('liveStreamImg');
    const placeholder = document.getElementById('videoPlaceholder');
    const cameraNameEl = document.getElementById('cameraName');
    const startBtn = document.getElementById('startStreamBtn');
    const stopBtn = document.getElementById('stopStreamBtn');
    const bboxToggle = document.getElementById('bboxToggle');
    const detectionMode = document.getElementById('detectionMode');
    const videoFileLabel = document.getElementById('videoFileLabel');
    const videoFileInput = document.getElementById('videoFileInput');
    const processedVideo = document.getElementById('processedVideo');
    let videoJobPoller = null;

    function getStreamUrl(camId) {
        const showBbox = bboxToggle ? (bboxToggle.checked ? 1 : 0) : 1;
        return `/api/video_feed/${camId}?bbox=${showBbox}`;
    }

    function startCameraStream(camId) {
        if (!streamImg) return;
        const cam = cameras.find(c => String(c.id) === String(camId));
        if (cam) {
            if (videoJobPoller) clearInterval(videoJobPoller);
            if (processedVideo) {
                processedVideo.pause();
                processedVideo.removeAttribute('src');
                processedVideo.style.display = 'none';
            }
            streamImg.src = getStreamUrl(cam.id);
            streamImg.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
            if (cameraNameEl) cameraNameEl.textContent = cam.name;
        }
    }

    function stopCameraStream() {
        if (!streamImg) return;
        streamImg.src = '';
        streamImg.style.display = 'none';
        if (videoJobPoller) clearInterval(videoJobPoller);
        if (processedVideo) {
            processedVideo.pause();
            processedVideo.removeAttribute('src');
            processedVideo.style.display = 'none';
        }
        if (placeholder) placeholder.style.display = 'flex';
        if (cameraNameEl) cameraNameEl.textContent = 'Stream Dihentikan';
    }

    function updateDetectionMode() {
        const videoMode = detectionMode?.value === 'video';
        if (videoFileLabel) videoFileLabel.style.display = videoMode ? 'inline-flex' : 'none';
        if (streamImg) streamImg.style.display = videoMode ? (videoJobPoller ? 'block' : 'none') : streamImg.src ? 'block' : 'none';
        if (processedVideo && !videoMode) processedVideo.style.display = 'none';
    }

    async function startVideoJob(file, camId) {
        if (!file || !camId) return;
        stopCameraStream();
        if (cameraNameEl) cameraNameEl.textContent = `Memproses ${file.name}`;

        const formData = new FormData();
        formData.append('video', file);
        formData.append('camera_id', camId);

        try {
            const response = await fetch('/api/video_jobs', { method: 'POST', body: formData });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.message || 'Upload video gagal');
            localStorage.setItem('platevision.videoJobId', result.job_id);
            monitorVideoJob(result.job_id);
        } catch (error) {
            if (cameraNameEl) cameraNameEl.textContent = error.message;
            if (placeholder) placeholder.style.display = 'flex';
        }
    }

    function monitorVideoJob(jobId) {
        if (!jobId) return;
        if (videoJobPoller) clearInterval(videoJobPoller);
        if (streamImg) {
            streamImg.src = `/api/video_jobs/${jobId}/feed`;
            streamImg.style.display = 'block';
        }
        if (placeholder) placeholder.style.display = 'none';
        if (processedVideo) processedVideo.style.display = 'none';

        const checkJob = async () => {
            try {
                const statusResult = await json(`/api/video_jobs/${jobId}`);
                const job = statusResult.data;
                if (job.status === 'completed') {
                    clearInterval(videoJobPoller);
                    if (streamImg) {
                        streamImg.src = '';
                        streamImg.style.display = 'none';
                    }
                    if (processedVideo) {
                        processedVideo.src = `${job.output_url}?t=${Date.now()}`;
                        processedVideo.style.display = 'block';
                        processedVideo.load();
                        processedVideo.play().catch(() => {});
                    }
                    if (placeholder) placeholder.style.display = 'none';
                    if (cameraNameEl) cameraNameEl.textContent = 'Hasil Deteksi Video';
                } else if (job.status === 'failed') {
                    clearInterval(videoJobPoller);
                    localStorage.removeItem('platevision.videoJobId');
                    if (cameraNameEl) cameraNameEl.textContent = `Gagal: ${job.error || 'proses video'}`;
                } else if (cameraNameEl) {
                    cameraNameEl.textContent = `Memproses video (${job.progress || 0}%)`;
                }
            } catch (error) {
                clearInterval(videoJobPoller);
                console.error('Status video error:', error);
            }
        };

        checkJob();
        videoJobPoller = setInterval(checkJob, 1000);
    }

    const savedMode = localStorage.getItem('platevision.mode') || 'stream';
    const savedCameraId = localStorage.getItem('platevision.cameraId');
    const savedVideoJobId = localStorage.getItem('platevision.videoJobId');
    if (detectionMode) detectionMode.value = savedMode;
    if (savedCameraId && cameras.some(camera => String(camera.id) === savedCameraId)) {
        select.value = savedCameraId;
    }

    const activeCam = cameras.find(item => String(item.id) === String(select?.value)) || cameras.find(item => item.active) || cameras[0];
    if (activeCam && select) select.value = activeCam.id;

    if (savedMode === 'video' && savedVideoJobId) {
        monitorVideoJob(savedVideoJobId);
    } else if (activeCam && activeCam.active) {
        startCameraStream(activeCam.id);
    } else {
        stopCameraStream();
    }

    if (select) {
        select.addEventListener('change', () => {
            localStorage.setItem('platevision.cameraId', select.value);
            if (detectionMode?.value === 'video') return;
            if (select.value) startCameraStream(select.value);
        });
    }

    detectionMode?.addEventListener('change', () => {
        localStorage.setItem('platevision.mode', detectionMode.value);
        if (detectionMode.value === 'video') stopCameraStream();
        updateDetectionMode();
    });

    videoFileInput?.addEventListener('change', () => {
        const selectedId = select?.value || activeCam?.id;
        const file = videoFileInput.files?.[0];
        if (file && selectedId) startVideoJob(file, selectedId);
    });

    if (bboxToggle) {
        bboxToggle.addEventListener('change', () => {
            const selectedId = select?.value || activeCam?.id;
            if (selectedId && detectionMode?.value !== 'video' && streamImg && streamImg.style.display !== 'none') {
                startCameraStream(selectedId);
            }
        });
    }

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            const selectedId = select?.value || activeCam?.id;
            if (!selectedId) return;
            if (detectionMode?.value === 'video') {
                startVideoJob(videoFileInput?.files?.[0], selectedId);
            } else {
                startCameraStream(selectedId);
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            localStorage.removeItem('platevision.videoJobId');
            stopCameraStream();
        });
    }

    updateDetectionMode();

    async function refreshRecentList() {
        try {
            const detRes = await json('/api/detections?limit=6');
            const items = detRes.data || [];
            const recent = document.getElementById('recentList');

            if (recent) {
                if (items.length === 0) {
                    recent.innerHTML = '<div class="empty-state py-4 text-muted text-center">Belum ada deteksi</div>';
                    return;
                }

                recent.innerHTML = items.map(item => {
                    const isVehicle = item.object_type === 'vehicle' || item.type === 'vehicle' || item.type === 'vehicle_with_plate' || item.type === 'plate';
                    const isPlate = item.type === 'vehicle_with_plate' || Boolean(item.plate && item.plate !== '-');
                    const label = isVehicle ? (isPlate ? esc(item.plate) : 'Kendaraan') : 'Orang';
                    const icon = isVehicle ? 'K' : '<i class="bi bi-person-fill"></i>';
                    let photoPath = '';
                    if (isVehicle) {
                        photoPath = item.vehicle_image_path || item.vehicleImagePath || item.plate_image_path || item.plateImagePath || '';
                    } else {
                        photoPath = item.face_image_path || item.faceImagePath || '';
                    }

                    return `
                        <div class="recent d-flex align-items-center justify-content-between p-2 rounded mb-2" 
                             style="cursor: pointer; transition: background 0.2s;" 
                             onclick="openImageModal('${photoPath}', '${label}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi AI: <b>${item.confidence_percent}%</b> · Status: <b>${esc(item.status)}</b>', '${item.object_type}')">
                            <div class="d-flex align-items-center gap-3">
                                <div class="plate-icon ${isPlate ? '' : 'bg-primary text-white'}">${icon}</div>
                                <div class="recent-info">
                                    <strong class="d-block">${label}</strong>
                                    <span class="text-muted small">${esc(item.camera)} · ${esc(item.timestamp)}</span>
                                </div>
                            </div>
                            <div class="text-end">
                                <b class="d-block text-primary">${item.confidence_percent}%</b>
                                <span class="badge ${item.status === 'Terbaca' ? 'bg-success-subtle text-success border border-success-subtle' : 'bg-warning-subtle text-warning border border-warning-subtle'} small">${esc(item.status)}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        } catch (e) {
            console.error('Error refreshing recent list:', e);
        }
    }

    await updateDashboardMetrics();
    await refreshRecentList();

    // Auto polling setiap 3.5 detik untuk dashboard realtime
    if (!window._dashboardInterval) {
        window._dashboardInterval = setInterval(async () => {
            if (window.location.pathname === '/' || window.location.pathname === '/dashboard') {
                await updateDashboardMetrics();
                await refreshRecentList();
            }
        }, 3500);
    }
}


// ============================================================
// MODUL: HASIL DETEKSI TERPADU (/detections)
// ============================================================

let detState = {
    page: 1,
    limit: 15,
    type: 'all',
    status: 'all',
    camera_id: '',
    search: '',
    period: 'today',
    start_date: '',
    end_date: ''
};

async function loadDetections() {
    const tableBody = document.getElementById('detectionTable');
    if (!tableBody) return;

    tableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-5"><div class="spinner-border spinner-border-sm text-primary me-2"></div>Memuat data deteksi...</td></tr>`;

    try {
        const queryParams = new URLSearchParams({
            page: detState.page,
            limit: detState.limit,
            type: detState.type,
            status: detState.status,
            camera_id: detState.camera_id,
            search: detState.search,
            ...getStateDateRange(detState)
        });

        const res = await json(`/api/detections?${queryParams.toString()}`);
        const items = res.data || [];
        const total = res.total || 0;
        const totalPages = res.total_pages || 1;

        // Update info pagination
        const infoEl = document.getElementById('detPaginationInfo');
        const pageEl = document.getElementById('detPageNumber');
        const prevBtn = document.getElementById('detPrevBtn');
        const nextBtn = document.getElementById('detNextBtn');

        if (infoEl) infoEl.textContent = `Menampilkan ${(detState.page - 1) * detState.limit + (items.length ? 1 : 0)} - ${(detState.page - 1) * detState.limit + items.length} dari ${total} deteksi`;
        if (pageEl) pageEl.textContent = `Hal ${detState.page} dari ${totalPages}`;
        if (prevBtn) prevBtn.disabled = detState.page <= 1;
        if (nextBtn) nextBtn.disabled = detState.page >= totalPages;

        if (items.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-5"><i class="bi bi-inbox fs-2 d-block mb-2 text-secondary"></i>Tidak ada data deteksi yang sesuai filter.</td></tr>`;
            return;
        }

        tableBody.innerHTML = items.map(item => {
            const isVehicle = item.object_type === 'vehicle' || item.type === 'vehicle' || item.type === 'vehicle_with_plate' || item.type === 'plate';
            const isPlate = item.type === 'vehicle_with_plate' || (isVehicle && item.has_plate);
            const targetLabel = isVehicle ? `<strong class=\"d-block\">${esc(isPlate ? item.plate : 'Kendaraan')}</strong>` : `<span class=\"text-muted fst-italic\">Orang</span>`;
            let typeBadge = '';
            if (isPlate) {
                typeBadge = `<span class=\"badge bg-info-subtle text-info border border-info-subtle\"><i class=\"bi bi-car-front\"></i> Kendaraan / Plat Nomor</span>`;
            } else if (isVehicle) {
                typeBadge = `<span class=\"badge bg-primary-subtle text-primary border border-primary-subtle\"><i class=\"bi bi-car-front\"></i> Kendaraan</span>`;
            } else {
                typeBadge = `<span class=\"badge bg-secondary-subtle text-secondary border border-secondary-subtle\"><i class=\"bi bi-person\"></i> Orang</span>`;
            }

            let statusBadge = '';
            if (item.status_code === 1) {
                statusBadge = `<span class="badge bg-success-subtle text-success border border-success-subtle">Terbaca</span>`;
            } else if (item.status_code === 2) {
                statusBadge = `<span class="badge bg-warning-subtle text-warning border border-warning-subtle">Perlu Cek</span>`;
            } else {
                statusBadge = `<span class="badge bg-danger-subtle text-danger border border-danger-subtle">Gagal</span>`;
            }

            let photo = '';
        if (isVehicle) {
            photo = item.vehicle_image_path || item.vehicleImagePath || item.plate_image_path || item.plateImagePath || '';
        } else {
            photo = item.face_image_path || item.faceImagePath || '';
        }
            const confidenceText = isPlate
                ? `Kendaraan ${item.vehicle_confidence_percent || 0}% · Plat ${item.plate_confidence_percent || 0}% · OCR ${item.ocr_confidence_percent || 0}%`
                : (isVehicle ? `Kendaraan ${item.vehicle_confidence_percent || item.confidence_percent || 0}%` : `Orang ${item.person_confidence_percent || item.confidence_percent || 0}%`);
            const thumbHtml = photo
                ? `<img src="/${esc(photo.replace(/^\/+/, ''))}" alt="Thumb" class="rounded border" style="width: 50px; height: 36px; object-fit: cover; cursor: pointer;" onclick="openImageModal('${esc(photo)}', '${esc(item.plate || 'Deteksi')}', '${esc(item.camera)} · ${esc(item.timestamp)}', '${esc(confidenceText)}')">`
                : `<div class="rounded border bg-light text-muted d-flex align-items-center justify-content-center small" style="width: 50px; height: 36px;"><i class="bi bi-image"></i></div>`;

            return `
                <tr>
                    <td>${thumbHtml}</td>
                    <td>${targetLabel}</td>
                    <td>${typeBadge}</td>
                    <td>${esc(item.camera)}</td>
                    <td>
                        <div class="d-flex align-items-center gap-2" style="max-width: 140px;">
                            <div class="progress flex-grow-1" style="height: 6px;">
                                <div class="progress-bar ${item.confidence_percent >= 70 ? 'bg-success' : (item.confidence_percent >= 40 ? 'bg-warning' : 'bg-danger')}" style="width: ${item.confidence_percent}%;"></div>
                            </div>
                            <span class="small fw-semibold text-nowrap">${item.confidence_percent}%</span>
                        </div>
                    </td>
                    <td><span class="small text-muted">${esc(item.timestamp)}</span></td>
                    <td>${statusBadge}</td>
                    <td class="text-center">
                        <button class="btn btn-outline-primary btn-sm px-2 py-1" title="Lihat Foto" onclick="openImageModal('${esc(photo)}', '${esc(item.plate || 'Detail Deteksi')}', '${esc(item.camera)} · ${esc(item.timestamp)}', '${esc(confidenceText)} · Status: ${esc(item.status)}')">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-outline-danger btn-sm px-2 py-1" title="Hapus Deteksi" onclick="deleteDetection(${item.id})">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loadDetections:', err);
        tableBody.innerHTML = `<tr><td colspan="8" class="text-center text-danger py-4">Gagal memuat data deteksi.</td></tr>`;
    }
}

async function deleteDetection(detectionId) {
    if (!await askConfirmation('Hapus histori deteksi ini beserta capture terkait?', 'Hapus Deteksi')) return;
    try {
        const result = await json(`/api/detections/${encodeURIComponent(detectionId)}`, { method: 'DELETE' });
        if (!result.success) throw new Error(result.message || 'Penghapusan gagal');
        await loadDetections();
    } catch (error) {
        console.error('Error deleteDetection:', error);
        showNotification(error.message || 'Gagal menghapus deteksi.', 'danger');
    }
}
window.deleteDetection = deleteDetection;

async function initDetectionsPage() {
    setDateFilterLimits();

    try {
        const camRes = await json('/api/cameras');
        const cams = camRes.data || [];
        const camSelect = document.getElementById('detCameraFilter');
        if (camSelect) {
            camSelect.innerHTML = '<option value="">Semua Kamera</option>' + cams.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
            camSelect.addEventListener('change', () => {
                detState.camera_id = camSelect.value;
                detState.page = 1;
                loadDetections();
            });
        }
    } catch (e) {}

    const typeSelect = document.getElementById('detTypeFilter');
    if (typeSelect) {
        typeSelect.addEventListener('change', () => {
            detState.type = typeSelect.value || 'all';
            detState.page = 1;
            loadDetections();
        });
    }

    const typeGroup = document.getElementById('detTypeButtonGroup');
    if (typeGroup) {
        typeGroup.addEventListener('click', e => {
            const btn = e.target.closest('button');
            if (!btn) return;
            typeGroup.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            detState.type = btn.dataset.type || 'all';
            if (typeSelect) typeSelect.value = detState.type;
            detState.page = 1;
            loadDetections();
        });
    }

    const statusSelect = document.getElementById('detStatusFilter');
    if (statusSelect) {
        statusSelect.addEventListener('change', () => {
            detState.status = statusSelect.value;
            detState.page = 1;
            loadDetections();
        });
    }

    const searchInput = document.getElementById('detectionSearch');
    let searchTimer = null;
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimer);
            searchTimer = setTimeout(() => {
                detState.search = searchInput.value.trim();
                detState.page = 1;
                loadDetections();
            }, 350);
        });
    }

    const periodSelect = document.getElementById('detPeriodFilter');
    if (periodSelect) {
        periodSelect.addEventListener('change', () => {
            detState.period = periodSelect.value;
            detState.start_date = '';
            detState.end_date = '';
            document.getElementById('detStartDate').value = '';
            document.getElementById('detEndDate').value = '';
            detState.page = 1;
            loadDetections();
        });
    }

    document.getElementById('detDateApply')?.addEventListener('click', () => {
        if (applyCustomDateRange(detState, 'detStartDate', 'detEndDate')) {
            detState.page = 1;
            loadDetections();
        }
    });

    document.getElementById('detPrevBtn')?.addEventListener('click', () => {
        if (detState.page > 1) {
            detState.page--;
            loadDetections();
        }
    });

    document.getElementById('detNextBtn')?.addEventListener('click', () => {
        detState.page++;
        loadDetections();
    });

    document.getElementById('detResetBtn')?.addEventListener('click', () => {
        detState.page = 1;
        detState.type = 'all';
        detState.status = 'all';
        detState.camera_id = '';
        detState.search = '';
        detState.period = 'today';
        detState.start_date = '';
        detState.end_date = '';

        const searchInput = document.getElementById('detectionSearch');
        if (searchInput) searchInput.value = '';
        const typeSelect = document.getElementById('detTypeFilter');
        if (typeSelect) typeSelect.value = 'all';
        const cameraSelect = document.getElementById('detCameraFilter');
        if (cameraSelect) cameraSelect.value = '';
        const statusSelect = document.getElementById('detStatusFilter');
        if (statusSelect) statusSelect.value = 'all';
        const periodSelect = document.getElementById('detPeriodFilter');
        if (periodSelect) periodSelect.value = 'today';
        const startInput = document.getElementById('detStartDate');
        if (startInput) startInput.value = '';
        const endInput = document.getElementById('detEndDate');
        if (endInput) endInput.value = '';

        loadDetections().catch(err => console.error('Error reset detections filter:', err));
    });

    await loadDetections();
}

// Parameter filter SAMA dengan yang dipakai loadDetections() (tanpa page/limit)
function detectionExportParams() {
    return new URLSearchParams({
        type: detState.type,
        status: detState.status,
        camera_id: detState.camera_id,
        search: detState.search,
        ...getStateDateRange(detState)
    });
}

function exportDetections() {
    window.location.href = `/api/export/detections?${detectionExportParams().toString()}`;
}
window.exportDetections = exportDetections;

function exportDetectionsPdf() {
    window.location.href = `/api/export/detections/pdf?${detectionExportParams().toString()}`;
}
window.exportDetectionsPdf = exportDetectionsPdf;


// ============================================================
// MODUL: RIWAYAT PLAT NOMOR (/history)
// ============================================================

let plateState = {
    page: 1,
    limit: 15,
    search: '',
    camera_id: '',
    status: 'all',
    period: 'today',
    start_date: '',
    end_date: ''
};

async function loadPlateHistory() {
    const tableBody = document.getElementById('plateHistoryTable');
    if (!tableBody) return;

    tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted py-5"><div class="spinner-border spinner-border-sm text-primary me-2"></div>Memuat riwayat plat...</td></tr>`;

    try {
        const queryParams = new URLSearchParams({
            page: plateState.page,
            limit: plateState.limit,
            search: plateState.search,
            camera_id: plateState.camera_id,
            status: plateState.status,
            ...getStateDateRange(plateState)
        });

        const res = await json(`/api/plate/history?${queryParams.toString()}`);
        const items = res.data || [];
        const total = res.total || 0;
        const totalPages = res.total_pages || 1;

        const infoEl = document.getElementById('platePaginationInfo');
        const pageEl = document.getElementById('platePageNumber');
        const prevBtn = document.getElementById('platePrevBtn');
        const nextBtn = document.getElementById('plateNextBtn');

        if (infoEl) infoEl.textContent = `Menampilkan ${(plateState.page - 1) * plateState.limit + (items.length ? 1 : 0)} - ${(plateState.page - 1) * plateState.limit + items.length} dari ${total} plat`;
        if (pageEl) pageEl.textContent = `Hal ${plateState.page} dari ${totalPages}`;
        if (prevBtn) prevBtn.disabled = plateState.page <= 1;
        if (nextBtn) nextBtn.disabled = plateState.page >= totalPages;

        if (items.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted py-5"><i class="bi bi-inbox fs-2 d-block mb-2 text-secondary"></i>Tidak ada riwayat plat yang ditemukan.</td></tr>`;
            return;
        }

        tableBody.innerHTML = items.map(item => {
            let statusBadge = '';
            if (item.status_code === 1) {
                statusBadge = `<span class="badge bg-success-subtle text-success border border-success-subtle">Terbaca</span>`;
            } else if (item.status_code === 2) {
                statusBadge = `<span class="badge bg-warning-subtle text-warning border border-warning-subtle">Perlu Cek</span>`;
            } else {
                statusBadge = `<span class="badge bg-danger-subtle text-danger border border-danger-subtle">Gagal</span>`;
            }

            const photo = item.image_path || '';
            const thumbHtml = photo
                ? `<img src="/${esc(photo.replace(/^\/+/, ''))}" alt="Plate" class="rounded border" style="width: 54px; height: 34px; object-fit: cover; cursor: pointer;" onclick="openImageModal('${esc(photo)}', 'Plat: ${esc(item.plate)}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi OCR: <b>${item.confidence_percent}%</b>')">`
                : `<div class="rounded border bg-light text-muted d-flex align-items-center justify-content-center small" style="width: 54px; height: 34px;"><b>P</b></div>`;

            return `
                <tr>
                    <td>${thumbHtml}</td>
                    <td><strong class="fs-6 text-primary">${esc(item.plate)}</strong></td>
                    <td>${esc(item.camera)}</td>
                    <td>
                        <div class="d-flex align-items-center gap-2" style="max-width: 130px;">
                            <div class="progress flex-grow-1" style="height: 6px;">
                                <div class="progress-bar ${item.confidence_percent >= 70 ? 'bg-success' : 'bg-warning'}" style="width: ${item.confidence_percent}%;"></div>
                            </div>
                            <span class="small fw-semibold text-nowrap">${item.confidence_percent}%</span>
                        </div>
                    </td>
                    <td><span class="small text-muted">${esc(item.timestamp)}</span></td>
                    <td>${statusBadge}</td>
                    <td class="text-center">
                        <button class="btn btn-outline-primary btn-sm px-2 py-1" title="Lihat Foto Crop" onclick="openImageModal('${esc(photo)}', 'Plat Nomor: ${esc(item.plate)}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Confidence OCR: <b>${item.confidence_percent}%</b> · Status: <b>${esc(item.status)}</b>')">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-outline-danger btn-sm px-2 py-1" title="Hapus Riwayat Plat" onclick="deletePlateHistory(${item.id})">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loadPlateHistory:', err);
        tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-danger py-4">Gagal memuat riwayat plat.</td></tr>`;
    }
}

async function deletePlateHistory(plateId) {
    if (!await askConfirmation('Hapus riwayat plat ini beserta event dan capture terkait?', 'Hapus Riwayat Plat')) return;
    try {
        const result = await json(`/api/plate/history/${encodeURIComponent(plateId)}`, { method: 'DELETE' });
        if (!result.success) throw new Error(result.message || 'Penghapusan gagal');
        await loadPlateHistory();
    } catch (error) {
        console.error('Error deletePlateHistory:', error);
        showNotification(error.message || 'Gagal menghapus riwayat plat.', 'danger');
    }
}
window.deletePlateHistory = deletePlateHistory;

async function initHistoryPage() {
    setDateFilterLimits();

    try {
        const camRes = await json('/api/cameras');
        const cams = camRes.data || [];
        const camSelect = document.getElementById('plateCameraFilter');
        if (camSelect) {
            camSelect.innerHTML = '<option value="">Semua Kamera</option>' + cams.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
            camSelect.addEventListener('change', () => {
                plateState.camera_id = camSelect.value;
                plateState.page = 1;
                loadPlateHistory();
            });
        }
    } catch (e) {}

    const statusSelect = document.getElementById('plateStatusFilter');
    if (statusSelect) {
        statusSelect.addEventListener('change', () => {
            plateState.status = statusSelect.value;
            plateState.page = 1;
            loadPlateHistory();
        });
    }

    const searchInput = document.getElementById('plateSearch');
    let timer = null;
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            clearTimeout(timer);
            timer = setTimeout(() => {
                plateState.search = searchInput.value.trim();
                plateState.page = 1;
                loadPlateHistory();
            }, 350);
        });
    }

    const periodSelect = document.getElementById('platePeriodFilter');
    if (periodSelect) {
        periodSelect.addEventListener('change', () => {
            plateState.period = periodSelect.value;
            plateState.start_date = '';
            plateState.end_date = '';
            document.getElementById('plateStartDate').value = '';
            document.getElementById('plateEndDate').value = '';
            plateState.page = 1;
            loadPlateHistory();
        });
    }

    document.getElementById('plateDateApply')?.addEventListener('click', () => {
        if (applyCustomDateRange(plateState, 'plateStartDate', 'plateEndDate')) {
            plateState.page = 1;
            loadPlateHistory();
        }
    });

    document.getElementById('platePrevBtn')?.addEventListener('click', () => {
        if (plateState.page > 1) {
            plateState.page--;
            loadPlateHistory();
        }
    });

    document.getElementById('plateNextBtn')?.addEventListener('click', () => {
        plateState.page++;
        loadPlateHistory();
    });

    document.getElementById('plateResetBtn')?.addEventListener('click', () => {
        plateState.page = 1;
        plateState.search = '';
        plateState.camera_id = '';
        plateState.status = 'all';
        plateState.period = 'today';
        plateState.start_date = '';
        plateState.end_date = '';

        const searchInput = document.getElementById('plateSearch');
        if (searchInput) searchInput.value = '';
        const cameraSelect = document.getElementById('plateCameraFilter');
        if (cameraSelect) cameraSelect.value = '';
        const statusSelect = document.getElementById('plateStatusFilter');
        if (statusSelect) statusSelect.value = 'all';
        const periodSelect = document.getElementById('platePeriodFilter');
        if (periodSelect) periodSelect.value = 'today';
        const startInput = document.getElementById('plateStartDate');
        if (startInput) startInput.value = '';
        const endInput = document.getElementById('plateEndDate');
        if (endInput) endInput.value = '';

        loadPlateHistory().catch(err => console.error('Error reset plate history filter:', err));
    });

    await loadPlateHistory();
}

// Parameter filter SAMA dengan yang dipakai loadPlateHistory() (tanpa page/limit)
function plateExportParams() {
    return new URLSearchParams({
        search: plateState.search,
        camera_id: plateState.camera_id,
        status: plateState.status,
        ...getStateDateRange(plateState)
    });
}

function exportPlates() {
    window.location.href = `/api/export/plates?${plateExportParams().toString()}`;
}
window.exportPlates = exportPlates;

function exportPlatesPdf() {
    window.location.href = `/api/export/plates/pdf?${plateExportParams().toString()}`;
}
window.exportPlatesPdf = exportPlatesPdf;


// ============================================================
// MODUL: STATISTIK STANDAR PERUSAHAAN (ENTERPRISE ANALYTICS)
// ============================================================

function analyticsParams(prefix, periodOverride = '') {
    const period = periodOverride || document.getElementById(`${prefix}Period`)?.value || 'today';
    const params = new URLSearchParams({ period });
    const start = document.getElementById(`${prefix}Start`)?.value;
    const end = document.getElementById(`${prefix}End`)?.value;
    const camera = document.getElementById(`${prefix}Camera`)?.value;
    const region = document.getElementById(`${prefix}Region`)?.value;
    const gate = document.getElementById(`${prefix}Gate`)?.value;
    const type = document.getElementById(`${prefix}ObjectType`)?.value;
    const direction = document.getElementById(`${prefix}Direction`)?.value;
    if (start) params.set('start_date', start);
    if (end) params.set('end_date', end);
    if (camera) params.set('camera_id', camera);
    if (region) params.set('region', region);
    if (gate) params.set('gate', gate);
    if (type) params.set('object_type', type);
    if (direction) params.set('direction', direction);
    return params;
}

async function populateAnalyticsCameraSelect(id) {
    const select = document.getElementById(id);
    if (!select) return;
    const res = await json('/api/cameras').catch(() => ({ data: [] }));
    select.innerHTML = '<option value="">Semua CCTV</option>' + (res.data || []).map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
}

async function loadRecap() {
    const start = document.getElementById('recapStart')?.value || '';
    const end = document.getElementById('recapEnd')?.value || '';
    if (start && end && start > end) {
        throw new Error('Tanggal mulai tidak boleh lebih besar dari tanggal akhir.');
    }

    const applyButton = document.getElementById('recapApply');
    const cameraTable = document.getElementById('recapCameraTable');
    const dailyTable = document.getElementById('recapDailyTable');
    if (applyButton) applyButton.disabled = true;
    if (cameraTable) cameraTable.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">Memuat data sesuai filter...</td></tr>';
    if (dailyTable) dailyTable.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">Memuat data sesuai filter...</td></tr>';

    try {
        const res = await json(`/api/analytics?${analyticsParams('recap').toString()}`);
        if (!res.success) throw new Error(res.message || 'Data rekap tidak dapat dimuat.');

        const data = res.data || {};
        const s = data.summary || {};
        const cards = [
            ['Kendaraan masuk', s.vehicle_entry], ['Kendaraan keluar', s.vehicle_exit], ['Total kendaraan', s.vehicles],
            ['Orang masuk', s.people_entry], ['Orang keluar', s.people_exit], ['Total orang', s.people]
        ];
        const totals = document.getElementById('recapTotals');
        if (totals) totals.innerHTML = cards.map(([label, value]) => `<div class="col-xl-2 col-md-4 col-6"><div class="stat-card recap-stat"><div><span>${label}</span><strong>${value || 0}</strong></div></div></div>`).join('');
        if (cameraTable) cameraTable.innerHTML = (data.cameras || []).map(c => `<tr><td><strong>${esc(c.camera)}</strong></td><td>${esc(c.direction)}</td><td>${c.vehicles}</td><td>${c.people}</td><td>${c.unique_plates}</td><td><span class="status ${c.status === 'Aktif' ? 'success' : 'warning'}">${c.status}</span></td></tr>`).join('') || '<tr><td colspan="6" class="text-center text-muted py-4">Belum ada event pada filter ini.</td></tr>';
        if (dailyTable) dailyTable.innerHTML = (data.daily || []).map(d => `<tr><td>${esc(d.date)}</td><td>${d.vehicles}</td><td>${d.unique_plates}</td><td>${d.entry}</td><td>${d.exit}</td><td>${d.people}</td></tr>`).join('') || '<tr><td colspan="6" class="text-center text-muted py-4">Belum ada data harian.</td></tr>';
    } catch (error) {
        const message = esc(error.message || 'Gagal memuat data rekapitulasi.');
        if (cameraTable) cameraTable.innerHTML = `<tr><td colspan="6" class="text-center text-danger py-4">${message}</td></tr>`;
        if (dailyTable) dailyTable.innerHTML = `<tr><td colspan="6" class="text-center text-danger py-4">${message}</td></tr>`;
        throw error;
    } finally {
        if (applyButton) applyButton.disabled = false;
    }
}

function exportRecap() {
    window.location.href = `/api/export/recap?${analyticsParams('recap').toString()}`;
}
window.exportRecap = exportRecap;

function exportRecapPdf() {
    window.location.href = `/api/export/recap/pdf?${analyticsParams('recap').toString()}`;
}
window.exportRecapPdf = exportRecapPdf;

window.loadRecap = loadRecap;

async function initRecap() {
    setDateFilterLimits();
    await populateAnalyticsCameraSelect('recapCamera');
    document.getElementById('recapApply')?.addEventListener('click', () => {
        loadRecap().catch(err => {
            if (err.message) showNotification(err.message, 'warning');
        });
    });
    document.getElementById('recapResetBtn')?.addEventListener('click', () => {
        const periodSelect = document.getElementById('recapPeriod');
        const cameraSelect = document.getElementById('recapCamera');
        const directionSelect = document.getElementById('recapDirection');
        const startInput = document.getElementById('recapStart');
        const endInput = document.getElementById('recapEnd');

        if (periodSelect) periodSelect.value = 'today';
        if (cameraSelect) cameraSelect.value = '';
        if (directionSelect) directionSelect.value = '';
        if (startInput) startInput.value = '';
        if (endInput) endInput.value = '';

        loadRecap().catch(err => console.error('Error reset recap filter:', err));
    });
    document.getElementById('recapExport')?.addEventListener('click', exportRecap);
    document.getElementById('recapExportPdf')?.addEventListener('click', exportRecapPdf);
    loadRecap().catch(err => console.error('Error loadRecap:', err));
}

window._currentStatsPeriod = 'today';
window._trendChart = null;
window._donutChart = null;
window._cameraChart = null;

async function loadEnterpriseStatistics(period = 'today') {
    window._currentStatsPeriod = period;

    const periodSelect = document.getElementById('statsPeriod');
    if (periodSelect) periodSelect.value = period;
    const start = document.getElementById('statsStart')?.value || '';
    const end = document.getElementById('statsEnd')?.value || '';
    if (start && end && start > end) {
        throw new Error('Tanggal mulai tidak boleh lebih besar dari tanggal akhir.');
    }

    const applyButton = document.getElementById('statsApply');
    if (applyButton) applyButton.disabled = true;
    const analyticsRes = await json(`/api/analytics?${analyticsParams('stats', period).toString()}`).catch(() => null);
    if (analyticsRes?.success && analyticsRes.data) {
        const data = analyticsRes.data;
        const s = data.summary || {};
        const kpiValues = {
            kpiTotalDets: s.vehicles, kpiTotalPlates: s.plates, kpiTotalFaces: s.people,
            kpiPlateRate: s.vehicles ? `${Math.round((s.plates / s.vehicles) * 100)}%` : '0%',
            kpiAvgConf: '-', kpiNeedCheck: '-', kpiCamStatus: `${s.active_cameras || 0} / ${s.total_cameras || 0}`, kpiCamUptime: '-'
        };
        Object.entries(kpiValues).forEach(([id, value]) => { const el = document.getElementById(id); if (el) el.textContent = value ?? 0; });
        const trend = data.daily?.length ? data.daily : data.hourly;
        const labels = trend.map(item => item.date || item.label);
        const vehicleData = trend.map(item => item.vehicles || 0);
        const peopleData = trend.map(item => item.people || 0);
        const trendCanvas = document.getElementById('trendChartCanvas');
        if (trendCanvas && window.Chart) {
            if (window._trendChart) window._trendChart.destroy();
            window._trendChart = new Chart(trendCanvas, { type: 'line', data: { labels, datasets: [{ label: 'Kendaraan', data: vehicleData, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.12)', fill: true, tension: .3 }, { label: 'Orang', data: peopleData, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.08)', fill: true, tension: .3 }] }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } } });
        }
        const camCanvas = document.getElementById('cameraBarCanvas');
        if (camCanvas && window.Chart) {
            if (window._cameraChart) window._cameraChart.destroy();
            window._cameraChart = new Chart(camCanvas, { type: 'bar', data: { labels: (data.cameras || []).map(c => c.camera), datasets: [{ label: 'Kendaraan', data: (data.cameras || []).map(c => c.vehicles), backgroundColor: '#2563eb' }, { label: 'Orang', data: (data.cameras || []).map(c => c.people), backgroundColor: '#f59e0b' }] }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } } });
        }
        const peak = (data.hourly || []).reduce((best, item) => item.vehicles > (best?.vehicles || -1) ? item : best, null);
        const peakContainer = document.getElementById('peakHoursList');
        if (peakContainer) peakContainer.innerHTML = peak ? `<div class="p-3 bg-light rounded border"><strong>${peak.label} - ${String((peak.hour + 1) % 24).padStart(2, '0')}:00</strong><span class="d-block text-primary mt-1">${peak.vehicles} kendaraan · ${peak.people} orang</span></div>` : '<div class="text-muted">Belum ada data.</div>';
        const topBody = document.getElementById('topPlatesTable');
        if (topBody) topBody.innerHTML = (data.top_plates || []).map((p, i) => `<tr><td>${i + 1}</td><td><strong>${esc(p.plate)}</strong></td><td>${p.count} kali</td><td>${esc(p.camera)}</td><td>${esc(p.last_seen)}</td><td>${p.confidence}%</td><td><span class="badge bg-success-subtle text-success">Aktual</span></td></tr>`).join('') || '<tr><td colspan="7" class="text-center text-muted py-4">Belum ada data plat.</td></tr>';
        const badgeEl = document.getElementById('statsPeriodBadge');
        if (badgeEl) badgeEl.textContent = { today: 'Hari Ini', '7d': '7 Hari Terakhir', '30d': '30 Hari', all: 'Semua Waktu' }[period] || period;
        if (applyButton) applyButton.disabled = false;
        return;
    }

    const badgeEl = document.getElementById('statsPeriodBadge');
    if (badgeEl) {
        const periodLabels = { 'today': 'Hari Ini', '7d': '7 Hari Terakhir', '30d': '30 Hari Terakhir', 'all': 'Semua Waktu' };
        badgeEl.textContent = periodLabels[period] || period;
    }

    try {
        const res = await json(`/api/statistics/enterprise?period=${period}`);
        const stats = res.data || {};
        const kpi = stats.kpi || {};
        const trend = stats.trend || {};
        const camDist = stats.camera_distribution || [];
        const statusBreakdown = stats.status_breakdown || {};
        const peakHours = stats.peak_hours || [];
        const topPlates = stats.top_plates || [];

        // 1. Update 6 KPI Cards
        const elTotalDets = document.getElementById('kpiTotalDets');
        const elTotalPlates = document.getElementById('kpiTotalPlates');
        const elPlateRate = document.getElementById('kpiPlateRate');
        const elTotalFaces = document.getElementById('kpiTotalFaces');
        const elAvgConf = document.getElementById('kpiAvgConf');
        const elNeedCheck = document.getElementById('kpiNeedCheck');
        const elCamStatus = document.getElementById('kpiCamStatus');
        const elCamUptime = document.getElementById('kpiCamUptime');

        if (elTotalDets) elTotalDets.textContent = kpi.total_detections || 0;
        if (elTotalPlates) elTotalPlates.textContent = kpi.total_plates || 0;
        if (elPlateRate) elPlateRate.textContent = `${kpi.plate_read_rate || 0}%`;
        if (elTotalFaces) elTotalFaces.textContent = kpi.total_faces || 0;
        if (elAvgConf) elAvgConf.textContent = `${kpi.avg_confidence || 0}%`;
        if (elNeedCheck) elNeedCheck.textContent = kpi.need_check_count || 0;
        if (elCamStatus) elCamStatus.textContent = `${kpi.active_cameras || 0} / ${kpi.total_cameras || 0}`;
        if (elCamUptime) elCamUptime.textContent = `${kpi.camera_availability || 0}%`;

        // 2. Render Chart.js Tren Deteksi Waktu Nyata (Line)
        const trendCanvas = document.getElementById('trendChartCanvas');
        if (trendCanvas && window.Chart) {
            if (window._trendChart) window._trendChart.destroy();

            const labels = trend.labels || [];
            const plateData = trend.plates || [];
            const faceData = trend.faces || [];

            window._trendChart = new Chart(trendCanvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Plat Kendaraan',
                            data: plateData,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.12)',
                            fill: true,
                            tension: 0.35,
                            borderWidth: 2.5,
                            pointRadius: labels.length > 20 ? 1.5 : 3.5,
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Wajah / Pengendara',
                            data: faceData,
                            borderColor: '#2563eb',
                            backgroundColor: 'rgba(37, 99, 235, 0.08)',
                            fill: true,
                            tension: 0.35,
                            borderWidth: 2.5,
                            pointRadius: labels.length > 20 ? 1.5 : 3.5,
                            pointHoverRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { font: { family: 'Segoe UI, Inter', size: 12, weight: '600' } }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(15, 23, 42, 0.92)',
                            titleFont: { weight: 'bold' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#f1f5f9' },
                            ticks: { precision: 0 }
                        },
                        x: {
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        // 3. Render Chart.js Donut Kualitas Deteksi SLA
        const donutCanvas = document.getElementById('qualityDonutCanvas');
        if (donutCanvas && window.Chart) {
            if (window._donutChart) window._donutChart.destroy();

            const validCount = statusBreakdown.valid || 0;
            const warningCount = statusBreakdown.warning || 0;
            const failedCount = statusBreakdown.failed || 0;

            const elV = document.getElementById('donutValValid');
            const elW = document.getElementById('donutValWarning');
            const elF = document.getElementById('donutValFailed');
            if (elV) elV.textContent = validCount;
            if (elW) elW.textContent = warningCount;
            if (elF) elF.textContent = failedCount;

            window._donutChart = new Chart(donutCanvas, {
                type: 'doughnut',
                data: {
                    labels: ['Valid (Jelas)', 'Perlu Review', 'Gagal'],
                    datasets: [{
                        data: [validCount, warningCount, failedCount],
                        backgroundColor: ['#16a34a', '#f59e0b', '#ef4444'],
                        borderWidth: 3,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '72%',
                    plugins: {
                        legend: { display: false },
                        tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.92)' }
                    }
                }
            });
        }

        // 4. Render Chart.js Bar Distribusi CCTV
        const camCanvas = document.getElementById('cameraBarCanvas');
        if (camCanvas && window.Chart) {
            if (window._cameraChart) window._cameraChart.destroy();

            const camNames = camDist.map(c => c.camera_name);
            const camPlates = camDist.map(c => c.plate_count);
            const camFaces = camDist.map(c => c.face_count);

            window._cameraChart = new Chart(camCanvas, {
                type: 'bar',
                data: {
                    labels: camNames.length ? camNames : ['Belum Ada Data'],
                    datasets: [
                        {
                            label: 'Plat Nomor',
                            data: camPlates.length ? camPlates : [0],
                            backgroundColor: '#10b981',
                            borderRadius: 4
                        },
                        {
                            label: 'Wajah / Orang',
                            data: camFaces.length ? camFaces : [0],
                            backgroundColor: '#3b82f6',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.92)' }
                    },
                    scales: {
                        x: { stacked: true, grid: { display: false } },
                        y: { stacked: true, beginAtZero: true, grid: { color: '#f1f5f9' }, ticks: { precision: 0 } }
                    }
                }
            });
        }

        // 5. Render Analisis Jam Sibuk (Peak Hours)
        const peakContainer = document.getElementById('peakHoursList');
        if (peakContainer) {
            if (!peakHours.length) {
                peakContainer.innerHTML = '<div class="text-muted text-center py-4">Belum ada data jam sibuk pada periode ini.</div>';
            } else {
                peakContainer.innerHTML = peakHours.map((ph, idx) => `
                    <div class="p-3 bg-light rounded border">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <strong class="small text-dark">
                                <span class="badge bg-primary me-2">#${idx + 1}</span>${esc(ph.time_range)}
                            </strong>
                            <span class="small fw-bold text-primary">${ph.count} deteksi (${ph.percentage}%)</span>
                        </div>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-primary" style="width: ${Math.min(100, ph.percentage * 2)}%;"></div>
                        </div>
                    </div>
                `).join('');
            }
        }

        // 6. Render Top 10 Plat Sering Terdeteksi
        const topPlatesBody = document.getElementById('topPlatesTable');
        if (topPlatesBody) {
            if (!topPlates.length) {
                topPlatesBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">Belum ada data plat kendaraan pada periode ini.</td></tr>';
            } else {
                topPlatesBody.innerHTML = topPlates.map((tp, i) => `
                    <tr>
                        <td class="fw-bold text-muted">${i + 1}</td>
                        <td><strong class="fs-6 text-primary">${esc(tp.plate_number)}</strong></td>
                        <td><span class="badge bg-secondary-subtle text-secondary border px-2 py-1">${tp.total_seen} kali</span></td>
                        <td>${esc(tp.last_camera)}</td>
                        <td><span class="small text-muted">${esc(tp.last_seen)}</span></td>
                        <td>
                            <span class="small fw-semibold ${tp.avg_confidence_percent >= 70 ? 'text-success' : 'text-warning'}">
                                ${tp.avg_confidence_percent}%
                            </span>
                        </td>
                        <td>
                            <span class="badge ${tp.status_code === 1 ? 'bg-success-subtle text-success border border-success-subtle' : 'bg-warning-subtle text-warning border border-warning-subtle'}">
                                ${esc(tp.status)}
                            </span>
                        </td>
                    </tr>
                `).join('');
            }
        }
    } catch (e) {
        console.error('Error loadEnterpriseStatistics:', e);
    } finally {
        if (applyButton) applyButton.disabled = false;
    }
}
window.loadEnterpriseStatistics = loadEnterpriseStatistics;

function exportStatsReport() {
    const period = window._currentStatsPeriod || document.getElementById('statsPeriod')?.value || 'today';
    window.location.href = `/api/export/statistics?${analyticsParams('stats', period).toString()}`;
}
window.exportStatsReport = exportStatsReport;

function exportStatsPdf() {
    const period = window._currentStatsPeriod || document.getElementById('statsPeriod')?.value || 'today';
    window.location.href = `/api/export/statistics/pdf?${analyticsParams('stats', period).toString()}`;
}
window.exportStatsPdf = exportStatsPdf;

async function initStatistics() {
    setDateFilterLimits();
    await populateAnalyticsCameraSelect('statsCamera');
    document.getElementById('statsApply')?.addEventListener('click', () => {
        loadEnterpriseStatistics(document.getElementById('statsPeriod')?.value || 'today').catch(err => showNotification(err.message, 'warning'));
    });
    document.getElementById('statsResetBtn')?.addEventListener('click', () => {
        const periodSelect = document.getElementById('statsPeriod');
        const startInput = document.getElementById('statsStart');
        const endInput = document.getElementById('statsEnd');
        const objectType = document.getElementById('statsObjectType');
        const cameraSelect = document.getElementById('statsCamera');

        if (periodSelect) periodSelect.value = 'today';
        if (startInput) startInput.value = '';
        if (endInput) endInput.value = '';
        if (objectType) objectType.value = '';
        if (cameraSelect) cameraSelect.value = '';
        window._currentStatsPeriod = 'today';

        loadEnterpriseStatistics('today').catch(err => console.error('Error reset stats filter:', err));
    });
    const periodGroup = document.getElementById('statsPeriodGroup');
    if (periodGroup) {
        periodGroup.addEventListener('click', e => {
            const btn = e.target.closest('button');
            if (!btn) return;
            periodGroup.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const period = btn.dataset.period || 'today';
            const periodSelect = document.getElementById('statsPeriod');
            if (periodSelect) periodSelect.value = period;
            loadEnterpriseStatistics(period).catch(err => console.error('Error loadEnterpriseStatistics:', err));
        });
    }

    await loadEnterpriseStatistics(document.getElementById('statsPeriod')?.value || 'today');
}


// ============================================================
// MODUL: PENGATURAN SISTEM (/settings)
// ============================================================

async function loadSettingsFromDb() {
    try {
        const res = await json('/api/settings');
        const settings = res.settings || {};
        const diag = res.diagnostics || {};

        // 1. Isi form
        const opName = document.getElementById('operatorName');
        const email = document.getElementById('profileEmail');
        const phone = document.getElementById('profilePhone');
        const role = document.getElementById('profileRole');
        if (opName) opName.value = settings.operator_name || 'Administrator';
        if (email) email.value = settings.profile_email || '';
        if (phone) phone.value = settings.profile_phone || '';
        if (role) role.value = settings.profile_role || 'Operator';
        const displayName = document.getElementById('profileDisplayName');
        const displayRole = document.getElementById('profileDisplayRole');
        const topbarName = document.getElementById('topbarName');
        const topbarRole = document.getElementById('topbarRole');
        const avatar = document.getElementById('topbarAvatar');
        const name = settings.operator_name || 'Administrator';
        const profileRole = settings.profile_role || 'Operator';
        if (displayName) displayName.textContent = name;
        if (displayRole) displayRole.textContent = profileRole;
        if (topbarName) topbarName.textContent = name;
        if (topbarRole) topbarRole.textContent = profileRole;
        if (avatar) avatar.textContent = name.charAt(0).toUpperCase();

        // 2. Isi diagnostik sistem
        const diagStatus = document.getElementById('diagDbStatus');
        const diagName = document.getElementById('diagDbName');
        const diagDets = document.getElementById('diagTotalDets');
        const diagPlates = document.getElementById('diagTotalPlates');
        const diagStorage = document.getElementById('diagStorage');
        const diagTime = document.getElementById('diagServerTime');

        if (diagStatus) {
            diagStatus.textContent = diag.db_connected ? 'Terhubung (Online)' : 'Terputus (Error)';
            diagStatus.className = `badge ${diag.db_connected ? 'bg-success-subtle text-success border border-success-subtle' : 'bg-danger-subtle text-danger border border-danger-subtle'}`;
        }
        if (diagName) diagName.textContent = diag.db_name || 'real_cctv';
        if (diagDets) diagDets.textContent = diag.table_counts?.full_detection || 0;
        if (diagPlates) diagPlates.textContent = diag.table_counts?.plate || 0;
        if (diagStorage) diagStorage.textContent = `${diag.storage?.total_size_mb || 0} MB (${diag.storage?.total_files || 0} file)`;
        if (diagTime) diagTime.textContent = diag.server_time || '-';
    } catch (e) {
        console.error('Error loadSettingsFromDb:', e);
    }
}

async function seedDemoData() {
    const btn = document.getElementById('btnSeedDemoData');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Menghasilkan data simulasi...`;
    }

    try {
        const res = await json('/api/settings/seed-demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: 35 })
        });

        showNotification(res.message || 'Data simulasi berhasil ditambahkan!', 'success');
        await loadSettingsFromDb();
    } catch (e) {
        showNotification('Gagal menambahkan data simulasi: ' + e.message, 'danger');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i class="bi bi-database-fill-add"></i> Generate Data Simulasi (+35 data)`;
        }
    }
}
window.seedDemoData = seedDemoData;

function initSettings() {
    const form = document.getElementById('settingsForm');
    if (!form) return;

    loadSettingsFromDb();

    form.addEventListener('submit', async event => {
        event.preventDefault();
        const msgEl = document.getElementById('settingsMessage');
        if (msgEl) msgEl.textContent = 'Menyimpan ke database MySQL...';

        const payload = {
            operator_name: document.getElementById('operatorName')?.value,
            profile_email: document.getElementById('profileEmail')?.value,
            profile_phone: document.getElementById('profilePhone')?.value,
            profile_role: document.getElementById('profileRole')?.value
        };

        try {
            const res = await json('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (msgEl) {
                msgEl.textContent = res.message || 'Pengaturan berhasil disimpan ke database!';
                setTimeout(() => { msgEl.textContent = ''; }, 4000);
            }
            await loadSettingsFromDb();
        } catch (err) {
            if (msgEl) msgEl.textContent = 'Gagal menyimpan: ' + err.message;
        }
    });
}


// ============================================================
// MODUL: SIDEBAR & JAM SISTEM
// ============================================================

function updateClock() {
    const clock = document.getElementById('clock');
    if (clock) {
        clock.textContent = new Date().toLocaleTimeString('id-ID', { hour12: false, timeZone: 'Asia/Jakarta' }) + ' WIB';
    }
}

function initSidebarToggle() {
    const layout = document.querySelector('.app-layout');
    const toggleBtn = document.getElementById('sidebarToggle');
    const closeBtn = document.getElementById('sidebarCloseBtn');
    const backdrop = document.getElementById('sidebarBackdrop');

    if (!layout || !toggleBtn) return;

    const isDesktopCollapsed = localStorage.getItem('platevision.sidebar_collapsed') === 'true';
    if (window.innerWidth >= 992 && isDesktopCollapsed) {
        layout.classList.add('sidebar-collapsed');
    }

    toggleBtn.addEventListener('click', () => {
        if (window.innerWidth >= 992) {
            layout.classList.toggle('sidebar-collapsed');
            const collapsed = layout.classList.contains('sidebar-collapsed');
            localStorage.setItem('platevision.sidebar_collapsed', collapsed ? 'true' : 'false');
        } else {
            layout.classList.toggle('sidebar-mobile-open');
        }
    });

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            layout.classList.remove('sidebar-mobile-open');
        });
    }

    if (backdrop) {
        backdrop.addEventListener('click', () => {
            layout.classList.remove('sidebar-mobile-open');
        });
    }

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && layout.classList.contains('sidebar-mobile-open')) {
            layout.classList.remove('sidebar-mobile-open');
        }
    });

    window.addEventListener('resize', () => {
        if (window.innerWidth >= 992 && layout.classList.contains('sidebar-mobile-open')) {
            layout.classList.remove('sidebar-mobile-open');
        }
    });
}


// ============================================================
// THEME MANAGER (DARK / LIGHT MODE)
// ============================================================

function getActiveTheme() {
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function updateThemeUI(theme) {
    const icon = document.getElementById('themeToggleIcon');
    const label = document.getElementById('themeToggleLabel');
    const btn = document.getElementById('themeToggleBtn');

    if (theme === 'dark') {
        document.documentElement.classList.add('dark');
        document.documentElement.setAttribute('data-bs-theme', 'dark');
        if (icon) {
            icon.className = 'bi bi-sun-fill text-warning';
        }
        if (label) {
            label.textContent = 'Mode Terang';
        }
        if (btn) {
            btn.title = 'Beralih ke Mode Terang';
            btn.setAttribute('aria-label', 'Beralih ke Mode Terang');
        }
    } else {
        document.documentElement.classList.remove('dark');
        document.documentElement.setAttribute('data-bs-theme', 'light');
        if (icon) {
            icon.className = 'bi bi-moon-stars-fill';
        }
        if (label) {
            label.textContent = 'Mode Gelap';
        }
        if (btn) {
            btn.title = 'Beralih ke Mode Gelap';
            btn.setAttribute('aria-label', 'Beralih ke Mode Gelap');
        }
    }

    refreshChartThemes(theme);
}

function refreshChartThemes(theme) {
    if (!window.Chart) return;
    const isDark = theme === 'dark';
    const textColor = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : '#f1f5f9';

    [window._trendChart, window._cameraChart, window._statusChart].forEach(chart => {
        if (!chart) return;
        try {
            if (chart.options && chart.options.scales) {
                if (chart.options.scales.x) {
                    chart.options.scales.x.ticks = chart.options.scales.x.ticks || {};
                    chart.options.scales.x.ticks.color = textColor;
                    if (chart.options.scales.x.grid) chart.options.scales.x.grid.color = gridColor;
                }
                if (chart.options.scales.y) {
                    chart.options.scales.y.ticks = chart.options.scales.y.ticks || {};
                    chart.options.scales.y.ticks.color = textColor;
                    if (chart.options.scales.y.grid) chart.options.scales.y.grid.color = gridColor;
                }
            }
            if (chart.options && chart.options.plugins && chart.options.plugins.legend) {
                chart.options.plugins.legend.labels = chart.options.plugins.legend.labels || {};
                chart.options.plugins.legend.labels.color = textColor;
            }
            chart.update();
        } catch (e) {
            console.warn('Error updating chart theme:', e);
        }
    });
}

function toggleTheme() {
    const currentTheme = getActiveTheme();
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    try {
        localStorage.setItem('platevision_theme', nextTheme);
        localStorage.setItem('theme', nextTheme);
    } catch (e) {}
    updateThemeUI(nextTheme);
}

function initTheme() {
    let savedTheme = null;
    try {
        savedTheme = localStorage.getItem('platevision_theme') || localStorage.getItem('theme');
    } catch (e) {}

    if (!savedTheme) {
        // Default light mode as existing look
        savedTheme = 'light';
    }

    updateThemeUI(savedTheme);

    const toggleBtn = document.getElementById('themeToggleBtn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleTheme);
    }
}
window.toggleTheme = toggleTheme;
window.initTheme = initTheme;


// ============================================================
// INISIALISASI HALAMAN (ROUTING CLIENT-SIDE)
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSidebarToggle();
    setInterval(updateClock, 1000);
    updateClock();

    if (document.getElementById('cameraTable')) {
        initMonitoring();
    }
    if (document.getElementById('totalVehicle') || document.getElementById('liveStreamImg')) {
        initDashboard();
    }
    if (document.getElementById('detectionTable')) {
        initDetectionsPage();
    }
    if (document.getElementById('plateHistoryTable')) {
        initHistoryPage();
    }
    if (document.getElementById('recapPage')) {
        initRecap();
    }
    if (document.getElementById('statsApply')) {
        initStatistics();
    }
    if (document.getElementById('settingsForm')) {
        initSettings();
    }
});