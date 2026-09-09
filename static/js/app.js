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

// Fetch JSON helper
async function json(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok && response.status >= 500) {
        throw new Error(`Server error (${response.status})`);
    }
    return response.json();
}

// ============================================================
// MODAL GLOBAL: PRATINJAU GAMBAR & DETAIL DETEKSI
// ============================================================

function openImageModal(imgSrc, title = 'Detail Foto Tangkapan CCTV', meta = '', details = '') {
    const modalEl = document.getElementById('imagePreviewModal');
    if (!modalEl) return;

    const titleEl = document.getElementById('imagePreviewTitle');
    const metaEl = document.getElementById('imagePreviewMeta');
    const srcEl = document.getElementById('imagePreviewSrc');
    const detailsEl = document.getElementById('imagePreviewDetails');

    if (titleEl) titleEl.textContent = title;
    if (metaEl) metaEl.textContent = meta;
    if (detailsEl) detailsEl.innerHTML = details;

    if (srcEl) {
        if (imgSrc && imgSrc !== 'null' && imgSrc !== 'undefined') {
            srcEl.src = '/' + imgSrc.replace(/^\/+/, '');
            srcEl.style.display = 'block';
        } else {
            srcEl.src = 'https://placehold.co/600x400/1e293b/94a3b8?text=Foto+Tidak+Tersedia';
            srcEl.style.display = 'block';
        }
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
        const result = await json('/api/cameras');
        const cameras = result.data || [];
        if (!cameras.length) {
            alert('Belum ada kamera untuk diekspor.');
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
        } else if (format === 'excel' || format === 'csv') {
            let csv = 'ID,Nama CCTV,URL Stream,Status\n';
            cameras.forEach(cam => {
                csv += `"${cam.id}","${String(cam.name || '').replace(/"/g, '""')}","${String(cam.url || '').replace(/"/g, '""')}","${cam.active ? 'Aktif' : 'Nonaktif'}"\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'daftar_cctv.csv';
            a.click();
            URL.revokeObjectURL(a.href);
        }
    } catch (err) {
        console.error('Gagal mengekspor kamera:', err);
        alert('Gagal mengekspor kamera.');
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
        if (button.dataset.action === 'delete' && confirm('Hapus kamera ini?')) {
            await json(`/api/cameras/${id}`, { method: 'DELETE' });
            cameras = await loadCameras();
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
            const summaryRes = await json('/api/statistics/summary');
            const data = summaryRes.data || {};

            const elTotalVeh = document.getElementById('totalVehicle');
            const elPlateRead = document.getElementById('plateRead');
            const elNeedCheck = document.getElementById('needCheck');
            const elCamStatus = document.getElementById('cameraStatus');

            if (elTotalVeh) elTotalVeh.textContent = data.today_detections || data.total_full_detections || 0;
            if (elPlateRead) elPlateRead.textContent = data.plate_success || data.total_plates || 0;
            if (elNeedCheck) elNeedCheck.textContent = data.need_check || 0;
            if (elCamStatus) elCamStatus.textContent = `${data.active_cameras || 0} / ${data.total_cameras || 0}`;
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

    function getStreamUrl(camId) {
        const showBbox = bboxToggle ? (bboxToggle.checked ? 1 : 0) : 1;
        return `/api/video_feed/${camId}?bbox=${showBbox}&t=${Date.now()}`;
    }

    function startCameraStream(camId) {
        if (!streamImg) return;
        const cam = cameras.find(c => String(c.id) === String(camId));
        if (cam) {
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
        if (placeholder) placeholder.style.display = 'flex';
        if (cameraNameEl) cameraNameEl.textContent = 'Stream Dihentikan';
    }

    const activeCam = cameras.find(item => item.active) || cameras[0];
    if (activeCam && activeCam.active) {
        if (select) select.value = activeCam.id;
        startCameraStream(activeCam.id);
    } else {
        stopCameraStream();
    }

    if (select) {
        select.addEventListener('change', () => {
            if (select.value) startCameraStream(select.value);
        });
    }

    if (bboxToggle) {
        bboxToggle.addEventListener('change', () => {
            const selectedId = select?.value || activeCam?.id;
            if (selectedId && streamImg && streamImg.style.display !== 'none') {
                startCameraStream(selectedId);
            }
        });
    }

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            const selectedId = select?.value || activeCam?.id;
            if (selectedId) startCameraStream(selectedId);
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            stopCameraStream();
        });
    }

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
                    const isPlate = item.type === 'plate' || Boolean(item.plate && item.plate !== '-');
                    const label = isPlate ? esc(item.plate) : 'Wajah / Pengendara';
                    const icon = isPlate ? 'P' : '<i class="bi bi-person-fill"></i>';
                    const photoPath = item.plate_image_path || item.face_image_path || '';

                    return `
                        <div class="recent d-flex align-items-center justify-content-between p-2 rounded mb-2" 
                             style="cursor: pointer; transition: background 0.2s;" 
                             onclick="openImageModal('${photoPath}', '${label}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi AI: <b>${item.confidence_percent}%</b> · Status: <b>${esc(item.status)}</b>')">
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
    search: ''
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
            search: detState.search
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
            const isPlate = item.type === 'plate' || (item.plate && item.plate !== '-');
            const targetLabel = isPlate ? `<strong>${esc(item.plate)}</strong>` : `<span class="text-muted fst-italic">Wajah / Pejalan Kaki</span>`;
            
            let typeBadge = '';
            if (item.type === 'combined') {
                typeBadge = `<span class="badge bg-primary-subtle text-primary border border-primary-subtle">Terpadu (Plat + Wajah)</span>`;
            } else if (item.type === 'plate') {
                typeBadge = `<span class="badge bg-info-subtle text-info border border-info-subtle"><i class="bi bi-card-heading"></i> Plat Nomor</span>`;
            } else {
                typeBadge = `<span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle"><i class="bi bi-person"></i> Orang</span>`;
            }

            let statusBadge = '';
            if (item.status_code === 1) {
                statusBadge = `<span class="badge bg-success-subtle text-success border border-success-subtle">Terbaca</span>`;
            } else if (item.status_code === 2) {
                statusBadge = `<span class="badge bg-warning-subtle text-warning border border-warning-subtle">Perlu Cek</span>`;
            } else {
                statusBadge = `<span class="badge bg-danger-subtle text-danger border border-danger-subtle">Gagal</span>`;
            }

            const photo = item.plate_image_path || item.face_image_path || '';
            const thumbHtml = photo
                ? `<img src="/${esc(photo.replace(/^\/+/, ''))}" alt="Thumb" class="rounded border" style="width: 50px; height: 36px; object-fit: cover; cursor: pointer;" onclick="openImageModal('${esc(photo)}', '${esc(item.plate || 'Deteksi')}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi: <b>${item.confidence_percent}%</b>')">`
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
                        <button class="btn btn-outline-primary btn-sm px-2 py-1" title="Lihat Foto" onclick="openImageModal('${esc(photo)}', '${esc(item.plate || 'Detail Deteksi')}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi: <b>${item.confidence_percent}%</b> · Status: <b>${esc(item.status)}</b>')">
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
    if (!confirm('Hapus histori deteksi ini beserta capture terkait?')) return;
    try {
        const result = await json(`/api/detections/${encodeURIComponent(detectionId)}`, { method: 'DELETE' });
        if (!result.success) throw new Error(result.message || 'Penghapusan gagal');
        await loadDetections();
    } catch (error) {
        console.error('Error deleteDetection:', error);
        alert(error.message || 'Gagal menghapus deteksi.');
    }
}
window.deleteDetection = deleteDetection;

async function initDetectionsPage() {
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

    const typeGroup = document.getElementById('detTypeButtonGroup');
    if (typeGroup) {
        typeGroup.addEventListener('click', e => {
            const btn = e.target.closest('button');
            if (!btn) return;
            typeGroup.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            detState.type = btn.dataset.type || 'all';
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

    await loadDetections();
}

function exportDetections() {
    window.location.href = '/api/export/detections';
}
window.exportDetections = exportDetections;


// ============================================================
// MODUL: RIWAYAT PLAT NOMOR (/history)
// ============================================================

let plateState = {
    page: 1,
    limit: 15,
    search: '',
    camera_id: '',
    status: 'all'
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
            status: plateState.status
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
    if (!confirm('Hapus riwayat plat ini beserta event dan capture terkait?')) return;
    try {
        const result = await json(`/api/plate/history/${encodeURIComponent(plateId)}`, { method: 'DELETE' });
        if (!result.success) throw new Error(result.message || 'Penghapusan gagal');
        await loadPlateHistory();
    } catch (error) {
        console.error('Error deletePlateHistory:', error);
        alert(error.message || 'Gagal menghapus riwayat plat.');
    }
}
window.deletePlateHistory = deletePlateHistory;

async function initHistoryPage() {
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

    await loadPlateHistory();
}

function exportPlates() {
    window.location.href = '/api/export/plates';
}
window.exportPlates = exportPlates;


// ============================================================
// MODUL: STATISTIK STANDAR PERUSAHAAN (ENTERPRISE ANALYTICS)
// ============================================================

window._currentStatsPeriod = 'today';
window._trendChart = null;
window._donutChart = null;
window._cameraChart = null;

async function loadEnterpriseStatistics(period = 'today') {
    window._currentStatsPeriod = period;

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
    }
}
window.loadEnterpriseStatistics = loadEnterpriseStatistics;

function exportStatsReport() {
    const period = window._currentStatsPeriod || 'today';
    window.location.href = `/api/export/statistics?period=${period}`;
}
window.exportStatsReport = exportStatsReport;

function initStatistics() {
    const periodGroup = document.getElementById('statsPeriodGroup');
    if (periodGroup) {
        periodGroup.addEventListener('click', e => {
            const btn = e.target.closest('button');
            if (!btn) return;
            periodGroup.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const period = btn.dataset.period || 'today';
            loadEnterpriseStatistics(period);
        });
    }

    loadEnterpriseStatistics('today');
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
        const compName = document.getElementById('companyName');
        const refInt = document.getElementById('refreshInterval');
        const sysTitle = document.getElementById('systemTitle');
        const confPlate = document.getElementById('minConfPlate');
        const confFace = document.getElementById('minConfFace');
        const streamUrl = document.getElementById('streamUrl');

        if (opName && settings.operator_name) opName.value = settings.operator_name;
        if (compName && settings.company_name) compName.value = settings.company_name;
        if (refInt && settings.refresh_interval) refInt.value = settings.refresh_interval;
        if (sysTitle && settings.system_title) sysTitle.value = settings.system_title;
        if (confPlate && settings.min_confidence_plate) confPlate.value = settings.min_confidence_plate;
        if (confFace && settings.min_confidence_face) confFace.value = settings.min_confidence_face;
        if (streamUrl && settings.stream_url) streamUrl.value = settings.stream_url;

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

        alert(res.message || 'Data simulasi berhasil ditambahkan!');
        await loadSettingsFromDb();
    } catch (e) {
        alert('Gagal menambahkan data simulasi: ' + e.message);
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
            company_name: document.getElementById('companyName')?.value,
            refresh_interval: document.getElementById('refreshInterval')?.value,
            system_title: document.getElementById('systemTitle')?.value,
            min_confidence_plate: document.getElementById('minConfPlate')?.value,
            min_confidence_face: document.getElementById('minConfFace')?.value,
            stream_url: document.getElementById('streamUrl')?.value
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
// INISIALISASI HALAMAN (ROUTING CLIENT-SIDE)
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
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
    if (document.getElementById('trendChartCanvas')) {
        initStatistics();
    }
    if (document.getElementById('settingsForm')) {
        initSettings();
    }
});
