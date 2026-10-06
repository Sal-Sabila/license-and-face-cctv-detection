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

    const periodId = startInputId.replace('StartDate', 'PeriodFilter');
    const periodSelect = document.getElementById(periodId);
    const selectedPeriod = periodSelect?.value || state.period || 'today';

    if (!startDate && !endDate) {
        state.start_date = '';
        state.end_date = '';
        state.period = selectedPeriod === 'custom' ? 'today' : selectedPeriod;
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

function updateCameraSummary(camerasList) {
    const totalEl = document.getElementById('summaryTotalCameras');
    const activeEl = document.getElementById('summaryActiveCameras');
    const inactiveEl = document.getElementById('summaryInactiveCameras');
    if (!totalEl && !activeEl && !inactiveEl) return;

    const total = camerasList.length;
    const active = camerasList.filter(c => Boolean(c.active)).length;
    const inactive = total - active;

    if (totalEl) totalEl.textContent = total;
    if (activeEl) activeEl.textContent = active;
    if (inactiveEl) inactiveEl.textContent = inactive;
}

function renderCameraRows(camerasList, totalCamerasCount = null) {
    const table = document.getElementById('cameraTable');
    if (!table) return;

    const total = totalCamerasCount !== null ? totalCamerasCount : camerasList.length;
    const pagInfo = document.getElementById('cameraPaginationInfo');
    if (pagInfo) {
        pagInfo.textContent = `Menampilkan ${camerasList.length} dari ${total} kamera`;
    }

    if (!camerasList.length) {
        table.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-5">Belum ada kamera yang sesuai.</td></tr>';
        return;
    }

    table.innerHTML = camerasList.map((camera, index) => {
        let dirBadge = '';
        if (camera.direction === 'entry') {
            dirBadge = '<span class="direction-badge dir-entry">Masuk</span>';
        } else if (camera.direction === 'exit') {
            dirBadge = '<span class="direction-badge dir-exit">Keluar</span>';
        } else {
            dirBadge = '<span class="direction-badge dir-unassigned">Belum diatur</span>';
        }

        const isAct = Boolean(camera.active);

        return `
        <tr>
            <td class="text-center text-muted col-num-text">${index + 1}</td>
            <td><strong class="camera-name">${esc(camera.name)}</strong></td>
            <td class="camera-url-cell"><span class="camera-url-code" title="${esc(camera.url)}">${esc(camera.url)}</span></td>
            <td class="text-center">${dirBadge}</td>
            <td class="text-center">
                <button type="button" role="switch" aria-checked="${isAct}" class="camera-toggle-switch ${isAct ? 'is-active' : ''}" data-action="toggle" data-id="${camera.id}" aria-label="${isAct ? 'Nonaktifkan kamera' : 'Aktifkan kamera'}" title="${isAct ? 'Nonaktifkan kamera' : 'Aktifkan kamera'}">
                    <span class="toggle-slider"></span>
                </button>
            </td>
            <td class="text-end pe-3">
                <div class="camera-actions-wrap justify-content-end">
                    <button type="button" class="btn-action-icon btn-action-edit" data-action="edit" data-id="${camera.id}" aria-label="Edit" title="Edit">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button type="button" class="btn-action-icon btn-action-delete" data-action="delete" data-id="${camera.id}" aria-label="Hapus" title="Hapus">
                        <i class="bi bi-trash3"></i>
                    </button>
                </div>
            </td>
        </tr>
    `;
    }).join('');
}

async function loadCameras() {
    try {
        const result = await json('/api/cameras');
        const cameras = result.data || [];
        const count = document.getElementById('cameraCountTitle');
        if (count) count.textContent = cameras.length;

        updateCameraSummary(cameras);

        const select = document.getElementById('cameraSelect');
        if (select) {
            select.innerHTML = cameras.map(item => `<option value="${item.id}">${esc(item.name)}</option>`).join('') || '<option value="">Belum ada kamera</option>';
        }

        const searchInput = document.getElementById('cameraSearchInput');
        const query = searchInput ? searchInput.value.trim().toLowerCase() : '';
        const displayList = query
            ? cameras.filter(c => (c.name || '').toLowerCase().includes(query) || (c.url || '').toLowerCase().includes(query))
            : cameras;

        renderCameraRows(displayList, cameras.length);
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
    document.getElementById('cameraNameInput').value = camera?.name || '';
    document.getElementById('cameraUrl').value = camera?.url || '';
    document.getElementById('cameraDirection').value = camera?.direction || 'unknown';
    document.getElementById('cameraActive').checked = camera ? camera.active : true;

    // Ganti judul modal sesuai mode
    const modalTitle = document.querySelector('#cameraModal .modal-title');
    if (modalTitle) modalTitle.textContent = camera ? 'Edit Kamera' : 'Tambah Kamera';

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

    const searchInput = document.getElementById('cameraSearchInput');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.trim().toLowerCase();
            const filtered = query
                ? cameras.filter(c => (c.name || '').toLowerCase().includes(query) || (c.url || '').toLowerCase().includes(query))
                : cameras;
            renderCameraRows(filtered, cameras.length);
        });
    }

    document.getElementById('cameraForm')?.addEventListener('submit', async event => {
        event.preventDefault();
        const id = document.getElementById('cameraId').value;
        const data = {
            name: document.getElementById('cameraNameInput').value,
            url: document.getElementById('cameraUrl').value,
            direction: document.getElementById('cameraDirection').value || 'unknown',
            active: document.getElementById('cameraActive').checked
        };
        try {
            await json(id ? `/api/cameras/${id}` : '/api/cameras', {
                method: id ? 'PUT' : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            bootstrap.Modal.getInstance(document.getElementById('cameraModal')).hide();
            cameras = await loadCameras();
            showNotification(id ? 'Kamera berhasil diperbarui.' : 'Kamera berhasil ditambahkan.', 'success');
        } catch (err) {
            console.error('Error simpan kamera:', err);
            showNotification('Gagal menyimpan kamera: ' + err.message, 'danger');
        }
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
// MODUL: DASHBOARD UTAMA & AKTIVITAS KENDARAAN MENCURIGAKAN
// ============================================================

let dashboardSuspiciousVehicleSummary = null;

function normalizeOccurrenceStatus(value) {
    return String(value || '').trim().toUpperCase().replaceAll('_', ' ');
}

function getFrequentOccurrenceItems(items) {
    return (Array.isArray(items) ? items : []).filter(item => {
        const count = Number(item?.occurrence_count) || 0;
        const status = normalizeOccurrenceStatus(item?.status);
        return count >= 2 && (status === 'SERING MUNCUL' || status === 'MENCURIGAKAN');
    });
}

function formatVehicleType(type) {
    const raw = String(type || '').toLowerCase().trim();
    if (raw === 'car') return 'Mobil';
    if (raw === 'motorcycle') return 'Sepeda Motor';
    if (raw === 'truck') return 'Truk';
    if (raw === 'bus') return 'Bus';
    if (raw === 'unknown' || !raw) return 'Kendaraan';
    return raw.charAt(0).toUpperCase() + raw.slice(1);
}

function vehicleIdentifier(item) {
    const rawPlate = String(item?.plate || '').trim();
    if (rawPlate && rawPlate !== '-') {
        return {
            text: rawPlate,
            isPlate: true
        };
    }
    const fallback = String(item?.identity || item?.object_group_id || 'Kendaraan').trim();
    return {
        text: fallback,
        isPlate: false
    };
}

function occurrencePhoto(item) {
    return item?.vehicle_image_path || item?.capture_path || item?.image_url || item?.image_path || item?.photo || '';
}

function formatOccurrenceTime(value) {
    if (!value) return 'Waktu tidak tersedia';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('id-ID', {
        dateStyle: 'medium',
        timeStyle: 'short',
    }).format(date);
}

async function initSuspiciousActivityPage() {
    const tableBody = document.getElementById('occurrenceTable');
    const pageSize = 20;
    let currentPage = 1;

    const [vehicleRes, camerasResult] = await Promise.all([
        json('/api/vehicle/occurrences?limit=200').catch(err => {
            console.error('Gagal memuat aktivitas kendaraan:', err);
            return { success: false, data: [] };
        }),
        json('/api/cameras').catch(error => {
            console.error('Gagal memuat daftar kamera:', error);
            return { success: false, data: [] };
        }),
    ]);

    const cameras = Array.isArray(camerasResult.data) ? camerasResult.data : [];
    const cameraSelect = document.getElementById('occurrenceCameraFilter');
    const statusFilter = document.getElementById('occurrenceStatusFilter');
    const periodFilter = document.getElementById('occurrencePeriodFilter');
    const searchInput = document.getElementById('occurrenceSearch');
    const customRange = document.getElementById('occurrenceCustomRange');
    const startDateInput = document.getElementById('occurrenceStartDate');
    const endDateInput = document.getElementById('occurrenceEndDate');
    const dateApplyButton = document.getElementById('occurrenceDateApply');
    const errorState = document.getElementById('occurrenceErrorState');
    const emptyState = document.getElementById('occurrenceEmptyState');
    const tableCard = document.getElementById('occurrenceTableCard');
    const prevButton = document.getElementById('occurrencePrevBtn');
    const nextButton = document.getElementById('occurrenceNextBtn');
    const pageNumber = document.getElementById('occurrencePageNumber');
    const paginationInfo = document.getElementById('occurrencePaginationInfo');
    let appliedStartDate = '';
    let appliedEndDate = '';

    if (cameraSelect) {
        cameraSelect.innerHTML = '<option value="">Semua Kamera</option>' +
            cameras.map(camera => `<option value="${esc(camera.id)}">${esc(camera.name || `CCTV ${camera.id}`)}</option>`).join('');
    }

    const hasErrors = vehicleRes.success === false;
    const rawItems = Array.isArray(vehicleRes.data) ? vehicleRes.data : [];
    const allItems = getFrequentOccurrenceItems(rawItems);

    function cameraNameFor(item) {
        if (item.camera_name && String(item.camera_name).trim()) return item.camera_name;
        const camera = cameras.find(entry => String(entry.id) === String(item.camera_id));
        return camera?.name || (item.camera_id === null || item.camera_id === undefined || item.camera_id === ''
            ? 'Kamera tidak diketahui'
            : `CCTV ${item.camera_id}`);
    }

    function renderOccurrenceRow(item, index) {
        const status = normalizeOccurrenceStatus(item.status);
        const statusClass = status === 'MENCURIGAKAN' ? 'occurrence-status-suspicious' : 'occurrence-status-frequent';
        const photo = occurrencePhoto(item);
        const safePhoto = photo ? buildCaptureUrl(photo) : '';
        const ident = vehicleIdentifier(item);
        const photoHtml = safePhoto
            ? `<img src="${esc(safePhoto)}" alt="Capture Kendaraan" class="det-thumb-img" onerror="this.onerror=null;this.parentElement.innerHTML='<div class=\\'det-thumb-placeholder\\'><i class=\\'bi bi-camera-video-off\\'></i></div>';">`
            : `<div class="det-thumb-placeholder" title="Foto tidak tersedia"><i class="bi bi-camera-video-off"></i></div>`;
        const lastSeen = formatOccurrenceTime(item.last_seen);
        const targetHtml = ident.isPlate
            ? `<span class="target-val-plate">${esc(ident.text)}</span>`
            : `<span class="text-muted">${esc(ident.text)}</span>`;

        return `
            <tr class="occurrence-row" data-occurrence-index="${index}" title="Klik untuk melihat detail kendaraan">
                <td class="col-det-photo">${photoHtml}</td>
                <td class="col-det-target">${targetHtml}</td>
                <td><span class="det-type-pill det-type-accent">${esc(formatVehicleType(item.vehicle_type))}</span></td>
                <td><strong>${Number(item.occurrence_count) || 0}x</strong></td>
                <td><div class="det-cctv-name" title="${esc(cameraNameFor(item))}">${esc(cameraNameFor(item))}</div></td>
                <td><div class="det-time-date">${esc(lastSeen)}</div></td>
                <td><span class="occurrence-status ${statusClass}">${esc(status)}</span></td>
                <td class="text-end pe-3">
                    <button type="button" class="btn-action-icon" data-occurrence-index="${index}"
                        title="Lihat detail kendaraan" aria-label="Lihat detail kendaraan">
                        <i class="bi bi-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    }

    function filteredItems() {
        const selectedStatus = statusFilter?.value || 'all';
        const selectedCamera = cameraSelect?.value || '';
        const selectedPeriod = periodFilter?.value || 'all';
        const search = (searchInput?.value || '').trim().toLowerCase();

        return allItems.filter(item => {
            const status = normalizeOccurrenceStatus(item.status).replaceAll(' ', '_');
            if (Number(item.occurrence_count) < 2 || !['SERING_MUNCUL', 'MENCURIGAKAN'].includes(status)) return false;
            if (selectedStatus !== 'all' && status !== selectedStatus) return false;
            if (selectedCamera && String(item.camera_id) !== selectedCamera) return false;

            if (selectedPeriod !== 'all') {
                const seen = new Date(item.last_seen);
                if (Number.isNaN(seen.getTime())) return false;
                const localDate = `${seen.getFullYear()}-${String(seen.getMonth() + 1).padStart(2, '0')}-${String(seen.getDate()).padStart(2, '0')}`;
                const dateRange = selectedPeriod === 'custom'
                    ? { start_date: appliedStartDate, end_date: appliedEndDate }
                    : getDateRange(selectedPeriod);
                if (!dateRange.start_date || !dateRange.end_date) return false;
                if (localDate < dateRange.start_date || localDate > dateRange.end_date) return false;
            }

            if (search) {
                const ident = vehicleIdentifier(item);
                const searchValues = [
                    ident.text,
                    item.plate,
                    item.identity,
                    item.vehicle_type,
                    formatVehicleType(item.vehicle_type),
                    cameraNameFor(item),
                ];
                if (!searchValues.some(value => String(value || '').toLowerCase().includes(search))) return false;
            }
            return true;
        });
    }

    function renderTable() {
        const rows = filteredItems();
        const total = rows.length;
        const totalPages = Math.max(1, Math.ceil(total / pageSize));
        currentPage = Math.min(currentPage, totalPages);
        const start = (currentPage - 1) * pageSize;
        const pageRows = rows.slice(start, start + pageSize);

        if (errorState) errorState.hidden = !hasErrors;
        if (emptyState) emptyState.hidden = total !== 0 || hasErrors;
        if (tableCard) tableCard.hidden = total === 0;
        if (tableBody) {
            tableBody.innerHTML = pageRows.map((item, index) => renderOccurrenceRow(item, start + index)).join('');
        }
        if (paginationInfo) {
            paginationInfo.textContent = total
                ? `Menampilkan ${start + 1}–${start + pageRows.length} dari ${total} kendaraan`
                : 'Menampilkan 0 dari 0 kendaraan';
        }
        if (pageNumber) {
            const pages = [];
            const low = Math.max(1, currentPage - 1);
            const high = Math.min(totalPages, low + 3);
            const first = Math.max(1, high - 3);
            if (first > 1) pages.push('<span class="occurrence-page-ellipsis">…</span>');
            for (let page = first; page <= high; page += 1) {
                pages.push(`<button type="button" class="btn-page-nav occurrence-page-number${page === currentPage ? ' active' : ''}" data-page="${page}" aria-label="Halaman ${page}" ${page === currentPage ? 'aria-current="page"' : ''}>${page}</button>`);
            }
            if (high < totalPages) pages.push('<span class="occurrence-page-ellipsis">…</span>');
            pageNumber.innerHTML = pages.join('');
        }
        if (prevButton) prevButton.disabled = currentPage <= 1;
        if (nextButton) nextButton.disabled = currentPage >= totalPages;
    }

    function showOccurrenceDetail(item) {
        const modal = document.getElementById('occurrenceDetailModal');
        const title = document.getElementById('occurrenceDetailTitle');
        const body = document.getElementById('occurrenceDetailBody');
        if (!modal || !title || !body) return;

        const ident = vehicleIdentifier(item);
        const status = normalizeOccurrenceStatus(item.status);
        const statusClass = status === 'MENCURIGAKAN' ? 'occurrence-status-suspicious' : 'occurrence-status-frequent';
        const photo = occurrencePhoto(item);
        const safePhoto = photo ? buildCaptureUrl(photo) : '';

        title.textContent = ident.isPlate ? `Detail Kendaraan · ${ident.text}` : 'Detail Kendaraan';

        const imageHtml = safePhoto
            ? `<div class="occurrence-detail-photo-wrap mb-3 text-center">
                   <img src="${esc(safePhoto)}" alt="Capture Kendaraan" class="occurrence-detail-photo img-fluid rounded border" onerror="this.onerror=null;this.parentElement.innerHTML='<div class=\\'occurrence-detail-placeholder mb-3 py-4 text-center rounded border text-muted\\'><i class=\\'bi bi-camera-video-off fs-1 d-block mb-1\\'></i><span>Foto kendaraan tidak tersedia</span></div>';">
               </div>`
            : `<div class="occurrence-detail-placeholder mb-3 py-4 text-center rounded border text-muted">
                   <i class="bi bi-camera-video-off fs-1 d-block mb-1"></i>
                   <span>Foto kendaraan tidak tersedia</span>
               </div>`;

        const plateDisplay = ident.isPlate ? ident.text : (item.identity || '-');
        const vehicleTypeDisplay = formatVehicleType(item.vehicle_type);
        const occurrenceDisplay = `${Number(item.occurrence_count) || 0}x`;
        const lastCameraDisplay = cameraNameFor(item);
        const lastSeenDisplay = formatOccurrenceTime(item.last_seen);

        let historyHtml = '';
        if (Array.isArray(item.history) && item.history.length > 0) {
            const historyList = item.history.map((h, i) => {
                const camName = h.camera_name || (cameras.find(c => String(c.id) === String(h.camera_id))?.name) || `CCTV ${h.camera_id}`;
                const timeStr = formatOccurrenceTime(h.seen_at);
                return `<li class="occurrence-history-item"><span class="occurrence-history-index">${i + 1}.</span> <span class="occurrence-history-time">${esc(timeStr)}</span> — <span class="occurrence-history-cctv">${esc(camName)}</span></li>`;
            }).join('');
            historyHtml = `
                <div class="occurrence-detail-history mt-3 pt-3 border-top">
                    <h6 class="occurrence-history-title mb-2"><i class="bi bi-clock-history me-1"></i>Riwayat Kemunculan Kendaraan</h6>
                    <ul class="occurrence-history-list list-unstyled mb-0">${historyList}</ul>
                </div>
            `;
        }

        body.innerHTML = `
            ${imageHtml}
            <div class="occurrence-detail-info-grid">
                <div class="occurrence-info-row">
                    <span class="occurrence-info-label">Plat:</span>
                    <strong class="occurrence-info-val ${ident.isPlate ? 'target-val-plate' : ''}">${esc(plateDisplay)}</strong>
                </div>
                <div class="occurrence-info-row">
                    <span class="occurrence-info-label">Jenis kendaraan:</span>
                    <span class="occurrence-info-val">${esc(vehicleTypeDisplay)}</span>
                </div>
                <div class="occurrence-info-row">
                    <span class="occurrence-info-label">Jumlah kemunculan:</span>
                    <span class="occurrence-info-val"><strong>${esc(occurrenceDisplay)}</strong></span>
                </div>
                <div class="occurrence-info-row">
                    <span class="occurrence-info-label">Status:</span>
                    <span class="occurrence-info-val"><span class="occurrence-status ${statusClass}">${esc(status)}</span></span>
                </div>
                <div class="occurrence-info-row">
                    <span class="occurrence-info-label">Kamera terakhir:</span>
                    <span class="occurrence-info-val">${esc(lastCameraDisplay)}</span>
                </div>
                <div class="occurrence-info-row">
                    <span class="occurrence-info-label">Terakhir terlihat:</span>
                    <span class="occurrence-info-val">${esc(lastSeenDisplay)}</span>
                </div>
            </div>
            ${historyHtml}
        `;

        bootstrap.Modal.getOrCreateInstance(modal).show();
    }

    [statusFilter, cameraSelect, periodFilter].forEach(control => {
        control?.addEventListener('change', () => {
            currentPage = 1;
            if (control === periodFilter && customRange) {
                customRange.hidden = periodFilter.value !== 'custom';
            }
            renderTable();
        });
    });

    dateApplyButton?.addEventListener('click', () => {
        const startDate = startDateInput?.value || '';
        const endDate = endDateInput?.value || '';
        if (!startDate || !endDate || startDate > endDate) {
            showNotification('Pilih rentang tanggal yang valid.', 'warning');
            return;
        }
        appliedStartDate = startDate;
        appliedEndDate = endDate;
        currentPage = 1;
        renderTable();
    });

    searchInput?.addEventListener('input', () => {
        currentPage = 1;
        renderTable();
    });

    prevButton?.addEventListener('click', () => {
        currentPage -= 1;
        renderTable();
    });

    nextButton?.addEventListener('click', () => {
        currentPage += 1;
        renderTable();
    });

    pageNumber?.addEventListener('click', event => {
        const button = event.target.closest('[data-page]');
        if (!button) return;
        currentPage = Number(button.dataset.page) || 1;
        renderTable();
    });

    tableBody?.addEventListener('click', event => {
        const row = event.target.closest('[data-occurrence-index]');
        if (!row) return;
        const index = Number(row.dataset.occurrenceIndex);
        const item = filteredItems()[index];
        if (item) showOccurrenceDetail(item);
    });

    renderTable();
}

async function refreshDashboardSuspiciousSummary() {
    const total = document.getElementById('suspiciousActivityTotal');
    const subtitle = document.getElementById('suspiciousActivitySub');

    try {
        const res = await json('/api/vehicle/occurrences?limit=200');
        if (res?.success === false || !Array.isArray(res?.data)) {
            dashboardSuspiciousVehicleSummary = null;
            if (total) total.textContent = '—';
            if (subtitle) subtitle.textContent = 'Data tidak tersedia';
            return;
        }

        const vehicles = getFrequentOccurrenceItems(res.data);
        const suspiciousVehicles = vehicles.filter(
            item => normalizeOccurrenceStatus(item.status) === 'MENCURIGAKAN'
        );

        let topVehicle = null;
        if (vehicles.length > 0) {
            topVehicle = vehicles.reduce((prev, curr) =>
                (Number(curr.occurrence_count) > Number(prev.occurrence_count)) ? curr : prev
            );
        }

        dashboardSuspiciousVehicleSummary = {
            frequentCount: vehicles.length,
            suspiciousCount: suspiciousVehicles.length,
            topVehicle: topVehicle,
        };

        if (total) total.textContent = String(suspiciousVehicles.length);
        if (subtitle) subtitle.textContent = 'kendaraan';

        renderDashboardSuspiciousInsight();
    } catch (e) {
        console.error('Gagal memuat summary kendaraan mencurigakan:', e);
        dashboardSuspiciousVehicleSummary = null;
        if (total) total.textContent = '—';
        if (subtitle) subtitle.textContent = 'Data tidak tersedia';
    }
}

function renderDashboardSuspiciousInsight() {
    const insight = document.getElementById('dashboardInsights');
    if (!insight || !dashboardSuspiciousVehicleSummary) return;

    const { frequentCount, topVehicle } = dashboardSuspiciousVehicleSummary;
    let existingHtml = insight.innerHTML;

    // Build vehicle insight elements
    const vehicleInsightHtml = frequentCount > 0
        ? `<li id="suspiciousActivityInsight"><a href="/aktivitas-mencurigakan">${frequentCount} kendaraan dengan frekuensi kemunculan tinggi terdeteksi.</a></li>`
        : '';
    const topVehicleHtml = topVehicle
        ? `<li id="topVehicleInsight">Kendaraan <strong>${esc(vehicleIdentifier(topVehicle).text)}</strong> merupakan kendaraan dengan kemunculan tertinggi.</li>`
        : '';

    // Replace or insert into dashboard insights
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = existingHtml;

    // Remove old dynamic vehicle/occurrence li tags
    const oldSuspicious = tempDiv.querySelector('#suspiciousActivityInsight');
    if (oldSuspicious) oldSuspicious.remove();
    const oldTop = tempDiv.querySelector('#topVehicleInsight');
    if (oldTop) oldTop.remove();

    // Filter out any li with person/orang
    Array.from(tempDiv.querySelectorAll('li')).forEach(li => {
        const txt = li.textContent.toLowerCase();
        if (txt.includes('orang') || txt.includes('person') || txt.includes('wajah')) {
            li.remove();
        }
    });

    let newContent = '';
    if (vehicleInsightHtml) newContent += vehicleInsightHtml;
    if (topVehicleHtml) newContent += topVehicleHtml;
    newContent += tempDiv.innerHTML;

    insight.innerHTML = newContent || '<li>Belum ada data cukup untuk insight kendaraan.</li>';
}

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

            // Kalkulasi Bar Proporsi Masuk / Keluar (Ringkasan Arus)
            const vIn = Number(data.vehicle_entry) || 0;
            const vOut = Number(data.vehicle_exit) || 0;
            const vSum = vIn + vOut;
            const vInPct = vSum > 0 ? (vIn / vSum) * 100 : 0;
            const vOutPct = vSum > 0 ? (vOut / vSum) * 100 : 0;

            const elVBarIn = document.getElementById('vehicleFlowBarIn');
            const elVBarOut = document.getElementById('vehicleFlowBarOut');
            if (elVBarIn) elVBarIn.style.width = `${vInPct}%`;
            if (elVBarOut) elVBarOut.style.width = `${vOutPct}%`;

            const elVEntryText = document.getElementById('vehicleFlowEntryText');
            const elVExitText = document.getElementById('vehicleFlowExitText');
            if (elVEntryText) elVEntryText.textContent = `Masuk ${vIn}`;
            if (elVExitText) elVExitText.textContent = `Keluar ${vOut}`;

            const pIn = Number(data.people_entry) || 0;
            const pOut = Number(data.people_exit) || 0;
            const pSum = pIn + pOut;
            const pInPct = pSum > 0 ? (pIn / pSum) * 100 : 0;
            const pOutPct = pSum > 0 ? (pOut / pSum) * 100 : 0;

            const elPBarIn = document.getElementById('peopleFlowBarIn');
            const elPBarOut = document.getElementById('peopleFlowBarOut');
            if (elPBarIn) elPBarIn.style.width = `${pInPct}%`;
            if (elPBarOut) elPBarOut.style.width = `${pOutPct}%`;

            const elPEntryText = document.getElementById('peopleFlowEntryText');
            const elPExitText = document.getElementById('peopleFlowExitText');
            if (elPEntryText) elPEntryText.textContent = `Masuk ${pIn}`;
            if (elPExitText) elPExitText.textContent = `Keluar ${pOut}`;

            const insight = document.getElementById('dashboardInsights');
            if (insight) {
                const insightData = summaryRes.data?.insights || [];
                const insightItems = [];
                if (dashboardSuspiciousVehicleSummary) {
                    const { frequentCount, topVehicle } = dashboardSuspiciousVehicleSummary;
                    if (frequentCount > 0) {
                        insightItems.push(`<li id="suspiciousActivityInsight"><a href="/aktivitas-mencurigakan">${frequentCount} kendaraan dengan frekuensi kemunculan tinggi terdeteksi.</a></li>`);
                    }
                    if (topVehicle) {
                        insightItems.push(`<li id="topVehicleInsight">Kendaraan <strong>${esc(vehicleIdentifier(topVehicle).text)}</strong> merupakan kendaraan dengan kemunculan tertinggi.</li>`);
                    }
                }
                insightData.forEach(item => {
                    const txt = String(item).toLowerCase();
                    if (!txt.includes('orang') && !txt.includes('person') && !txt.includes('wajah')) {
                        insightItems.push(`<li>${esc(item)}</li>`);
                    }
                });
                insight.innerHTML = insightItems.join('') || '<li>Belum ada data cukup untuk insight kendaraan.</li>';
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

    let lastOccurrenceRefresh = 0;
    await refreshDashboardSuspiciousSummary();
    lastOccurrenceRefresh = Date.now();

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
        return `/api/video_feed/${camId}?bbox=1`;
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
        const videoFileCol = document.getElementById('videoFileCol');
        if (videoFileCol) videoFileCol.style.display = videoMode ? 'block' : 'none';
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
                        processedVideo.play().catch(() => { });
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
                    const detectionId = item.detection_id || item.id || '';
                    const eventKey = item.event_key || '';
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

                    const isSuccess = item.status === 'Terbaca';
                    const statusPillClass = isSuccess ? 'recent-status-success' : 'recent-status-warning';

                    return `
                        <div class="recent recent-row" 
                             data-detection-id="${detectionId}"
                             data-event-key="${esc(eventKey)}"
                             onclick="openImageModal('${photoPath}', '${label}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi AI: <b>${item.confidence_percent}%</b> · Status: <b>${esc(item.status)}</b>', '${item.object_type}')">
                            <div class="recent-badge-col">
                                <div class="recent-badge">${icon}</div>
                            </div>
                            <div class="recent-info-col">
                                <strong class="recent-type-label">${label}</strong>
                                <span class="recent-meta-label">${esc(item.camera)} · ${esc(item.timestamp)}</span>
                            </div>
                            <div class="recent-status-col">
                                <span class="recent-conf-rate">${item.confidence_percent}%</span>
                                <span class="recent-status-pill ${statusPillClass}">${esc(item.status)}</span>
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
                if (Date.now() - lastOccurrenceRefresh >= 15000) {
                    await refreshDashboardSuspiciousSummary();
                    lastOccurrenceRefresh = Date.now();
                }
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
            const detectionId = item.detection_id || item.id || '';
            const eventKey = item.event_key || '';
            const isVehicle = item.object_type === 'vehicle' || item.type === 'vehicle' || item.type === 'vehicle_with_plate' || item.type === 'plate';
            const isPlate = item.type === 'vehicle_with_plate' || (isVehicle && item.has_plate);

            let targetLabel = '';
            if (isPlate) {
                targetLabel = `<span class="target-val-plate">${esc(item.plate || 'Tanpa Plat')}</span>`;
            } else if (isVehicle) {
                targetLabel = `<span class="target-val-vehicle">Kendaraan</span>`;
            } else {
                targetLabel = `<span class="target-val-person">Orang</span>`;
            }

            let typeBadge = '';
            if (isPlate) {
                typeBadge = `<span class="det-type-pill det-type-accent"><i class="bi bi-card-heading"></i> Plat Nomor</span>`;
            } else if (isVehicle) {
                typeBadge = `<span class="det-type-pill det-type-accent"><i class="bi bi-car-front"></i> Kendaraan</span>`;
            } else {
                typeBadge = `<span class="det-type-pill det-type-neutral"><i class="bi bi-person"></i> Orang</span>`;
            }

            let statusBadge = '';
            if (item.status_code === 1) {
                statusBadge = `<span class="det-status-pill status-terbaca">Terbaca</span>`;
            } else if (item.status_code === 2) {
                statusBadge = `<span class="det-status-pill status-perlu-cek">Perlu Cek</span>`;
            } else {
                statusBadge = `<span class="det-status-pill status-gagal">Gagal</span>`;
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
                ? `<img src="/${esc(photo.replace(/^\/+/, ''))}" alt="Thumb" class="det-thumb-img" onerror="this.onerror=null;this.parentElement.innerHTML='<div class=\\'det-thumb-placeholder\\'><i class=\\'bi bi-image\\'></i></div>';" onclick="openImageModal('${esc(photo)}', '${esc(item.plate || 'Deteksi')}', '${esc(item.camera)} · ${esc(item.timestamp)}', '${esc(confidenceText)}')">`
                : `<div class="det-thumb-placeholder"><i class="bi bi-image"></i></div>`;

            const confVal = Math.round(item.confidence_percent || 0);
            const confClass = confVal >= 70 ? 'conf-high' : (confVal >= 40 ? 'conf-mid' : 'conf-low');

            let timeHtml = esc(item.timestamp || '-');
            if (item.timestamp && item.timestamp.includes(' ')) {
                const parts = item.timestamp.split(' ');
                timeHtml = `<div class="det-time-date">${esc(parts[0])}</div><div class="det-time-clock">${esc(parts.slice(1).join(' '))}</div>`;
            }

            return `
                <tr data-detection-id="${detectionId}" data-event-key="${esc(eventKey)}">
                    <td class="col-det-photo">${thumbHtml}</td>
                    <td class="col-det-target">${targetLabel}</td>
                    <td class="col-det-type">${typeBadge}</td>
                    <td class="col-det-cctv"><div class="det-cctv-name" title="${esc(item.camera || '-')}">${esc(item.camera || '-')}</div></td>
                    <td class="col-det-conf">
                        <div class="det-conf-container">
                            <div class="det-conf-track">
                                <div class="det-conf-fill ${confClass}" style="width: ${confVal}%;"></div>
                            </div>
                            <span class="det-conf-num">${confVal}%</span>
                        </div>
                    </td>
                    <td class="col-det-time">${timeHtml}</td>
                    <td class="col-det-status">${statusBadge}</td>
                    <td class="col-det-action text-end pe-3">
                        <div class="camera-actions-wrap justify-content-end">
                            <button type="button" class="btn-action-icon" title="Lihat detail" aria-label="Lihat detail" onclick="openImageModal('${esc(photo)}', '${esc(item.plate || 'Detail Deteksi')}', '${esc(item.camera)} · ${esc(item.timestamp)}', '${esc(confidenceText)} · Status: ${esc(item.status)}')">
                                <i class="bi bi-eye"></i>
                            </button>
                            <button type="button" class="btn-action-icon btn-action-delete" title="Hapus" aria-label="Hapus" onclick="deleteDetection(${detectionId})">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
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
    } catch (e) { }

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

        const typeGroup = document.getElementById('detTypeButtonGroup');
        if (typeGroup) {
            typeGroup.querySelectorAll('button').forEach(b => {
                b.classList.toggle('active', (b.dataset.type || 'all') === 'all');
            });
        }

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
            const plateId = item.plate_id || item.id || '';
            const detectionId = item.detection_id || '';
            const eventKey = item.event_key || '';

            let statusBadge = '';
            if (item.status_code === 1) {
                statusBadge = `<span class="det-status-pill status-terbaca">Terbaca</span>`;
            } else if (item.status_code === 2) {
                statusBadge = `<span class="det-status-pill status-perlu-cek">Perlu Cek</span>`;
            } else {
                statusBadge = `<span class="det-status-pill status-gagal">Gagal</span>`;
            }

            const photo = item.image_path || '';
            const thumbHtml = photo
                ? `<img src="/${esc(photo.replace(/^\/+/, ''))}" alt="Plat" class="plate-crop-img" onclick="openImageModal('${esc(photo)}', 'Plat: ${esc(item.plate)}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Akurasi OCR: <b>${item.confidence_percent}%</b>')">`
                : `<div class="plate-crop-placeholder" title="Tidak ada foto"><i class="bi bi-card-heading"></i></div>`;

            const confClass = item.confidence_percent >= 70 ? 'conf-high' : 'conf-mid';

            const timeParts = (item.timestamp || '').split(' ');
            const timeHtml = timeParts.length >= 2
                ? `<div class="det-time-date">${esc(timeParts[0])}</div><div class="det-time-clock">${esc(timeParts.slice(1).join(' '))}</div>`
                : `<div class="det-time-date">${esc(item.timestamp || '-')}</div>`;

            return `
                <tr data-plate-id="${plateId}" data-detection-id="${detectionId}" data-event-key="${esc(eventKey)}">
                    <td>${thumbHtml}</td>
                    <td><span class="target-val-plate">${esc(item.plate)}</span></td>
                    <td><div class="det-cctv-name" title="${esc(item.camera)}">${esc(item.camera)}</div></td>
                    <td>
                        <div class="det-conf-container">
                            <div class="det-conf-track">
                                <div class="det-conf-fill ${confClass}" style="width: ${item.confidence_percent}%;"></div>
                            </div>
                            <span class="det-conf-num">${item.confidence_percent}%</span>
                        </div>
                    </td>
                    <td>${timeHtml}</td>
                    <td>${statusBadge}</td>
                    <td class="text-end pe-3">
                        <div class="camera-actions-wrap justify-content-end">
                            <button type="button" class="btn-action-icon" title="Lihat foto" aria-label="Lihat foto" onclick="openImageModal('${esc(photo)}', 'Plat Nomor: ${esc(item.plate)}', '${esc(item.camera)} · ${esc(item.timestamp)}', 'Confidence OCR: <b>${item.confidence_percent}%</b> · Status: <b>${esc(item.status)}</b>')">
                                <i class="bi bi-eye"></i>
                            </button>
                            <button type="button" class="btn-action-icon btn-action-delete" title="Hapus" aria-label="Hapus" onclick="deletePlateHistory(${plateId})">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
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
    } catch (e) { }

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
    select.innerHTML = '<option value="">Semua Kamera</option>' + (res.data || []).map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
}

// Formatter angka untuk rekapitulasi (tabular-nums format Indonesia)
function formatRecapNumber(val) {
    if (val === null || val === undefined || isNaN(Number(val))) return '0';
    return Number(val).toLocaleString('id-ID');
}

// Kartu metrik rekapitulasi (mengikuti gaya kartu metrik Dashboard)
function renderRecapCard(label, value, subHtml = '', isTotal = false) {
    const totalClass = isTotal ? ' recap-card-total' : '';
    const subContent = subHtml || '<span class="recap-sub-placeholder">&nbsp;</span>';
    return `
        <div class="recap-metric-col">
            <div class="recap-metric-card${totalClass}">
                <div class="recap-card-content">
                    <span class="recap-card-label">${esc(label)}</span>
                    <strong class="recap-card-value">${formatRecapNumber(value)}</strong>
                    <div class="recap-card-sub">${subContent}</div>
                </div>
            </div>
        </div>`;
}

// Wrapper backward compatibility untuk renderMiniStatCard
function renderMiniStatCard(label, value, extra) {
    const isTotal = String(label).toLowerCase().includes('total');
    let sub = '';
    if (extra) {
        const clean = extra.replace(/[\u2191\u2193]/g, '').trim();
        const icon = (extra.includes('\u2191') || extra.toLowerCase().includes('dalam')) ? 'bi-arrow-down-left' : 'bi-arrow-up-right';
        sub = `<i class="bi ${icon}"></i> ${esc(clean)}`;
    }
    return renderRecapCard(label, value, sub, isTotal);
}

// Helper badge arah untuk tabel Rekap CCTV
function renderRecapDirectionBadge(dir) {
    const d = (dir || '').trim();
    const lower = d.toLowerCase();
    if (lower === 'masuk' || lower.includes('in')) {
        return `<span class="badge-direction badge-dir-in">${esc(d || 'Masuk')}</span>`;
    } else if (lower === 'keluar' || lower.includes('out')) {
        return `<span class="badge-direction badge-dir-out">${esc(d || 'Keluar')}</span>`;
    }
    return `<span class="badge-direction badge-dir-neutral">${esc(d || 'Tidak ditentukan')}</span>`;
}

// Helper titik status (7px) untuk tabel Rekap CCTV
function renderRecapStatusCell(status) {
    const s = (status || '').trim();
    const lower = s.toLowerCase();
    let dotClass = 'dot-offline';
    let textClass = 'recap-status-text-muted';

    if (lower === 'aktif' || lower === 'active' || lower === 'online') {
        dotClass = 'dot-active';
        textClass = 'recap-status-text-normal';
    }

    return `
        <span class="recap-status-cell">
            <span class="recap-status-dot ${dotClass}"></span>
            <span class="${textClass}">${esc(s || '-')}</span>
        </span>`;
}

// Renderer baris tabel Rekap CCTV
function renderRecapCameraRow(c) {
    return `
        <tr>
            <td class="text-start"><span class="recap-camera-name">${esc(c.camera || '-')}</span></td>
            <td class="text-start">${renderRecapDirectionBadge(c.direction)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(c.vehicles)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(c.people)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(c.unique_plates)}</td>
            <td class="text-start">${renderRecapStatusCell(c.status)}</td>
        </tr>`;
}

// Renderer baris tabel Total per Hari
function renderRecapDailyRow(d) {
    return `
        <tr>
            <td class="text-start"><span class="recap-date-cell">${esc(d.date || '-')}</span></td>
            <td class="text-end tabular-nums">${formatRecapNumber(d.vehicles)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(d.unique_plates)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(d.entry)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(d.exit)}</td>
            <td class="text-end tabular-nums">${formatRecapNumber(d.people)}</td>
        </tr>`;
}

// State pencarian & data cache untuk Rekapitulasi
window._lastRecapData = null;
let _recapSearchQuery = '';

function renderRecapTables() {
    const cameraTable = document.getElementById('recapCameraTable');
    const dailyTable = document.getElementById('recapDailyTable');
    if (!window._lastRecapData) return;

    const data = window._lastRecapData;
    const q = (_recapSearchQuery || '').toLowerCase().trim();

    const cameras = data.cameras || [];
    const daily = data.daily || [];

    const filteredCameras = q ? cameras.filter(c => {
        const cam = (c.camera || '').toLowerCase();
        const dir = (c.direction || '').toLowerCase();
        const st = (c.status || '').toLowerCase();
        return cam.includes(q) || dir.includes(q) || st.includes(q);
    }) : cameras;

    const filteredDaily = q ? daily.filter(d => {
        const dateStr = (d.date || '').toLowerCase();
        return dateStr.includes(q);
    }) : daily;

    if (cameraTable) {
        cameraTable.innerHTML = filteredCameras.map(renderRecapCameraRow).join('') ||
            `<tr><td colspan="6" class="text-center text-muted py-4">${q ? `Tidak ada data CCTV yang cocok dengan "${esc(_recapSearchQuery)}".` : 'Belum ada event pada filter ini.'}</td></tr>`;
    }

    if (dailyTable) {
        dailyTable.innerHTML = filteredDaily.map(renderRecapDailyRow).join('') ||
            `<tr><td colspan="6" class="text-center text-muted py-4">${q ? `Tidak ada data harian yang cocok dengan "${esc(_recapSearchQuery)}".` : 'Belum ada data harian.'}</td></tr>`;
    }
}

// Kartu rincian untuk halaman Statistik - menggunakan ulang component/class
// card statistik yang sudah ada di baris atas (Total Deteksi / Deteksi Plat /
// Wajah-Pengendara / Ketersediaan CCTV), bukan model box besar.
function renderStatsKpiCard(iconVariant, iconName, label, value, descHtml) {
    return `
        <div class="col-xl-3 col-md-6 col-sm-6 col-12">
            <div class="stat-card p-3 h-100 shadow-sm">
                <div class="stat-icon ${iconVariant}"><i class="bi ${iconName}"></i></div>
                <div class="w-100">
                    <span class="text-muted small">${label}</span>
                    <strong class="fs-4 d-block my-1">${value || 0}</strong>
                    ${descHtml}
                </div>
            </div>
        </div>`;
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
        window._lastRecapData = data;
        const s = data.summary || {};

        const totals = document.getElementById('recapTotals');
        if (totals) {
            totals.innerHTML = `
                <div class="recap-metrics-section">
                    <div class="recap-group-block">
                        <div class="recap-group-label">Kendaraan</div>
                        <div class="recap-cards-grid">
                            ${renderRecapCard('Kendaraan masuk', s.vehicle_entry, '<i class="bi bi-arrow-down-left"></i> Arah ke dalam')}
                            ${renderRecapCard('Kendaraan keluar', s.vehicle_exit, '<i class="bi bi-arrow-up-right"></i> Arah ke luar')}
                            ${renderRecapCard('Total kendaraan', s.vehicles, '', true)}
                        </div>
                    </div>
                    <div class="recap-group-block">
                        <div class="recap-group-label">Orang</div>
                        <div class="recap-cards-grid">
                            ${renderRecapCard('Orang masuk', s.people_entry, '<i class="bi bi-arrow-down-left"></i> Arah ke dalam')}
                            ${renderRecapCard('Orang keluar', s.people_exit, '<i class="bi bi-arrow-up-right"></i> Arah ke luar')}
                            ${renderRecapCard('Total orang', s.people, '', true)}
                        </div>
                    </div>
                </div>`;
        }

        renderRecapTables();
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

    // Pencarian instan ter-debounce
    const searchInput = document.getElementById('recapSearch');
    let searchTimer = null;
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimer);
            searchTimer = setTimeout(() => {
                _recapSearchQuery = searchInput.value.trim();
                renderRecapTables();
            }, 200);
        });
    }

    // Pergantian periode cepat
    const periodSelect = document.getElementById('recapPeriod');
    if (periodSelect) {
        periodSelect.addEventListener('change', () => {
            if (periodSelect.value !== 'custom') {
                const startInput = document.getElementById('recapStart');
                const endInput = document.getElementById('recapEnd');
                if (startInput) startInput.value = '';
                if (endInput) endInput.value = '';
            }
        });
    }

    // Tombol Terapkan
    document.getElementById('recapApply')?.addEventListener('click', () => {
        loadRecap().catch(err => {
            if (err.message) showNotification(err.message, 'warning');
        });
    });

    // Tombol Reset
    document.getElementById('recapResetBtn')?.addEventListener('click', () => {
        const pSelect = document.getElementById('recapPeriod');
        const cameraSelect = document.getElementById('recapCamera');
        const objectTypeSelect = document.getElementById('recapObjectType');
        const directionSelect = document.getElementById('recapDirection');
        const startInput = document.getElementById('recapStart');
        const endInput = document.getElementById('recapEnd');
        const sInput = document.getElementById('recapSearch');

        if (pSelect) pSelect.value = 'today';
        if (cameraSelect) cameraSelect.value = '';
        if (objectTypeSelect) objectTypeSelect.value = '';
        if (directionSelect) directionSelect.value = '';
        if (startInput) startInput.value = '';
        if (endInput) endInput.value = '';
        if (sInput) sInput.value = '';
        _recapSearchQuery = '';

        loadRecap().catch(err => console.error('Error reset recap filter:', err));
    });

    // Listener tombol ekspor (juga di-trigger lewat onclick pada menu)
    document.getElementById('recapExport')?.addEventListener('click', exportRecap);
    document.getElementById('recapExportPdf')?.addEventListener('click', exportRecapPdf);

    loadRecap().catch(err => console.error('Error loadRecap:', err));
}

window._currentStatsPeriod = 'today';
window._trendChart = null;
window._donutChart = null;
window._cameraChart = null;
window._lastStatsCameras = [];
window._statsTopPlatesData = [];
window._statsSearchQuery = '';

// Helper untuk membaca CSS variables tema untuk Chart.js
function getThemeChartColors() {
    const s = getComputedStyle(document.documentElement);
    return {
        primary: s.getPropertyValue('--primary').trim() || '#087e8b',
        warning: s.getPropertyValue('--warning').trim() || '#f59e0b',
        line: s.getPropertyValue('--line').trim() || '#e2e8f0',
        muted: s.getPropertyValue('--muted').trim() || '#64748b',
        text: s.getPropertyValue('--text').trim() || '#0f172a',
        panel: s.getPropertyValue('--panel').trim() || '#ffffff'
    };
}

// Render chart beban lalu lintas per titik CCTV
function renderCameraBarChart(cameras) {
    const camCanvas = document.getElementById('cameraBarCanvas');
    if (!camCanvas || !window.Chart) return;
    if (cameras) window._lastStatsCameras = cameras;
    const cList = window._lastStatsCameras || [];

    if (window._cameraChart) {
        window._cameraChart.destroy();
        window._cameraChart = null;
    }

    const colors = getThemeChartColors();
    const rawLabels = cList.map(c => c.camera || c.camera_name || '-');
    const labels = rawLabels.map(l => l.length > 18 ? l.slice(0, 16) + '…' : l);
    const vehicleData = cList.map(c => c.vehicles ?? c.plate_count ?? 0);
    const peopleData = cList.map(c => c.people ?? c.face_count ?? 0);

    window._cameraChart = new Chart(camCanvas, {
        type: 'bar',
        data: {
            labels: labels.length ? labels : ['Belum ada data'],
            datasets: [
                {
                    label: 'Kendaraan',
                    data: vehicleData.length ? vehicleData : [0],
                    backgroundColor: colors.primary,
                    borderRadius: 3,
                    borderSkipped: false,
                    maxBarThickness: 32
                },
                {
                    label: 'Orang',
                    data: peopleData.length ? peopleData : [0],
                    backgroundColor: colors.warning,
                    borderRadius: 3,
                    borderSkipped: false,
                    maxBarThickness: 32
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'start',
                    labels: {
                        boxWidth: 10,
                        boxHeight: 10,
                        borderRadius: 2,
                        usePointStyle: false,
                        color: colors.muted,
                        font: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 11, weight: '500' },
                        padding: 12
                    }
                },
                tooltip: {
                    backgroundColor: colors.panel,
                    borderColor: colors.line,
                    borderWidth: 1,
                    titleColor: colors.text,
                    bodyColor: colors.text,
                    titleFont: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 12, weight: '500' },
                    bodyFont: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 11, weight: '400' },
                    padding: 8,
                    cornerRadius: 6,
                    displayColors: true,
                    boxWidth: 8,
                    boxHeight: 8,
                    boxPadding: 4,
                    callbacks: {
                        title: (items) => {
                            if (!items.length) return '';
                            const idx = items[0].dataIndex;
                            return rawLabels[idx] || items[0].label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: colors.muted,
                        font: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 11, weight: '400' },
                        maxRotation: 25,
                        minRotation: 0
                    },
                    border: { display: false }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: colors.line,
                        lineWidth: 0.5
                    },
                    ticks: {
                        precision: 0,
                        color: colors.muted,
                        font: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 11, weight: '400' }
                    },
                    border: { display: false }
                }
            }
        }
    });
}
window.renderCameraBarChart = renderCameraBarChart;

// Render chart aktivitas per jam (24 jam)
function renderHourlyBarChart(hourly) {
    const canvas = document.getElementById('hourlyBarCanvas');
    if (!canvas || !window.Chart) return;
    if (hourly) window._lastHourlyData = hourly;
    const hList = window._lastHourlyData || [];

    if (window._hourlyChart) {
        window._hourlyChart.destroy();
        window._hourlyChart = null;
    }

    const colors = getThemeChartColors();
    const labels = hList.map(h => h.label || `${String(h.hour).padStart(2, '0')}:00`);
    const vehicleData = hList.map(h => h.vehicles || 0);
    const peopleData = hList.map(h => h.people || 0);

    window._hourlyChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Kendaraan',
                    data: vehicleData,
                    backgroundColor: colors.primary,
                    borderRadius: 3,
                    borderSkipped: false,
                    maxBarThickness: 20
                },
                {
                    label: 'Orang',
                    data: peopleData,
                    backgroundColor: colors.warning,
                    borderRadius: 3,
                    borderSkipped: false,
                    maxBarThickness: 20
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 10,
                        boxHeight: 10,
                        borderRadius: 2,
                        color: colors.muted,
                        font: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 11, weight: '500' },
                        padding: 10
                    }
                },
                tooltip: {
                    backgroundColor: colors.panel,
                    borderColor: colors.line,
                    borderWidth: 1,
                    titleColor: colors.text,
                    bodyColor: colors.text,
                    padding: 8,
                    cornerRadius: 6
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: colors.muted,
                        font: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 10 },
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 12
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: colors.line },
                    ticks: {
                        color: colors.muted,
                        font: { family: '"Segoe UI", Inter, Arial, sans-serif', size: 10 },
                        precision: 0
                    }
                }
            }
        }
    });
}
window.renderHourlyBarChart = renderHourlyBarChart;

// Render analisis distribusi jenis kendaraan
function renderVehicleTypeBreakdown(vt) {
    const container = document.getElementById('statsVehicleTypeBreakdown');
    if (!container) return;
    const total = vt?.total || 0;
    const types = [
        { key: 'car', label: 'Mobil', count: vt?.car || 0, icon: 'bi-car-front', color: '#3b82f6' },
        { key: 'motorcycle', label: 'Sepeda Motor', count: vt?.motorcycle || 0, icon: 'bi-bicycle', color: '#10b981' },
        { key: 'bus', label: 'Bus', count: vt?.bus || 0, icon: 'bi-bus-front', color: '#f59e0b' },
        { key: 'truck', label: 'Truck', count: vt?.truck || 0, icon: 'bi-truck', color: '#ef4444' },
        { key: 'other', label: 'Lainnya / Unknown', count: vt?.other || 0, icon: 'bi-question-circle', color: '#6b7280' },
    ];

    const itemsHtml = types.map(t => {
        const pct = total > 0 ? ((t.count / total) * 100).toFixed(1) : '0.0';
        return `
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="d-flex align-items-center gap-2 small fw-medium">
                        <i class="bi ${t.icon}" style="color: ${t.color}"></i>
                        <span>${t.label}</span>
                    </span>
                    <span class="small text-muted"><strong class="text-body">${t.count.toLocaleString('id-ID')}</strong> unit (${pct}%)</span>
                </div>
                <div class="progress" style="height: 6px; background: rgba(107, 114, 128, 0.15);">
                    <div class="progress-bar" role="progressbar" style="width: ${pct}%; background-color: ${t.color};" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div class="p-1">
            <div class="d-flex justify-content-between align-items-center pb-2 mb-3 border-bottom">
                <span class="text-muted small">Total Kendaraan Terklasifikasi</span>
                <strong class="fs-6">${total.toLocaleString('id-ID')} unit</strong>
            </div>
            ${itemsHtml}
        </div>
    `;
}
window.renderVehicleTypeBreakdown = renderVehicleTypeBreakdown;

// Render performa pembacaan plat
function renderPlatePerformance(pp) {
    const container = document.getElementById('statsPlatePerformance');
    if (!container) return;
    const total = pp?.total_plates || 0;
    const valid = pp?.valid_plates || 0;
    const check = pp?.check_plates || 0;
    const failed = pp?.failed_plates || 0;
    const readRate = pp?.read_rate !== undefined && pp?.read_rate !== null ? `${pp.read_rate}%` : '0%';
    const avgConf = pp?.avg_confidence !== undefined && pp?.avg_confidence !== null ? `${pp.avg_confidence}%` : 'Tidak tersedia';

    container.innerHTML = `
        <div class="p-1">
            <div class="row g-2 mb-3">
                <div class="col-6">
                    <div class="p-3 rounded border text-center" style="background: rgba(59, 130, 246, 0.05);">
                        <div class="text-muted small mb-1">Total Plat Terdeteksi</div>
                        <strong class="fs-4 text-primary">${total.toLocaleString('id-ID')}</strong>
                    </div>
                </div>
                <div class="col-6">
                    <div class="p-3 rounded border text-center" style="background: rgba(16, 185, 129, 0.05);">
                        <div class="text-muted small mb-1">Read Rate</div>
                        <strong class="fs-4 text-success">${readRate}</strong>
                    </div>
                </div>
            </div>
            <div class="list-group list-group-flush small">
                <div class="list-group-item d-flex justify-content-between align-items-center px-1 py-2 border-bottom">
                    <span><i class="bi bi-check-circle text-success me-2"></i>Plat Terbaca / Valid</span>
                    <strong class="text-success">${valid.toLocaleString('id-ID')} unit</strong>
                </div>
                <div class="list-group-item d-flex justify-content-between align-items-center px-1 py-2 border-bottom">
                    <span><i class="bi bi-exclamation-triangle text-warning me-2"></i>Plat Perlu Dicek</span>
                    <strong class="text-warning">${check.toLocaleString('id-ID')} unit</strong>
                </div>
                <div class="list-group-item d-flex justify-content-between align-items-center px-1 py-2 border-bottom">
                    <span><i class="bi bi-x-circle text-danger me-2"></i>Plat Gagal / Buram</span>
                    <strong class="text-danger">${failed.toLocaleString('id-ID')} unit</strong>
                </div>
                <div class="list-group-item d-flex justify-content-between align-items-center px-1 py-2">
                    <span><i class="bi bi-speedometer2 text-info me-2"></i>Rata-rata Akurasi (Confidence)</span>
                    <strong class="text-body">${avgConf}</strong>
                </div>
            </div>
        </div>
    `;
}
window.renderPlatePerformance = renderPlatePerformance;

function renderTopPlatesTable() {
    // Stub dipertahankan untuk backward-compatibility jika ada panggilan lama
}

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

    try {
        const analyticsRes = await json(`/api/analytics?${analyticsParams('stats', period).toString()}`).catch(() => null);
        if (analyticsRes?.success && analyticsRes.data) {
            const data = analyticsRes.data;
            const s = data.summary || {};
            const platePerf = data.plate_performance || {};
            const vt = data.vehicle_types || {};

            // 1. Update 4 KPI Cards (Total Kendaraan, Total Orang, Plat Terbaca, CCTV Aktif)
            const elTotalDets = document.getElementById('kpiTotalDets');
            const elTotalPlates = document.getElementById('kpiTotalPlates');
            const elPlateRate = document.getElementById('kpiPlateRate');
            const elTotalFaces = document.getElementById('kpiTotalFaces');
            const elCamStatus = document.getElementById('kpiCamStatus');
            const elCamUptime = document.getElementById('kpiCamUptime');

            if (elTotalDets) elTotalDets.textContent = (s.vehicles || 0).toLocaleString('id-ID');
            if (elTotalFaces) elTotalFaces.textContent = (s.people || 0).toLocaleString('id-ID');
            if (elTotalPlates) elTotalPlates.textContent = (platePerf.valid_plates ?? s.plates ?? 0).toLocaleString('id-ID');
            if (elPlateRate) elPlateRate.textContent = platePerf.read_rate !== undefined ? `${platePerf.read_rate}%` : (s.vehicles ? `${Math.round((s.plates / s.vehicles) * 100)}%` : '0%');
            if (elCamStatus) elCamStatus.textContent = `${s.active_cameras || 0} / ${s.total_cameras || 0}`;
            if (elCamUptime) elCamUptime.textContent = s.total_cameras ? `${Math.round((s.active_cameras / s.total_cameras) * 100)}%` : '0%';

            // 2. Rincian Masuk & Keluar (Dua Grup Berdampingan)
            const statsBreakdown = document.getElementById('statsBreakdown');
            if (statsBreakdown) {
                statsBreakdown.innerHTML = `
                    <div class="stats-breakdown-group">
                        <span class="stats-group-label">Rincian masuk</span>
                        <div class="stats-subgrid-2">
                            <div class="stats-sub-card">
                                <span class="metric-card-label">Kendaraan masuk</span>
                                <strong class="metric-card-value">${(s.vehicle_entry || 0).toLocaleString('id-ID')}</strong>
                                <span class="stats-dir-muted"><i class="bi bi-arrow-up-right"></i> Arah ke dalam</span>
                            </div>
                            <div class="stats-sub-card">
                                <span class="metric-card-label">Orang masuk</span>
                                <strong class="metric-card-value">${(s.people_entry || 0).toLocaleString('id-ID')}</strong>
                                <span class="stats-dir-muted"><i class="bi bi-arrow-up-right"></i> Arah ke dalam</span>
                            </div>
                        </div>
                    </div>
                    <div class="stats-breakdown-group">
                        <span class="stats-group-label">Rincian keluar</span>
                        <div class="stats-subgrid-2">
                            <div class="stats-sub-card">
                                <span class="metric-card-label">Kendaraan keluar</span>
                                <strong class="metric-card-value">${(s.vehicle_exit || 0).toLocaleString('id-ID')}</strong>
                                <span class="stats-dir-muted"><i class="bi bi-arrow-down-right"></i> Arah ke luar</span>
                            </div>
                            <div class="stats-sub-card">
                                <span class="metric-card-label">Orang keluar</span>
                                <strong class="metric-card-value">${(s.people_exit || 0).toLocaleString('id-ID')}</strong>
                                <span class="stats-dir-muted"><i class="bi bi-arrow-down-right"></i> Arah ke luar</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            // 3. Render Chart.js Aktivitas per CCTV
            renderCameraBarChart(data.cameras || []);

            // 4. Render Analisis Jam Sibuk
            const peak = (data.hourly || []).reduce((best, item) => (item.vehicles > (best?.vehicles || -1) ? item : best), null);
            const peakContainer = document.getElementById('peakHoursList');
            if (peakContainer) {
                if (peak && (peak.vehicles > 0 || peak.people > 0)) {
                    const startH = String(peak.hour ?? 0).padStart(2, '0');
                    const endH = String(((peak.hour ?? 0) + 1) % 24).padStart(2, '0');
                    peakContainer.innerHTML = `
                        <div class="stats-peak-content">
                            <div class="stats-peak-time">${startH}:00 - ${endH}:00 WIB</div>
                            <div class="stats-peak-desc"><strong>${(peak.vehicles || 0).toLocaleString('id-ID')}</strong> kendaraan · <strong>${(peak.people || 0).toLocaleString('id-ID')}</strong> orang</div>
                        </div>
                    `;
                } else {
                    peakContainer.innerHTML = '<div class="stats-peak-desc text-muted">Belum ada data jam sibuk pada periode ini.</div>';
                }
            }

            // 5. Render Grafik Aktivitas per Jam
            renderHourlyBarChart(data.hourly || []);

            // 6. Render Distribusi Jenis Kendaraan
            renderVehicleTypeBreakdown(vt);

            // 7. Render Performa Pembacaan Plat
            renderPlatePerformance(platePerf);

            return;
        }

        // Fallback ke /api/statistics/enterprise jika /api/analytics tidak tersedia
        const res = await json(`/api/statistics/enterprise?period=${period}`);
        const stats = res.data || {};
        const kpi = stats.kpi || {};
        const camDist = stats.camera_distribution || [];
        const peakHours = stats.peak_hours || [];
        const platePerf = stats.plate_performance || {};
        const vt = stats.vehicle_types || {};

        const elTotalDets = document.getElementById('kpiTotalDets');
        const elTotalPlates = document.getElementById('kpiTotalPlates');
        const elPlateRate = document.getElementById('kpiPlateRate');
        const elTotalFaces = document.getElementById('kpiTotalFaces');
        const elCamStatus = document.getElementById('kpiCamStatus');
        const elCamUptime = document.getElementById('kpiCamUptime');

        if (elTotalDets) elTotalDets.textContent = (kpi.total_detections || 0).toLocaleString('id-ID');
        if (elTotalPlates) elTotalPlates.textContent = (platePerf.valid_plates ?? kpi.total_plates ?? 0).toLocaleString('id-ID');
        if (elPlateRate) elPlateRate.textContent = platePerf.read_rate !== undefined ? `${platePerf.read_rate}%` : `${kpi.plate_read_rate || 0}%`;
        if (elTotalFaces) elTotalFaces.textContent = (kpi.total_faces || 0).toLocaleString('id-ID');
        if (elCamStatus) elCamStatus.textContent = `${kpi.active_cameras || 0} / ${kpi.total_cameras || 0}`;
        if (elCamUptime) elCamUptime.textContent = `${kpi.camera_availability || 0}%`;

        // Camera bar chart fallback
        const camerasAdapted = camDist.map(c => ({
            camera: c.camera_name,
            vehicles: c.plate_count,
            people: c.face_count
        }));
        renderCameraBarChart(camerasAdapted);

        // Peak hours fallback
        const peakContainer = document.getElementById('peakHoursList');
        if (peakContainer) {
            if (peakHours.length) {
                const ph = peakHours[0];
                peakContainer.innerHTML = `
                    <div class="stats-peak-content">
                        <div class="stats-peak-time">${esc(ph.time_range)}</div>
                        <div class="stats-peak-desc"><strong>${(ph.count || 0).toLocaleString('id-ID')}</strong> kendaraan · <strong>${(ph.people || 0).toLocaleString('id-ID')}</strong> orang</div>
                    </div>
                `;
            } else {
                peakContainer.innerHTML = '<div class="stats-peak-desc text-muted">Belum ada data jam sibuk pada periode ini.</div>';
            }
        }

        renderVehicleTypeBreakdown(vt);
        renderPlatePerformance(platePerf);
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

    // Live search Top 10
    const searchInput = document.getElementById('statsSearch');
    if (searchInput) {
        let searchTimer = null;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimer);
            searchTimer = setTimeout(() => {
                window._statsSearchQuery = e.target.value;
                renderTopPlatesTable();
            }, 150);
        });
    }

    // Auto set tanggal awal & akhir saat rentang cepat dipilih
    const periodSelect = document.getElementById('statsPeriod');
    if (periodSelect) {
        periodSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            const today = new Date();
            const formatDate = d => d.toISOString().split('T')[0];
            const startInput = document.getElementById('statsStart');
            const endInput = document.getElementById('statsEnd');
            if (!startInput || !endInput) return;

            if (val === 'today') {
                startInput.value = formatDate(today);
                endInput.value = formatDate(today);
            } else if (val === '7d') {
                const d = new Date();
                d.setDate(d.getDate() - 7);
                startInput.value = formatDate(d);
                endInput.value = formatDate(today);
            } else if (val === '30d') {
                const d = new Date();
                d.setDate(d.getDate() - 30);
                startInput.value = formatDate(d);
                endInput.value = formatDate(today);
            } else if (val === 'all') {
                startInput.value = '';
                endInput.value = '';
            }
        });
    }

    // Terapkan filter
    document.getElementById('statsApply')?.addEventListener('click', () => {
        loadEnterpriseStatistics(document.getElementById('statsPeriod')?.value || 'today')
            .catch(err => showNotification(err.message, 'warning'));
    });

    // Reset filter
    document.getElementById('statsResetBtn')?.addEventListener('click', () => {
        const periodSelect = document.getElementById('statsPeriod');
        const startInput = document.getElementById('statsStart');
        const endInput = document.getElementById('statsEnd');
        const objectType = document.getElementById('statsObjectType');
        const cameraSelect = document.getElementById('statsCamera');
        const searchInput = document.getElementById('statsSearch');

        if (periodSelect) periodSelect.value = 'today';
        if (startInput) startInput.value = '';
        if (endInput) endInput.value = '';
        if (objectType) objectType.value = '';
        if (cameraSelect) cameraSelect.value = '';
        if (searchInput) searchInput.value = '';
        window._statsSearchQuery = '';
        window._currentStatsPeriod = 'today';

        loadEnterpriseStatistics('today').catch(err => console.error('Error reset stats filter:', err));
    });

    // Theme observer untuk chart jika beralih dark/light mode
    if (!window._statsThemeObserverBound) {
        window._statsThemeObserverBound = true;
        try {
            const obs = new MutationObserver(() => {
                setTimeout(() => {
                    if (document.getElementById('cameraBarCanvas') && window._lastStatsCameras) {
                        if (typeof renderCameraBarChart === 'function') renderCameraBarChart();
                    }
                    if (document.getElementById('hourlyBarCanvas') && window._lastHourlyData) {
                        if (typeof renderHourlyBarChart === 'function') renderHourlyBarChart(window._lastHourlyData);
                    }
                }, 50);
            });
            obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-bs-theme', 'data-theme'] });
        } catch (e) { }
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
        const cardAvatar = document.getElementById('profileCardAvatar');
        const name = settings.operator_name || 'Super Admin CCTV';
        const profileRole = settings.profile_role || 'Kepala Operator';
        if (displayName) displayName.textContent = name;
        if (displayRole) displayRole.textContent = profileRole;
        if (topbarName) topbarName.textContent = name;
        if (topbarRole) topbarRole.textContent = profileRole;
        if (avatar) avatar.textContent = name.charAt(0).toUpperCase();
        if (cardAvatar) cardAvatar.textContent = name.charAt(0).toUpperCase();

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
            icon.className = 'bi bi-sun';
        }
        if (label) {
            label.textContent = 'Mode Terang';
        }
        if (btn) {
            btn.title = 'Beralih ke mode terang';
            btn.setAttribute('aria-label', 'Beralih ke mode terang');
        }
    } else {
        document.documentElement.classList.remove('dark');
        document.documentElement.setAttribute('data-bs-theme', 'light');
        if (icon) {
            icon.className = 'bi bi-moon-stars';
        }
        if (label) {
            label.textContent = 'Mode Gelap';
        }
        if (btn) {
            btn.title = 'Beralih ke mode gelap';
            btn.setAttribute('aria-label', 'Beralih ke mode gelap');
        }
    }

    refreshChartThemes(theme);
}

function refreshChartThemes(theme) {
    if (!window.Chart) return;
    const isDark = theme === 'dark';
    const textColor = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : '#f1f5f9';

    if (document.getElementById('cameraBarCanvas') && typeof renderCameraBarChart === 'function' && window._lastStatsCameras) {
        try {
            renderCameraBarChart();
        } catch (e) {
            console.warn('Error updating cameraBarChart theme:', e);
        }
    }

    [window._trendChart, window._statusChart].forEach(chart => {
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
    } catch (e) { }
    updateThemeUI(nextTheme);
}

function initTheme() {
    let savedTheme = null;
    try {
        savedTheme = localStorage.getItem('platevision_theme') || localStorage.getItem('theme');
    } catch (e) { }

    if (!savedTheme) {
        savedTheme = 'dark';
    }

    updateThemeUI(savedTheme);

    const toggleBtn = document.getElementById('themeToggleBtn');
    if (toggleBtn && !toggleBtn._hasThemeListener) {
        toggleBtn.addEventListener('click', toggleTheme);
        toggleBtn._hasThemeListener = true;
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
    loadSettingsFromDb();
    setInterval(updateClock, 1000);
    updateClock();

    if (document.getElementById('cameraTable')) {
        initMonitoring();
    }
    if (document.getElementById('totalVehicle') || document.getElementById('liveStreamImg')) {
        initDashboard();
    }
    if (document.getElementById('suspiciousActivityPage')) {
        initSuspiciousActivityPage();
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