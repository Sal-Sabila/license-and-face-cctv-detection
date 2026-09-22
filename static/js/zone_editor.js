/**
 * ZONE EDITOR — Atur polygon MID & NEAR per kamera CCTV langsung dari web.
 *
 * Fitur:
 *  - Pilih kamera dari dropdown
 *  - Gambar polygon MID (biru) & NEAR (hijau) dengan klik pada gambar
 *  - Validasi polygon (menyilang, terlalu kecil, dsb)
 *  - Pop-up notifikasi sukses/gagal dengan alasan jelas
 *  - Panduan langkah demi langkah
 *  - Simpan ke DB via POST /api/zones/<id>
 *  - Reset ke default via POST /api/zones/<id>/reset
 */

// ============================================================
// STATE
// ============================================================

const zoneState = {
    cameraId: null,
    cameraName: '',
    mid: [],
    near: [],
    editing: 'mid',
    imageLoaded: false,
    loading: false,
    dirty: false,
    originalMid: [],
    originalNear: [],
};

const ZONE_COLORS = {
    mid: {
        stroke: 'rgba(37, 99, 235, 0.9)',
        fill: 'rgba(37, 99, 235, 0.15)',
        solid: '#2563eb',
        label: 'MID (Person + Vehicle)',
    },
    near: {
        stroke: 'rgba(34, 197, 94, 0.9)',
        fill: 'rgba(34, 197, 94, 0.15)',
        solid: '#22c55e',
        label: 'NEAR (Plate + OCR)',
    },
};

// ============================================================
// INIT
// ============================================================

function initZoneEditor() {
    const canvas = document.getElementById('zoneEditorCanvas');
    if (!canvas) return;

    console.log('[ZONE EDITOR] Inisialisasi halaman editor zona');

    bindEvents();
    loadCameraOptionsForZone();
}

function bindEvents() {
    document.getElementById('zoneCameraSelect')?.addEventListener('change', e => {
        const id = e.target.value;
        if (!id) return;

        if (zoneState.dirty) {
            askConfirmation(
                'Ada perubahan yang belum disimpan. Yakin pindah kamera?',
                'Perubahan Belum Disimpan'
            ).then(ok => {
                if (ok) loadZone(id);
                else {
                    document.getElementById('zoneCameraSelect').value = zoneState.cameraId;
                }
            });
        } else {
            loadZone(id);
        }
    });

    document.getElementById('zoneEditorCanvas')?.addEventListener('click', handleCanvasClick);

    document.getElementById('zoneModeMid')?.addEventListener('change', () => {
        zoneState.editing = 'mid';
        updateModeHelper();
        drawZoneCanvas();
    });
    document.getElementById('zoneModeNear')?.addEventListener('change', () => {
        zoneState.editing = 'near';
        updateModeHelper();
        drawZoneCanvas();
    });

    window.addEventListener('resize', () => {
        if (!zoneState.imageLoaded) return;
        resizeCanvas();
        drawZoneCanvas();
    });

    window.addEventListener('beforeunload', e => {
        if (zoneState.dirty) {
            e.preventDefault();
            e.returnValue = '';
        }
    });
}

// ============================================================
// LOAD CAMERA LIST
// ============================================================

async function loadCameraOptionsForZone() {
    const select = document.getElementById('zoneCameraSelect');
    if (!select) return;

    try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        const cameras = data.data || [];

        if (!cameras.length) {
            select.innerHTML = '<option value="">Belum ada kamera</option>';
            return;
        }

        select.innerHTML = cameras.map(c =>
            `<option value="${c.id || c.camera_id}">${c.name || c.location || ('CAM ' + (c.id || c.camera_id))}</option>`
        ).join('');

        const firstId = cameras[0].id || cameras[0].camera_id;
        const firstName = cameras[0].name || cameras[0].location || ('CAM ' + firstId);

        zoneState.cameraId = firstId;
        zoneState.cameraName = firstName;
        select.value = firstId;
        loadZone(firstId);

    } catch (err) {
        console.error('[ZONE EDITOR] Gagal memuat daftar kamera:', err);
        select.innerHTML = '<option value="">Gagal memuat kamera</option>';
        showNotification('Gagal memuat daftar kamera. Cek koneksi server.', 'danger');
    }
}

// ============================================================
// LOAD ZONE
// ============================================================

async function loadZone(cameraId) {
    if (!cameraId) return;

    zoneState.cameraId = cameraId;
    zoneState.imageLoaded = false;
    zoneState.loading = true;
    zoneState.dirty = false;

    const select = document.getElementById('zoneCameraSelect');
    if (select) {
        zoneState.cameraName = select.options[select.selectedIndex]?.text || ('CAM ' + cameraId);
    }

    const img = document.getElementById('zoneEditorImage');
    if (img) {
        img.src = `/api/video_feed/${cameraId}/snapshot?t=${Date.now()}`;

        img.onload = () => {
            zoneState.imageLoaded = true;
            zoneState.loading = false;
            resizeCanvas();
            drawZoneCanvas();
            console.log(`[ZONE EDITOR] CAM ${cameraId} frame loaded`);
        };

        img.onerror = () => {
            zoneState.loading = false;
            zoneState.imageLoaded = true;
            resizeCanvas();
            drawZoneCanvas();
            showNotification(
                'Gagal memuat frame CCTV. Cek koneksi kamera atau stream URL.',
                'warning'
            );
        };
    }

    try {
        const res = await fetch(`/api/zones/${cameraId}`);
        const data = await res.json();

        if (data.success && data.data) {
            zoneState.mid = normalizePolygon(data.data.mid);
            zoneState.near = normalizePolygon(data.data.near);
            zoneState.originalMid = [...zoneState.mid.map(p => [...p])];
            zoneState.originalNear = [...zoneState.near.map(p => [...p])];

            if (data.data.is_custom) {
                showNotification(`Zona kustom untuk ${zoneState.cameraName} dimuat.`, 'info');
            } else {
                showNotification(`Menggunakan zona default untuk ${zoneState.cameraName}.`, 'info');
            }

            console.log(
                `[ZONE EDITOR] CAM ${cameraId} zona: ` +
                `MID=${zoneState.mid.length} titik, NEAR=${zoneState.near.length} titik`
            );
        } else {
            zoneState.mid = [];
            zoneState.near = [];
            zoneState.originalMid = [];
            zoneState.originalNear = [];
        }
    } catch (err) {
        console.error('[ZONE EDITOR] Gagal load zona:', err);
        zoneState.mid = [];
        zoneState.near = [];
        showNotification('Gagal memuat zona dari server.', 'danger');
    }

    if (zoneState.imageLoaded) {
        resizeCanvas();
        drawZoneCanvas();
    }

    updateModeHelper();
    updateDirtyIndicator();
}

function normalizePolygon(poly) {
    if (!Array.isArray(poly)) return [];
    return poly
        .filter(p => Array.isArray(p) && p.length === 2)
        .map(p => [Number(p[0]), Number(p[1])])
        .filter(p => !isNaN(p[0]) && !isNaN(p[1]));
}

// ============================================================
// UI HELPERS
// ============================================================

function updateModeHelper() {
    const mode = zoneState.editing;
    const count = zoneState[mode].length;
    const el = document.getElementById('zoneModeHelper');
    if (el) {
        el.innerHTML = `<i class="bi bi-info-circle"></i> ` +
            `Mode: <strong>${ZONE_COLORS[mode].label}</strong> — ` +
            `titik saat ini: <strong>${count}</strong>` +
            (count < 3 ? ' (minimal 3 titik)' : ' ✓');
    }
}

function updateDirtyIndicator() {
    const el = document.getElementById('zoneDirtyIndicator');
    if (!el) return;

    if (zoneState.dirty) {
        el.innerHTML = '<span class="badge bg-warning text-dark"><i class="bi bi-exclamation-circle"></i> Ada perubahan belum disimpan</span>';
    } else {
        el.innerHTML = '<span class="badge bg-success"><i class="bi bi-check-circle"></i> Tersimpan</span>';
    }
}

function markDirty() {
    zoneState.dirty = true;
    updateDirtyIndicator();
}

// ============================================================
// CANVAS SIZE & DRAW
// ============================================================

function resizeCanvas() {
    const img = document.getElementById('zoneEditorImage');
    const canvas = document.getElementById('zoneEditorCanvas');
    if (!img || !canvas) return;

    const w = img.clientWidth;
    const h = img.clientHeight;
    if (w === 0 || h === 0) return;

    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
}

function drawZoneCanvas() {
    const canvas = document.getElementById('zoneEditorCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    drawPolygon(ctx, zoneState.mid, w, h, ZONE_COLORS.mid);
    drawPolygon(ctx, zoneState.near, w, h, ZONE_COLORS.near);
}

function drawPolygon(ctx, points, w, h, colors) {
    if (!points.length) return;

    if (points.length >= 3) {
        ctx.beginPath();
        points.forEach(([x, y], i) => {
            const px = x * w;
            const py = y * h;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fillStyle = colors.fill;
        ctx.fill();
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 2.5;
        ctx.stroke();
    } else if (points.length === 2) {
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 2;
        ctx.moveTo(points[0][0] * w, points[0][1] * h);
        ctx.lineTo(points[1][0] * w, points[1][1] * h);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    points.forEach(([x, y], idx) => {
        const px = x * w;
        const py = y * h;

        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = colors.solid;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Segoe UI, Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(idx + 1), px, py);
    });
}

// ============================================================
// CANVAS CLICK
// ============================================================

function handleCanvasClick(event) {
    const canvas = document.getElementById('zoneEditorCanvas');
    if (!canvas || !zoneState.cameraId) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;

    if (x < 0 || x > 1 || y < 0 || y > 1) return;

    zoneState[zoneState.editing].push([x, y]);
    markDirty();
    drawZoneCanvas();
    updateModeHelper();

    console.log(
        `[ZONE EDITOR] Titik #${zoneState[zoneState.editing].length} ` +
        `${zoneState.editing.toUpperCase()}: (${x.toFixed(3)}, ${y.toFixed(3)})`
    );
}

// ============================================================
// UNDO / CLEAR
// ============================================================

function undoPoint() {
    const target = zoneState[zoneState.editing];
    if (target.length > 0) {
        target.pop();
        markDirty();
        drawZoneCanvas();
        updateModeHelper();
        showNotification(`Titik terakhir ${zoneState.editing.toUpperCase()} dihapus.`, 'info');
    } else {
        showNotification(`Zona ${zoneState.editing.toUpperCase()} sudah kosong.`, 'info');
    }
}

function clearCurrentZone() {
    const target = zoneState[zoneState.editing];
    if (target.length === 0) {
        showNotification(`Zona ${zoneState.editing.toUpperCase()} sudah kosong.`, 'info');
        return;
    }

    const label = zoneState.editing.toUpperCase();
    askConfirmation(`Kosongkan semua titik zona ${label}?`, 'Kosongkan Zona').then(ok => {
        if (!ok) return;
        zoneState[zoneState.editing] = [];
        markDirty();
        drawZoneCanvas();
        updateModeHelper();
        showNotification(`Zona ${label} dikosongkan.`, 'success');
    });
}

// ============================================================
// VALIDASI POLYGON
// ============================================================

function polygonArea(points) {
    if (!points || points.length < 3) return 0;
    let area = 0;
    for (let i = 0; i < points.length; i++) {
        const j = (i + 1) % points.length;
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }
    return Math.abs(area) / 2;
}

function isPolygonSelfIntersecting(points) {
    if (!points || points.length < 4) return false;
    const n = points.length;

    function ccw(a, b, c) {
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
    }

    function segmentsIntersect(p1, p2, p3, p4) {
        return ccw(p1, p3, p4) !== ccw(p2, p3, p4) &&
               ccw(p1, p2, p3) !== ccw(p1, p2, p4);
    }

    for (let i = 0; i < n; i++) {
        const p1 = points[i];
        const p2 = points[(i + 1) % n];
        for (let j = i + 1; j < n; j++) {
            const p3 = points[j];
            const p4 = points[(j + 1) % n];

            if (Math.abs(i - j) <= 1) continue;
            if (i === 0 && j === n - 1) continue;

            if (segmentsIntersect(p1, p2, p3, p4)) {
                return true;
            }
        }
    }
    return false;
}

function validateZones() {
    const errors = [];

    if (zoneState.mid.length < 3) {
        errors.push('Zona MID butuh minimal 3 titik.');
    }
    if (zoneState.near.length < 3) {
        errors.push('Zona NEAR butuh minimal 3 titik.');
    }

    if (errors.length) return errors;

    if (isPolygonSelfIntersecting(zoneState.mid)) {
        errors.push('Zona MID menyilang (self-intersect). Perbaiki urutan titik.');
    }
    if (isPolygonSelfIntersecting(zoneState.near)) {
        errors.push('Zona NEAR menyilang (self-intersect). Perbaiki urutan titik.');
    }

    const midArea = polygonArea(zoneState.mid);
    const nearArea = polygonArea(zoneState.near);

    if (midArea < 0.03) {
        errors.push(`Zona MID terlalu kecil (luas ${(midArea * 100).toFixed(1)}%). Perbesar polygon.`);
    }
    if (nearArea < 0.03) {
        errors.push(`Zona NEAR terlalu kecil (luas ${(nearArea * 100).toFixed(1)}%). Perbesar polygon.`);
    }

    const validCoord = (poly) => poly.every(p =>
        p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1
    );
    if (!validCoord(zoneState.mid)) errors.push('Ada titik MID di luar area gambar.');
    if (!validCoord(zoneState.near)) errors.push('Ada titik NEAR di luar area gambar.');

    return errors;
}

// ============================================================
// SAVE ZONE
// ============================================================

async function saveZone() {
    if (!zoneState.cameraId) {
        showNotification('Pilih kamera terlebih dahulu.', 'warning');
        return;
    }

    const errors = validateZones();
    if (errors.length) {
        const msg = 'Validasi gagal:\n• ' + errors.join('\n• ');
        showNotification(msg, 'danger', 'Zona Tidak Valid');
        console.warn('[ZONE EDITOR] Validasi gagal:', errors);
        return;
    }

    const ok = await askConfirmation(
        `Simpan zona MID (${zoneState.mid.length} titik) & NEAR (${zoneState.near.length} titik) ` +
        `untuk ${zoneState.cameraName}?`,
        'Simpan Zona'
    );
    if (!ok) return;

    try {
        const res = await fetch(`/api/zones/${zoneState.cameraId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mid: zoneState.mid,
                near: zoneState.near,
            }),
        });

        const data = await res.json();

        if (data.success) {
            zoneState.dirty = false;
            zoneState.originalMid = [...zoneState.mid.map(p => [...p])];
            zoneState.originalNear = [...zoneState.near.map(p => [...p])];
            updateDirtyIndicator();
            showNotification(
                `Zona ${zoneState.cameraName} berhasil disimpan. AI akan pakai zona baru segera.`,
                'success',
                'Berhasil Disimpan'
            );
            console.log('[ZONE EDITOR] Zona disimpan:', data);
        } else {
            showNotification(
                'Gagal menyimpan zona: ' + (data.message || 'Unknown error'),
                'danger',
                'Gagal Menyimpan'
            );
            console.error('[ZONE EDITOR] Server error:', data);
        }
    } catch (err) {
        showNotification(
            'Gagal menyimpan zona. Cek koneksi server.',
            'danger',
            'Error Koneksi'
        );
        console.error('[ZONE EDITOR] Fetch error:', err);
    }
}

// ============================================================
// RESET ZONE
// ============================================================

async function resetZone() {
    if (!zoneState.cameraId) {
        showNotification('Pilih kamera terlebih dahulu.', 'warning');
        return;
    }

    const ok = await askConfirmation(
        `Reset zona ${zoneState.cameraName} ke default? Zona kustom akan dihapus.`,
        'Reset Zona'
    );
    if (!ok) return;

    try {
        const res = await fetch(`/api/zones/${zoneState.cameraId}/reset`, {
            method: 'POST',
        });
        const data = await res.json();

        if (data.success) {
            zoneState.dirty = false;

            if (data.data) {
                zoneState.mid = normalizePolygon(data.data.mid);
                zoneState.near = normalizePolygon(data.data.near);
                zoneState.originalMid = [...zoneState.mid.map(p => [...p])];
                zoneState.originalNear = [...zoneState.near.map(p => [...p])];
                drawZoneCanvas();
                updateModeHelper();
            } else {
                loadZone(zoneState.cameraId);
            }

            updateDirtyIndicator();
            showNotification(
                `Zona ${zoneState.cameraName} berhasil direset ke default.`,
                'success',
                'Reset Berhasil'
            );
        } else {
            showNotification(
                'Gagal reset: ' + (data.message || 'Unknown error'),
                'danger'
            );
        }
    } catch (err) {
        showNotification('Gagal reset zona. Cek koneksi server.', 'danger');
        console.error('[ZONE EDITOR] Reset error:', err);
    }
}

// ============================================================
// REFRESH FRAME
// ============================================================

function refreshFrame() {
    if (!zoneState.cameraId) return;
    const img = document.getElementById('zoneEditorImage');
    if (img) {
        img.src = `/api/video_feed/${zoneState.cameraId}/snapshot?t=${Date.now()}`;
        showNotification('Memuat ulang frame CCTV...', 'info');
    }
}

// ============================================================
// EXPORT
// ============================================================

window.saveZone = saveZone;
window.resetZone = resetZone;
window.undoPoint = undoPoint;
window.clearCurrentZone = clearCurrentZone;
window.refreshFrame = refreshFrame;

// ============================================================
// AUTO-INIT
// ============================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initZoneEditor);
} else {
    initZoneEditor();
}