// static/js/tech_dashboard.js

document.addEventListener('DOMContentLoaded', function() {
    loadStatistics();
    loadImages();
    loadReports();
    setupUpload();
});

function setupUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadImage(e.target.files[0]);
    });
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault(); uploadArea.classList.add('bg-gray-50');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('bg-gray-50');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault(); uploadArea.classList.remove('bg-gray-50');
        if (e.dataTransfer.files.length > 0) uploadImage(e.dataTransfer.files[0]);
    });
}

async function uploadImage(file) {
    const allowed = ['image/png','image/jpeg','image/jpg','image/tiff','image/tif'];
    if (!allowed.includes(file.type)) return showUploadResult('error','Invalid file type.');
    if (file.size > 16 * 1024 * 1024) return showUploadResult('error','Max 16MB.');

    setVisible('uploadProgress', true); setVisible('uploadResult', false);
    const fd = new FormData(); fd.append('image', file);

    try {
        const res = await fetch('/technician/api/upload', { method: 'POST', body: fd });
        const data = await res.json();
        setVisible('uploadProgress', false);
        if (res.ok && data.success) {
            const p = data.prediction || {};
            showUploadResult('success',
                `Image uploaded & analyzed.<br>
                 <strong>Total Cells:</strong> ${p.total_cells ?? 0}<br>
                 <strong>Abnormal Cells:</strong> ${p.abnormal_cells ?? 0}<br>
                 <strong>Confidence:</strong> ${((p.overall_confidence ?? 0)*100).toFixed(1)}%`);
            loadStatistics(); loadImages(); loadReports();
            document.getElementById('fileInput').value = '';
        } else {
            showUploadResult('error', data.error || 'Upload failed.');
        }
    } catch (e) {
        setVisible('uploadProgress', false);
        showUploadResult('error','Network error.'); console.error(e);
    }
}

function showUploadResult(type, msg) {
    const el = document.getElementById('uploadResult');
    el.innerHTML = `<div class="p-3 rounded ${type==='success'?'bg-green-100 text-green-800':'bg-red-100 text-red-800'}">${msg}</div>`;
    setVisible('uploadResult', true);
}
function setVisible(id, vis){const el=document.getElementById(id); if(!el) return; vis?el.classList.remove('hidden'):el.classList.add('hidden');}

async function loadStatistics() {
    try {
        const res = await fetch('/technician/api/stats'); const d = await res.json();
        document.getElementById('stat-total').textContent = d.total_uploads ?? 0;
        document.getElementById('stat-pending').textContent = d.pending_review ?? 0;
        document.getElementById('stat-completed').textContent = d.completed ?? 0;
    } catch(e){ console.error(e); }
}

async function loadImages() {
    try {
        const res = await fetch('/technician/api/images'); const d = await res.json();
        const tbody = document.getElementById('imagesTableBody');
        if (!d.images || d.images.length===0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center p-3 text-gray-500">No images uploaded yet</td></tr>'; return;
        }
        tbody.innerHTML = d.images.map(img => {
            const badge = getStatusBadge(img.status);
            const cells = img.prediction ? img.prediction.cell_count : '-';
            const abn   = img.prediction ? img.prediction.abnormal_count : '-';
            return `<tr class="border-t">
                <td class="px-4 py-2">${img.id}</td>
                <td class="px-4 py-2">${escapeHtml(img.filename)}</td>
                <td class="px-4 py-2">${new Date(img.upload_time).toLocaleString()}</td>
                <td class="px-4 py-2">${badge}</td>
                <td class="px-4 py-2">${cells}</td>
                <td class="px-4 py-2">${abn}</td>
                <td class="px-4 py-2">
                  ${img.report
                    ? `<span class="inline-block px-2 py-1 text-xs rounded bg-green-100 text-green-800">Report Available</span>`
                    : `<span class="inline-block px-2 py-1 text-xs rounded bg-yellow-100 text-yellow-800">Pending</span>`}
                </td>
            </tr>`;
        }).join('');
    } catch(e){
        console.error(e);
        document.getElementById('imagesTableBody').innerHTML = '<tr><td colspan="7" class="text-center p-3 text-red-600">Error</td></tr>';
    }
}

async function loadReports() {
    try {
        const res = await fetch('/technician/api/reports'); const d = await res.json();
        const tbody = document.getElementById('reportsTableBody');
        if (!d.reports || d.reports.length===0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center p-3 text-gray-500">No reports available yet</td></tr>'; return;
        }
        tbody.innerHTML = d.reports.map(r => {
            const badge = r.status==='Positive'
              ? '<span class="inline-block px-2 py-1 text-xs rounded bg-red-100 text-red-800">Positive</span>'
              : '<span class="inline-block px-2 py-1 text-xs rounded bg-green-100 text-green-800">Negative</span>';
            return `<tr class="border-t">
              <td class="px-4 py-2">${r.id}</td>
              <td class="px-4 py-2">${escapeHtml(r.filename)}</td>
              <td class="px-4 py-2">${badge}</td>
              <td class="px-4 py-2">${escapeHtml(r.doctor)}</td>
              <td class="px-4 py-2">${new Date(r.created_at).toLocaleString()}</td>
              <td class="px-4 py-2">
                <button class="px-3 py-1 text-sm rounded bg-blue-600 text-white" onclick="downloadReport(${r.id})">Download PDF</button>
              </td></tr>`;
        }).join('');
    } catch(e){
        console.error(e);
        document.getElementById('reportsTableBody').innerHTML = '<tr><td colspan="6" class="text-center p-3 text-red-600">Error</td></tr>';
    }
}

function downloadReport(id){ window.location.href = `/technician/api/download/report/${id}`; }

function getStatusBadge(s){
    const m = {
        pending:  '<span class="inline-block px-2 py-1 text-xs rounded bg-gray-200 text-gray-800">Pending</span>',
        predicted:'<span class="inline-block px-2 py-1 text-xs rounded bg-yellow-100 text-yellow-800">Predicted</span>',
        reviewed: '<span class="inline-block px-2 py-1 text-xs rounded bg-green-100 text-green-800">Reviewed</span>'
    };
    return m[s] || m.pending;
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}

function showSection(section){
    document.querySelectorAll('.section-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.list-group-item').forEach(el => el.classList.remove('ring','ring-blue-400'));
    const sec = document.getElementById(`section-${section}`); if (sec) sec.classList.remove('hidden');
    if (event && event.target && event.target.classList) event.target.classList.add('ring','ring-blue-400');
    if (section==='images') loadImages(); else if (section==='reports') loadReports();
}
