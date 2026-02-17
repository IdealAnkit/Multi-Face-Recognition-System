const btnEnrollStart = document.getElementById('btn-enroll-start');
const btnEnrollCancel = document.getElementById('btn-enroll-cancel');
const btnAttendStart = document.getElementById('btn-attend-start');
const btnAttendStop = document.getElementById('btn-attend-stop');

function setUiState(state) {
    if (state === 'IDLE') {
        btnEnrollStart.disabled = false;
        btnEnrollCancel.disabled = true;
        btnAttendStart.disabled = false;
        btnAttendStop.disabled = true;
    } else if (state === 'ENROLLMENT') {
        btnEnrollStart.disabled = true;
        btnEnrollCancel.disabled = false;
        btnAttendStart.disabled = true;
        btnAttendStop.disabled = true;
    } else if (state === 'ATTENDANCE') {
        btnEnrollStart.disabled = true;
        btnEnrollCancel.disabled = true;
        btnAttendStart.disabled = true;
        btnAttendStop.disabled = false;
    }
}

// Initial State
setUiState('IDLE');

function startEnrollment() {
    const name = document.getElementById('enroll-name').value;
    const id = document.getElementById('enroll-id').value;

    if (!name || !id) {
        alert("Please enter both Name and ID");
        return;
    }

    fetch('/start_enrollment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: name, id: id }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            updateStatus("Enrollment Started - Follow on-screen instructions");
            setUiState('ENROLLMENT');
            startPolling();
        } else {
            alert(data.message);
        }
    });
}

function startAttendance() {
    fetch('/start_attendance', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        updateStatus("Attendance Mode Active");
        setUiState('ATTENDANCE');
        startPolling();
    });
}

function stopSystem() {
    fetch('/stop', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        updateStatus("System Stopped");
        setUiState('IDLE');
        stopPolling();
    });
}

function downloadCSV() {
    window.location.href = '/download_csv';
}

function downloadSessionCSV() {
    window.location.href = '/download_session_csv';
}

function updateStatus(msg) {
    const overlay = document.getElementById('status-overlay');
    if (overlay) overlay.textContent = msg;
}

// Polling for live attendance updates AND system status
let pollingInterval = null;

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(fetchData, 1000); // Check every second
}

function stopPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = null;
}

function fetchData() {
    // We can multiplex or just call multiple endpoints. 
    // To keep it simple, let's just get attendance data.
    // Ideally, we should check 'mode' from backend to auto-reset UI if enrollment finishes.
    
    fetch('/get_attendance_data')
    .then(response => response.json())
    .then(data => {
        updateAttendanceList(data.students);
        
        // Auto-reset UI if backend is stopped (e.g. Enrollment finished)
        if (data.mode === 'STOPPED' && btnAttendStop.disabled === false) { 
             // If we think we are in attendance mode but backend says STOPPED -> Reset
             setUiState('IDLE');
             updateStatus("System Stopped");
        }
        if (data.mode === 'STOPPED' && btnEnrollCancel.disabled === false) {
             // If we think we are in enrollment mode but backend says STOPPED -> Reset
             setUiState('IDLE');
             updateStatus("Enrollment Complete");
        }
    });
}

function updateAttendanceList(students) {
    const list = document.getElementById('attendance-list');
    const badge = document.getElementById('count-badge');
    
    if (!students) return; // Safety check

    badge.textContent = students.length;

    if (students.length === 0) {
        list.innerHTML = '<li class="empty-state">No attendance marked yet in this session.</li>';
        return;
    }

    // Rebuild list (simple approach for now)
    let html = '';
    students.forEach(student => {
        html += `
            <li>
                <span class="student-name"><strong>${student.name}</strong></span>
                <span class="student-id">${student.id}</span>
            </li>
        `;
    });
    list.innerHTML = html;
}
