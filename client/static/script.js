// Use relative URLs — page is served by the same FastAPI server
const API = "";


// ============================
// Upload Document
// ============================

async function uploadPDF() {

    const file = document.getElementById("pdfFile").files[0];
    const statusEl = document.getElementById("uploadStatus");

    if (!file) {
        statusEl.innerHTML =
            `<span class="status-error">⚠️ Please choose a PDF or DOCX file.</span>`;
        return;
    }

    statusEl.innerHTML =
        `<span class="status-loading">⏳ Uploading <strong>${file.name}</strong>...</span>`;

    const formData = new FormData();
    formData.append("file", file);

    try {

        const res = await fetch(API + "/upload/", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            if (res.status === 409) {
                statusEl.innerHTML = `
                    <span class="status-warning">
                        ⚠️ <strong>File Already Exists</strong><br>
                        A file named <strong>${file.name}</strong> has already been uploaded.<br>
                        Do you want to replace it?<br>
                        <button class="btn btn-primary" onclick="confirmReplace()" style="margin-top: 10px; background: linear-gradient(135deg, #ea580c, #f97316); box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4); border: none; cursor: pointer;">Yes, Replace It</button>
                    </span>
                `;
            } else {
                statusEl.innerHTML =
                    `<span class="status-error">❌ ${data.detail}</span>`;
            }
            return;
        }

        statusEl.innerHTML = `
            <span class="status-success">
                ✅ <strong>Upload Successful</strong><br>
                File: ${data.file_name}<br>
                Chunks Created: ${data.total_chunks}<br>
                <em>Refreshing document list...</em>
            </span>
        `;

        setTimeout(loadDocuments, 500);

    }

    catch (error) {

        console.error(error);

        statusEl.innerHTML =
            `<span class="status-error">❌ Upload failed. Is the server running?</span>`;

    }

}

async function confirmReplace() {
    const file = document.getElementById("pdfFile").files[0];
    const statusEl = document.getElementById("uploadStatus");

    if (!file) {
        statusEl.innerHTML =
            `<span class="status-error">⚠️ No file selected for replacement.</span>`;
        return;
    }

    statusEl.innerHTML =
        `<span class="status-loading">⏳ Replacing <strong>${file.name}</strong>...</span>`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(API + "/upload/?replace=true", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            statusEl.innerHTML =
                `<span class="status-error">❌ ${data.detail}</span>`;
            return;
        }

        statusEl.innerHTML = `
            <span class="status-success">
                ✅ <strong>Replacement Successful</strong><br>
                File: ${data.file_name}<br>
                Chunks Created: ${data.total_chunks}<br>
                <em>Refreshing document list...</em>
            </span>
        `;

        setTimeout(loadDocuments, 500);
    }
    catch (error) {
        console.error(error);
        statusEl.innerHTML =
            `<span class="status-error">❌ Replacement failed. Is the server running?</span>`;
    }
}



// ============================
// Load Uploaded Documents
// ============================

async function loadDocuments() {

    const list = document.getElementById("documentList");

    list.innerHTML =
        `<li class="loading-item">⏳ Loading documents...</li>`;

    try {

        const res = await fetch(API + "/documents");

        if (!res.ok) {
            throw new Error("Server Error");
        }

        const data = await res.json();

        const selectedInput = document.getElementById("selectedDocument");
        const currentSelected = selectedInput ? selectedInput.value : "";

        list.innerHTML = "";

        // All Documents option
        const allLi = document.createElement("li");
        allLi.className = `doc-item${!currentSelected ? " selected-document" : ""}`;
        allLi.innerHTML = `
            <span class="doc-icon">📚</span>
            <span class="doc-name">All Documents</span>
        `;
        allLi.addEventListener("click", function () {
            selectDocument(this, "");
        });
        list.appendChild(allLi);

        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {

                const li = document.createElement("li");
                li.className = `doc-item${currentSelected === doc ? " selected-document" : ""}`;

                const ext = doc.split(".").pop().toUpperCase();
                const icon = ext === "PDF" ? "\uD83D\uDCD5" : "\uD83D\uDCD8";

                li.innerHTML = `
                    <span class="doc-icon">${icon}</span>
                    <span class="doc-name">${doc}</span>
                    <span class="doc-badge">${ext}</span>
                    <button
                        class="btn-delete"
                        title="Delete ${doc}"
                        aria-label="Delete ${doc}"
                    >🗑</button>
                `;

                li.addEventListener("click", function (e) {
                    if (e.target.closest(".btn-delete")) return;
                    selectDocument(this, doc);
                });

                // Attach the click handler after rendering so `li` is in scope.
                li.querySelector(".btn-delete").addEventListener("click", function (e) {
                    e.stopPropagation();
                    deleteDocument(this, doc);
                });

                list.appendChild(li);

            });
        }

        const countEl = document.getElementById("docCount");

        if (countEl) {

            countEl.textContent = data.documents ? data.documents.length : 0;

        }

    }

    catch (error) {

        console.error(error);

        list.innerHTML =
            `<li class="error-item">⚠️ Unable to load documents.</li>`;

    }

}


function selectDocument(element, documentName) {

    document.querySelectorAll(".doc-item")
        .forEach(item => {
            item.classList.remove("selected-document");
        });

    if (element) {
        element.classList.add("selected-document");
    }

    const input = document.getElementById("selectedDocument");
    if (input) {
        input.value = documentName || "";
    }

    console.log("Selected document:", documentName);
}


async function askQuestion() {

    const question = document.getElementById("question").value.trim();

    const topK = parseInt(
        document.getElementById("topK").value
    );

    const answerEl = document.getElementById("answer");
    const responseTimeEl = document.getElementById("responseTime");
    const askBtn = document.getElementById("askBtn");

    if (!question) {

        answerEl.innerHTML =
            `<span class="status-error">⚠️ Please enter a question first.</span>`;

        return;

    }

    askBtn.disabled = true;
    askBtn.innerHTML = "⏳ Thinking...";

    responseTimeEl.innerHTML = "⏱️ Response Time: Calculating...";

    answerEl.innerHTML =
        `<span class="status-loading">🤖 Searching .....</span>`;

    try {

        // Get the selected document BEFORE fetch()
        const selectedDocument =
            document.getElementById("selectedDocument").value;

        const res = await fetch(API + "/chat/", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                question: question,

                top_k: topK,

                selected_document: selectedDocument || null

            })

        });

        const data = await res.json();

        console.log(data);
        console.log(data.metadata);
        console.log(data.metadata?.response_time);

        askBtn.disabled = false;
        askBtn.innerHTML = "🔍 Ask AI";

        if (!res.ok) {

            answerEl.innerHTML =
                `<span class="status-error">❌ ${data.detail || "Error getting answer."}</span>`;

            return;

        }

        let html = "";

        // ===========================
        // Retrieval Information
        // ===========================
        if (data.metadata) {

            if (data.metadata.response_time !== undefined) {

                const sec = Number(data.metadata.response_time).toFixed(3);

                responseTimeEl.innerHTML =
                    `⏱️ Response Time: <strong>${sec} sec</strong>` +
                    (data.metadata.cached
                        ? ` <span class="badge-cached">Cached</span>`
                        : "");

            } else {

                responseTimeEl.innerHTML = "⏱️ Response Time: --";

            }

            html += `
                <div class="retrieval-info">

                    <strong>Matching sections selected:</strong>
                    ${data.metadata.top_k}

                    <br>

                    <strong>Retrieved matching sections:</strong>
                    ${data.metadata.retrieval_count}

                </div>
            `;

        }

        // ===========================
        // Answer
        // ===========================
        html += `
            <div class="answer-box">

                <div class="answer-label">
                    🤖 AI Answer
                </div>

                <div class="answer-text">
                    ${data.answer}
                </div>

            </div>
        `;

        // If answer not found don't show sources
        if (
            data.answer ===
            "I couldn't find this information in the uploaded company policies."
        ) {

            answerEl.innerHTML = html;
            return;

        }

        // ===========================
        // Sources
        // ===========================
        if (data.sources && data.sources.length > 0) {

            html += `
                <div class="sources-section">

                    <div class="sources-label">
                        🗄️ Relevant Sections
                    </div>

                    <div class="sources-list">
            `;

            data.sources.forEach(src => {

                const score = (src.score * 100).toFixed(1);

                let scoreClass = "score-low";

                if (src.score >= 0.70)
                    scoreClass = "score-high";
                else if (src.score >= 0.40)
                    scoreClass = "score-mid";

                html += `
                    <div class="source-chunk">

                        <div class="source-meta">

                            <span class="chunk-badge ${scoreClass}">
                                ⚡ ${score}% Match
                            </span>

                            <span class="source-doc">
                                📄 ${src.document}
                            </span>

                            <span class="chunk-id">
                                ID: ${src.point_id}
                            </span>

                        </div>

                        <p class="chunk-preview">
                            ${src.text}
                        </p>

                    </div>
                `;

            });

            html += `
                    </div>
                </div>
            `;

        }

        answerEl.innerHTML = html;

    }
    catch (error) {

        console.error(error);

        askBtn.disabled = false;
        askBtn.innerHTML = "🔍 Ask AI";

        responseTimeEl.innerHTML = "⏱️ Response Time: --";

        answerEl.innerHTML =
            `<span class="status-error">❌ Failed to get a response. Is the server running?</span>`;

    }

}



// ============================
// Delete Document
// ============================

/**
 * Entry point called by every delete button (both Jinja-rendered and
 * JS-rendered rows).  Immediately disables the button to prevent double-
 * clicks, then opens a confirmation modal before making the API call.
 *
 * @param {HTMLButtonElement} btn      - The clicked delete button element.
 * @param {string}            filename - The exact filename to delete.
 *                                       May be a JSON-encoded string from
 *                                       the Jinja template (e.g. '"foo.pdf"');
 *                                       we parse it to get the raw string.
 */
async function deleteDocument(btn, filename) {

    // Jinja passes the filename through tojson which wraps it in quotes;
    // JS addEventListener calls pass the raw string.  Handle both.
    try { filename = JSON.parse(filename); } catch (_) { /* raw string — fine */ }

    // Disable the button immediately so repeated clicks cannot fire.
    btn.disabled = true;
    btn.textContent = "⏳";

    const confirmed = await showDeleteModal(filename);

    if (!confirmed) {
        // User cancelled — restore the button state.
        btn.disabled = false;
        btn.textContent = "\uD83D\uDDD1";
        return;
    }

    // Show spinner in the button while the request is in flight.
    btn.classList.add("loading");
    btn.textContent = "⟳";

    try {
        const res = await fetch(
            API + "/upload/" + encodeURIComponent(filename),
            { method: "DELETE" }
        );

        const data = await res.json();

        if (!res.ok) {
            // Restore button so the user can retry.
            btn.disabled = false;
            btn.classList.remove("loading");
            btn.textContent = "\uD83D\uDDD1";
            showToast("error", `\u274C ${data.detail || "Deletion failed."}`);
            return;
        }

        // Optimistically remove the row from the DOM — no page reload needed.
        const li = btn.closest("li");
        if (li) {
            li.style.transition = "opacity 0.25s ease, transform 0.25s ease";
            li.style.opacity = "0";
            li.style.transform = "translateX(20px)";
            setTimeout(() => {
                li.remove();
                // Update the count badge.
                const countEl = document.getElementById("docCount");
                if (countEl) {
                    const current = parseInt(countEl.textContent) || 0;
                    countEl.textContent = Math.max(0, current - 1);
                }
                // Show empty state if list is now empty.
                const list = document.getElementById("documentList");
                if (list && list.querySelectorAll(".doc-item").length === 0) {
                    list.innerHTML =
                        `<li class="empty-item">\uD83D\uDCED No documents uploaded yet.</li>`;
                }
            }, 260);
        }

        showToast("success", `\u2705 "${filename}" deleted successfully.`);

    } catch (error) {
        console.error("Delete error:", error);
        btn.disabled = false;
        btn.classList.remove("loading");
        btn.textContent = "\uD83D\uDDD1";
        showToast("error", "\u274C Deletion failed. Is the server running?");
    }
}


// ============================
// Confirmation Modal
// ============================

/**
 * Shows a glassmorphism modal asking the user to confirm deletion.
 * Returns a Promise<boolean> that resolves true (confirm) or false (cancel).
 *
 * @param {string} filename - Display name shown inside the modal.
 */
function showDeleteModal(filename) {
    return new Promise(resolve => {

        // Build the backdrop and modal markup.
        const backdrop = document.createElement("div");
        backdrop.className = "modal-backdrop";
        backdrop.id = "deleteModalBackdrop";

        backdrop.innerHTML = `
            <div class="modal" role="dialog" aria-modal="true"
                 aria-labelledby="modalTitle">
                <div class="modal-icon">\uD83D\uDDD1\uFE0F</div>
                <div class="modal-title" id="modalTitle">Delete Document?</div>
                <p class="modal-body">
                    This action <strong>cannot be undone</strong>. All embeddings,
                    metadata, and the file on disk will be permanently removed.
                </p>
                <div class="modal-filename">${filename}</div>
                <div class="modal-actions">
                    <button class="btn-modal-cancel" id="modalCancelBtn">Cancel</button>
                    <button class="btn-modal-confirm" id="modalConfirmBtn">
                        Yes, Delete It
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(backdrop);

        const confirmBtn = backdrop.querySelector("#modalConfirmBtn");
        const cancelBtn = backdrop.querySelector("#modalCancelBtn");

        function cleanup(result) {
            backdrop.remove();
            resolve(result);
        }

        confirmBtn.addEventListener("click", () => {
            confirmBtn.disabled = true;
            confirmBtn.textContent = "Deleting...";
            cleanup(true);
        });

        cancelBtn.addEventListener("click", () => cleanup(false));

        // Close on backdrop click (but not on modal content click).
        backdrop.addEventListener("click", e => {
            if (e.target === backdrop) cleanup(false);
        });

        // Close on Escape key.
        function onKey(e) {
            if (e.key === "Escape") {
                document.removeEventListener("keydown", onKey);
                cleanup(false);
            }
        }
        document.addEventListener("keydown", onKey);
    });
}


// ============================
// Toast Notifications
// ============================

/**
 * Appends a transient toast to #toastContainer.
 *
 * @param {"success"|"error"} type    - Controls colour styling.
 * @param {string}            message - Text to display.
 * @param {number}            [ttl]   - Auto-dismiss delay in ms (default 4000).
 */
function showToast(type, message, ttl = 4000) {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const icon = type === "success" ? "\u2705" : "\u26A0\uFE0F";

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-text">${message}</span>
    `;

    container.appendChild(toast);

    // Auto-dismiss with a slide-out animation.
    setTimeout(() => {
        toast.classList.add("removing");
        toast.addEventListener("animationend", () => toast.remove(), { once: true });
    }, ttl);
}



document.addEventListener("DOMContentLoaded", () => {

    loadDocuments();

    // Wire up delete buttons that were server-side rendered by Jinja2.
    // These have a data-filename attribute but no onclick, so we use
    // event delegation on the document list container.
    document.addEventListener("click", e => {
        const btn = e.target.closest(".btn-delete");
        if (!btn) return;                         // click was elsewhere
        if (btn.dataset.filename === undefined) return;  // JS-rendered btn already has its own listener
        deleteDocument(btn, btn.dataset.filename);
    });

    document
        .getElementById("question")
        .addEventListener("keydown", function (e) {

            if (e.key === "Enter" && !e.shiftKey) {

                e.preventDefault();

                askQuestion();

            }

        });

});


