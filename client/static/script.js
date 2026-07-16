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

            statusEl.innerHTML =
                `<span class="status-error">❌ ${data.detail}</span>`;

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



// ============================
// Load Uploaded Documents
// ============================

async function loadDocuments() {

    const list = document.getElementById("documentList");

    list.innerHTML =
        `<li class="loading-item">⏳ Loading documents...</li>`;

    try {

        const res = await fetch(API + "/upload/documents");

        if (!res.ok) {
            throw new Error("Server Error");
        }

        const data = await res.json();

        list.innerHTML = "";

        if (!data.documents || data.documents.length === 0) {

            list.innerHTML =
                `<li class="empty-item">📭 No documents uploaded yet.</li>`;

            return;
        }

        data.documents.forEach(doc => {

            const li = document.createElement("li");

            li.className = "doc-item";

            const ext = doc.split(".").pop().toUpperCase();

            const icon = ext === "PDF"
                ? "📕"
                : "📘";

            li.innerHTML = `
                <span class="doc-icon">${icon}</span>
                <span class="doc-name">${doc}</span>
                <span class="doc-badge">${ext}</span>
            `;

            list.appendChild(li);

        });

        const countEl = document.getElementById("docCount");

        if (countEl) {

            countEl.textContent = data.documents.length;

        }

    }

    catch (error) {

        console.error(error);

        list.innerHTML =
            `<li class="error-item">⚠️ Unable to load documents.</li>`;

    }

}



// ============================
// Ask Question
// ============================

async function askQuestion() {

    const question = document.getElementById("question").value.trim();

    const topK = parseInt(
        document.getElementById("topK").value
    );

    const answerEl = document.getElementById("answer");

    const askBtn = document.getElementById("askBtn");

    if (!question) {

        answerEl.innerHTML =
            `<span class="status-error">⚠️ Please enter a question first.</span>`;

        return;

    }

    askBtn.disabled = true;
    askBtn.innerHTML = "⏳ Thinking...";

    answerEl.innerHTML =
        `<span class="status-loading">🤖 Retrieving relevant documents...</span>`;

    try {

        const res = await fetch(API + "/chat/", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                question: question,

                top_k: topK

            })

        });

        const data = await res.json();

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

            html += `
                <div class="retrieval-info">

                    <strong>Top K Selected:</strong>
                    ${data.metadata.top_k}

                    <br>

                    <strong>Chunks Retrieved:</strong>
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

                        🗄️ Retrieved Chunks

                    </div>

                    <div class="sources-list">
            `;

            data.sources.forEach(src => {

                const score =
                    (src.score * 100).toFixed(1);

                let scoreClass = "score-low";

                if (src.score >= 0.70)
                    scoreClass = "score-high";

                else if (src.score >= 0.40)
                    scoreClass = "score-mid";

                html += `

                    <div class="source-chunk">

                        <div class="source-meta">

                            <span class="chunk-badge chunk-num">

                                Chunk #${src.chunk_number}

                            </span>

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

        answerEl.innerHTML =
            `<span class="status-error">❌ Failed to get a response. Is the server running?</span>`;

    }

}



// ============================
// Enter to Ask
// ============================

document.addEventListener("DOMContentLoaded", () => {

    loadDocuments();

    document
        .getElementById("question")
        .addEventListener("keydown", function (e) {

            if (e.key === "Enter" && !e.shiftKey) {

                e.preventDefault();

                askQuestion();

            }

        });

});


