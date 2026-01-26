async function analyze() {
    const codeInput = document.getElementById("code");
    const loadingSection = document.getElementById("loading");
    const resultsSection = document.getElementById("results");
    const analyzeBtn = document.getElementById("analyzeBtn");

    const code = codeInput.value.trim();
    if (!code) {
        alert("Please enter some Python code data to analyze.");
        return;
    }

    // UI Reset
    analyzeBtn.disabled = true;
    resultsSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");

    try {
        const res = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code })
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        const data = await res.json();

        // Update UI with results
        document.getElementById("time").innerText = data.time_complexity;
        document.getElementById("space").innerText = data.space_complexity;
        document.getElementById("cyclo").innerText = data.cyclomatic_complexity;
        document.getElementById("readability").innerText = data.readability_score;
        document.getElementById("suggestions").innerText = data.optimization_suggestions;

        resultsSection.classList.remove("hidden");
    } catch (err) {
        console.error(err);
        alert("Failed to analyze code. Make sure the backend is running.");
    } finally {
        loadingSection.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
}
