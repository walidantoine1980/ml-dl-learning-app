// ==========================================
// SCROLLYTELLING LOGIC
// ==========================================
const steps = document.querySelectorAll('.step-content');
const visLayers = document.querySelectorAll('.vis-layer');

let currentActiveStep = 1;

const observerOptions = {
    root: document.getElementById('scroll-container'),
    rootMargin: '-40% 0px -40% 0px',
    threshold: 0
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const stepId = parseInt(entry.target.getAttribute('data-step'));
            if (stepId !== currentActiveStep) {
                activateVisual(stepId);
            }
        }
    });
}, observerOptions);

steps.forEach(step => observer.observe(step));

function activateVisual(stepId) {
    currentActiveStep = stepId;
    
    steps.forEach(s => {
        if(parseInt(s.getAttribute('data-step')) === stepId) s.classList.remove('opacity-40');
        else s.classList.add('opacity-40');
    });

    visLayers.forEach((layer, index) => {
        const layerId = index + 1;
        if (layerId === stepId) {
            layer.classList.remove('opacity-0', 'pointer-events-none');
            layer.classList.add('opacity-100', 'pointer-events-auto');
            
            // Triggers
            if(stepId === 1) triggerVennAnimation();
            if(stepId === 6 && typeof resetKMeansAnimation === 'function') resetKMeansAnimation();
            if(stepId === 8 && typeof drawNNConnectionsFw === 'function') drawNNConnectionsFw();
            if(stepId === 9 && typeof drawNNConnectionsBw === 'function') drawNNConnectionsBw();
            if(stepId === 12) updateMedicalDiagnosis();
            
        } else {
            layer.classList.remove('opacity-100', 'pointer-events-auto');
            layer.classList.add('opacity-0', 'pointer-events-none');
        }
    });
}

// ==========================================
// VIS 1: VENN
// ==========================================
function triggerVennAnimation() {
    const ia = document.getElementById('venn-ia'), ml = document.getElementById('venn-ml'), dl = document.getElementById('venn-dl');
    if(!ia) return;
    [ia, ml, dl].forEach(el => { el.classList.remove('opacity-100', 'scale-100'); el.classList.add('opacity-0', 'scale-90'); });
    setTimeout(() => { ia.classList.remove('opacity-0', 'scale-90'); ia.classList.add('opacity-100', 'scale-100'); }, 100);
    setTimeout(() => { ml.classList.remove('opacity-0', 'scale-90'); ml.classList.add('opacity-100', 'scale-100'); }, 600);
    setTimeout(() => { dl.classList.remove('opacity-0', 'scale-90'); dl.classList.add('opacity-100', 'scale-100'); }, 1100);
}

// ==========================================
// VIS 2: REGRESSION
// ==========================================
const ctxReg = document.getElementById('regressionChart').getContext('2d');
const dataPointsReg = [{x: 20, y: 50}, {x: 40, y: 90}, {x: 60, y: 120}, {x: 80, y: 160}, {x: 100, y: 210}, {x: 120, y: 230}];
const wSlider = document.getElementById('wSlider'), bSlider = document.getElementById('bSlider');
const wValue = document.getElementById('wValue'), bValue = document.getElementById('bValue');

const chartReg = new Chart(ctxReg, {
    type: 'scatter',
    data: {
        datasets: [{ label: 'Maisons', data: dataPointsReg, backgroundColor: '#60a5fa', borderColor: '#3b82f6', pointRadius: 6 },
        { label: 'Modèle', type: 'line', data: [], borderColor: '#f472b6', borderWidth: 3, fill: false, pointRadius: 0, borderDash: [5, 5] }]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: { x: { min: 0, max: 140 }, y: { min: 0, max: 300 } },
        plugins: { legend: { display: false } }
    }
});

function updateRegressionLine() {
    let w = parseFloat(wSlider.value), b = parseFloat(bSlider.value);
    wValue.textContent = w.toFixed(2); bValue.textContent = b.toFixed(0);
    chartReg.data.datasets[1].data = [{ x: 0, y: b }, { x: 140, y: w * 140 + b }];
    chartReg.update();
}
wSlider.addEventListener('input', updateRegressionLine);
bSlider.addEventListener('input', updateRegressionLine);
updateRegressionLine();

// ==========================================
// VIS 3: LOSS FUNCTION (Parabola)
// ==========================================
const ctxLoss = document.getElementById('lossChart').getContext('2d');
const lossData = [];
for(let w = -5; w <= 5; w+=0.5) lossData.push({x: w, y: w*w});

new Chart(ctxLoss, {
    type: 'line',
    data: {
        datasets: [{ label: 'Erreur J(w)', data: lossData, borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderWidth: 3, fill: true, tension: 0.4, pointRadius: 0 }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: { x: { title: {display: true, text: 'Poids w', color:'#94a3b8'}, grid: {color: '#334155'} }, y: { display: false } },
        plugins: { legend: { display: false } }
    }
});

// ==========================================
// VIS 4: GRADIENT DESCENT
// ==========================================
const ctxGrad = document.getElementById('gradientChart').getContext('2d');
const lrSlider = document.getElementById('lrSlider');
const lrValue = document.getElementById('lrValue');
const btnGrad = document.getElementById('runGradientBtn');

let gradW = -4.5;
let gradChart = new Chart(ctxGrad, {
    type: 'line',
    data: {
        datasets: [
            { label: 'Erreur J(w)', data: lossData, borderColor: '#ef4444', borderWidth: 3, tension: 0.4, pointRadius: 0 },
            { type: 'scatter', label: 'Bille', data: [{x: gradW, y: gradW*gradW}], backgroundColor: '#eab308', borderColor: '#ca8a04', pointRadius: 10 }
        ]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: {min: -5, max: 5}, y: {display:false} }, plugins: { legend: { display: false } }, animation: { duration: 200 } }
});

lrSlider.addEventListener('input', () => lrValue.textContent = lrSlider.value);
btnGrad.addEventListener('click', () => {
    let lr = parseFloat(lrSlider.value);
    gradW = gradW - lr * (2 * gradW);
    if(gradW > 6) gradW = 6; if(gradW < -6) gradW = -6;
    
    gradChart.data.datasets[1].data = [{x: gradW, y: gradW*gradW}];
    gradChart.update();
    
    if(Math.abs(gradW) < 0.1) {
        btnGrad.textContent = "Minimum atteint !";
        setTimeout(() => { gradW = -4.5; btnGrad.textContent = "Lâcher la bille"; gradChart.data.datasets[1].data = [{x: gradW, y: gradW*gradW}]; gradChart.update(); }, 2000);
    }
});

// ==========================================
// VIS 5: CLASSIFICATION
// ==========================================
const ctxClass = document.getElementById('classificationChart').getContext('2d');
new Chart(ctxClass, {
    type: 'scatter',
    data: {
        datasets: [
            { label: 'Spams', data: Array.from({length: 30}, () => ({x: 60+Math.random()*40, y: 50+Math.random()*50})), backgroundColor: '#ef4444', pointRadius: 5 },
            { label: 'Valides', data: Array.from({length: 30}, () => ({x: 10+Math.random()*40, y: 10+Math.random()*40})), backgroundColor: '#22c55e', pointRadius: 5 },
            { label: 'Frontière', type: 'line', data: [{x: 0, y: 90}, {x: 100, y: 10}], borderColor: '#a855f7', borderWidth: 3, fill: false, pointRadius: 0, borderDash: [10, 5] }
        ]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: {min: 0, max: 100}, y: {min: 0, max: 100} }, plugins: { legend: { labels: { color: '#e2e8f0' } } } }
});

// ==========================================
// VIS 6: CLUSTERING
// ==========================================
const ctxCluster = document.getElementById('clusteringChart').getContext('2d');
const rawDataCluster = [
    ...Array.from({length: 15}, () => ({x: 20+Math.random()*20, y: 20+Math.random()*20})),
    ...Array.from({length: 15}, () => ({x: 70+Math.random()*20, y: 30+Math.random()*20})),
    ...Array.from({length: 15}, () => ({x: 45+Math.random()*20, y: 70+Math.random()*20}))
];

let clusteringChart = new Chart(ctxCluster, {
    type: 'scatter', data: { datasets: [{ label: 'Clients', data: rawDataCluster, backgroundColor: '#94a3b8', pointRadius: 6 }] },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: {min: 0, max: 100}, y: {min: 0, max: 100} }, plugins: { legend: { display: false } } }
});

let kmeansStep = 0;
document.getElementById('runKmeansBtn').addEventListener('click', () => {
    kmeansStep++;
    if(kmeansStep === 1) {
        clusteringChart.data.datasets = [
            { label: 'Groupe 1', data: rawDataCluster.slice(0,15), backgroundColor: '#3b82f6', pointRadius: 6 },
            { label: 'Groupe 2', data: rawDataCluster.slice(15,30), backgroundColor: '#ec4899', pointRadius: 6 },
            { label: 'Groupe 3', data: rawDataCluster.slice(30,45), backgroundColor: '#eab308', pointRadius: 6 }
        ];
        clusteringChart.update();
        document.getElementById('runKmeansBtn').textContent = 'Réinitialiser';
    } else {
        kmeansStep = 0; resetKMeansAnimation(); document.getElementById('runKmeansBtn').textContent = 'Étape Suivante';
    }
});
function resetKMeansAnimation() { clusteringChart.data.datasets = [{ label: 'Clients', data: rawDataCluster, backgroundColor: '#94a3b8', pointRadius: 6 }]; clusteringChart.update(); }

// ==========================================
// VIS 7: OVERFITTING
// ==========================================
const ctxOver = document.getElementById('overfitChart').getContext('2d');
const overfitPoints = [{x: 1, y: 2}, {x: 2, y: 3}, {x: 3, y: 5}, {x: 4, y: 4}, {x: 5, y: 6}, {x: 6, y: 8}, {x: 7, y: 7}];

const overfitChart = new Chart(ctxOver, {
    type: 'scatter',
    data: {
        datasets: [
            { label: 'Données', data: overfitPoints, backgroundColor: '#38bdf8', pointRadius: 6 },
            { label: 'Modèle', type: 'line', data: [], borderColor: '#22c55e', borderWidth: 3, fill: false, pointRadius: 0, tension: 0.4 }
        ]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: {min: 0, max: 8}, y: {min: 0, max: 10} }, plugins: { legend: { display: false } } }
});

document.querySelectorAll('.fit-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const type = e.target.getAttribute('data-fit');
        let lineData = []; let color = ''; let tension = 0;
        if(type === 'under') { lineData = [{x: 0, y: 1}, {x: 8, y: 9}]; color = '#64748b'; tension = 0; }
        if(type === 'good') { lineData = [{x: 0, y: 1}, {x: 2, y: 3}, {x: 4, y: 4.5}, {x: 6, y: 7.5}, {x: 8, y: 8}]; color = '#22c55e'; tension = 0.4; }
        if(type === 'over') { lineData = [{x:0,y:0}, {x:1,y:2}, {x:1.5,y:0}, {x:2,y:3}, {x:2.5,y:7}, {x:3,y:5}, {x:3.5,y:1}, {x:4,y:4}, {x:4.5,y:8}, {x:5,y:6}, {x:5.5,y:2}, {x:6,y:8}, {x:6.5,y:10}, {x:7,y:7}, {x:8,y:0}]; color = '#ef4444'; tension = 0.4; }
        overfitChart.data.datasets[1].data = lineData; overfitChart.data.datasets[1].borderColor = color; overfitChart.data.datasets[1].tension = tension; overfitChart.update();
    });
});
document.querySelector('.fit-btn[data-fit="good"]').click();

// ==========================================
// VIS 8 & 9: NEURAL NETWORK (FW & BW)
// ==========================================
function drawNNConnectionsFw() {
    const nnCanvasFw = document.getElementById('nnCanvasFw'); const nnCtxFw = nnCanvasFw.getContext('2d');
    const container = document.getElementById('nn-container-fw'); if(!container) return;
    nnCanvasFw.width = container.clientWidth; nnCanvasFw.height = container.clientHeight;
    const inputs = document.querySelectorAll('.input-neuron'); const hiddens = document.querySelectorAll('.hidden-neuron-fw'); const outputs = document.querySelectorAll('.output-neuron-fw');
    nnCtxFw.clearRect(0, 0, nnCanvasFw.width, nnCanvasFw.height); nnCtxFw.lineWidth = 1; nnCtxFw.strokeStyle = 'rgba(51, 65, 85, 0.5)';
    drawLines(inputs, hiddens, nnCtxFw, nnCanvasFw); drawLines(hiddens, outputs, nnCtxFw, nnCanvasFw);
}

window.fireNetworkFw = function(inputIndex) {
    const inputs = document.querySelectorAll('.input-neuron'); const hiddens = document.querySelectorAll('.hidden-neuron-fw'); const outputs = document.querySelectorAll('.output-neuron-fw');
    inputs.forEach(el => el.classList.remove('active-purple', 'active-pink')); inputs[inputIndex].classList.add('active-purple');
    setTimeout(() => { hiddens.forEach(el => { el.classList.add('active-purple'); el.textContent = (Math.random()).toFixed(1); }); }, 400);
    setTimeout(() => { hiddens.forEach(el => el.classList.remove('active-purple')); outputs.forEach(el => { el.classList.add('active-pink'); el.textContent = '1.0'; }); }, 1000);
    setTimeout(() => { outputs.forEach(el => el.classList.remove('active-pink')); inputs[inputIndex].classList.remove('active-purple'); hiddens.forEach(el => el.textContent = ''); outputs.forEach(el => el.textContent = '?'); }, 2500);
};

function drawNNConnectionsBw() {
    const nnCanvasBw = document.getElementById('nnCanvasBw'); const nnCtxBw = nnCanvasBw.getContext('2d');
    const container = document.getElementById('nn-container-bw'); if(!container) return;
    nnCanvasBw.width = container.clientWidth; nnCanvasBw.height = container.clientHeight;
    const inputs = document.querySelectorAll('.input-neuron-bw'); const hiddens = document.querySelectorAll('.hidden-neuron-bw'); const errNode = document.querySelectorAll('#vis-9 .bg-red-500\\/20');
    nnCtxBw.clearRect(0, 0, nnCanvasBw.width, nnCanvasBw.height); nnCtxBw.lineWidth = 1; nnCtxBw.strokeStyle = 'rgba(51, 65, 85, 0.5)';
    drawLines(inputs, hiddens, nnCtxBw, nnCanvasBw); drawLines(hiddens, errNode, nnCtxBw, nnCanvasBw);
}

window.fireNetworkBw = function() {
    const inputs = document.querySelectorAll('.input-neuron-bw'); const hiddens = document.querySelectorAll('.hidden-neuron-bw');
    setTimeout(() => { hiddens.forEach(el => { el.style.borderColor = '#ef4444'; el.style.boxShadow = '0 0 20px #ef4444'; el.textContent = '-0.1'; }); }, 400);
    setTimeout(() => { hiddens.forEach(el => { el.style.borderColor = ''; el.style.boxShadow = ''; }); inputs.forEach(el => { el.style.borderColor = '#ef4444'; el.style.boxShadow = '0 0 20px #ef4444'; }); }, 1000);
    setTimeout(() => { inputs.forEach(el => { el.style.borderColor = ''; el.style.boxShadow = ''; }); hiddens.forEach(el => el.textContent = ''); }, 2000);
};

function drawLines(fromNodes, toNodes, ctx, canvas) {
    fromNodes.forEach(from => {
        const fRect = from.getBoundingClientRect(); const cRect = canvas.getBoundingClientRect();
        const x1 = fRect.right - cRect.left; const y1 = fRect.top + fRect.height/2 - cRect.top;
        toNodes.forEach(to => {
            const tRect = to.getBoundingClientRect(); const x2 = tRect.left - cRect.left; const y2 = tRect.top + tRect.height/2 - cRect.top;
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        });
    });
}
window.addEventListener('resize', () => { drawNNConnectionsFw(); drawNNConnectionsBw(); });

// ==========================================
// VIS 11: NLP
// ==========================================
const ctxNlp = document.getElementById('nlpChart').getContext('2d');
new Chart(ctxNlp, {
    type: 'scatter',
    data: {
        datasets: [
            { label: 'Royauté', data: [{x: 80, y: 80}, {x: 85, y: 75}], backgroundColor: '#f472b6', pointRadius: 8 },
            { label: 'Animaux', data: [{x: 20, y: 20}, {x: 25, y: 15}], backgroundColor: '#3b82f6', pointRadius: 8 },
            { label: 'Tech', data: [{x: 80, y: 20}, {x: 85, y: 25}], backgroundColor: '#22c55e', pointRadius: 8 }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { display: false, min: 0, max: 100 }, y: { display: false, min: 0, max: 100 } },
        animation: {
            onComplete: function() {
                const ctx = this.ctx; ctx.font = "bold 14px Inter"; ctx.fillStyle = "#cbd5e1"; ctx.textAlign = "center";
                const texts = ["Roi", "Reine", "Chien", "Chat", "Ordinateur", "Code"];
                let txtIdx = 0;
                this.data.datasets.forEach((dataset) => {
                    const meta = this.getDatasetMeta(this.data.datasets.indexOf(dataset));
                    meta.data.forEach((bar) => { ctx.fillText(texts[txtIdx++], bar.x, bar.y - 15); });
                });
            }
        }
    }
});


// ==========================================
// VIS 12: MEDICAL DIAGNOSIS (ML PLAYGROUND)
// ==========================================
const ageSlider = document.getElementById('ageSlider');
const cholSlider = document.getElementById('cholSlider');
const bpSlider = document.getElementById('bpSlider');
const medGaugeFill = document.getElementById('medGaugeFill');
const medScore = document.getElementById('medScore');
const medLabel = document.getElementById('medLabel');

function updateMedicalDiagnosis() {
    const age = parseInt(ageSlider.value);
    const chol = parseInt(cholSlider.value);
    const bp = parseInt(bpSlider.value);

    document.getElementById('ageVal').textContent = age + " ans";
    document.getElementById('cholVal').textContent = chol;
    document.getElementById('bpVal').textContent = bp;

    // Simulated Logistic Regression Weights
    // Formula: z = w1*age + w2*chol + w3*bp + bias
    // We normalize ranges to keep z reasonable
    const ageNorm = (age - 20) / 70; // 0 to 1
    const cholNorm = (chol - 100) / 250; // 0 to 1
    const bpNorm = (bp - 80) / 120; // 0 to 1

    const z = (ageNorm * 3.5) + (cholNorm * 4.0) + (bpNorm * 2.5) - 4.5;
    
    // Sigmoid function
    const probability = 1 / (1 + Math.exp(-z));
    const percentage = Math.round(probability * 100);

    medScore.textContent = percentage + "%";

    // Gauge rotation mapping (0% -> 0deg, 100% -> 180deg)
    const rotation = percentage * 1.8;
    medGaugeFill.style.transform = `rotate(${rotation}deg)`;

    // Color logic
    if (percentage < 30) {
        medGaugeFill.style.borderColor = "#22c55e"; // green
        medLabel.textContent = "Risque Faible";
        medLabel.className = "text-sm font-bold uppercase tracking-wider text-green-400";
    } else if (percentage < 70) {
        medGaugeFill.style.borderColor = "#eab308"; // yellow
        medLabel.textContent = "Risque Modéré";
        medLabel.className = "text-sm font-bold uppercase tracking-wider text-yellow-400";
    } else {
        medGaugeFill.style.borderColor = "#ef4444"; // red
        medLabel.textContent = "Risque Élevé";
        medLabel.className = "text-sm font-bold uppercase tracking-wider text-red-400";
    }
}

ageSlider.addEventListener('input', updateMedicalDiagnosis);
cholSlider.addEventListener('input', updateMedicalDiagnosis);
bpSlider.addEventListener('input', updateMedicalDiagnosis);


// ==========================================
// VIS 13: SENTIMENT ANALYSIS (DL PLAYGROUND)
// ==========================================
const analyzeBtn = document.getElementById('analyzeBtn');
const nlpInput = document.getElementById('nlpInput');
const nlpResult = document.getElementById('nlpResult');
const nlpHighlighted = document.getElementById('nlpHighlighted');

// Fake Sentiment Dictionary
const positiveWords = ['incroyable', 'super', 'génial', 'bien', 'bon', 'bonne', 'parfait', 'excellent', 'adore', 'beau', 'magnifique', 'top', 'rapide', 'efficace'];
const negativeWords = ['catastrophique', 'nul', 'mauvais', 'faible', 'lent', 'horrible', 'déçu', 'problème', 'bug', 'pire', 'cher', 'inutile', 'déteste'];

analyzeBtn.addEventListener('click', () => {
    const text = nlpInput.value.trim();
    if (!text) return;

    nlpResult.classList.remove('hidden');
    
    // Process text
    let posCount = 0;
    let negCount = 0;
    let totalWords = 0;

    // We split by punctuation and spaces, but keep punctuation in output using regex
    const words = text.split(/(\b[^\s]+\b)/); 
    
    let highlightedHTML = "";

    words.forEach(w => {
        if (!w.trim() || w.length === 1) {
            highlightedHTML += w;
            return;
        }
        
        totalWords++;
        const lowerW = w.toLowerCase();
        
        let isPos = false;
        let isNeg = false;
        
        positiveWords.forEach(pw => { if (lowerW.includes(pw)) isPos = true; });
        negativeWords.forEach(nw => { if (lowerW.includes(nw)) isNeg = true; });

        if (isPos) {
            posCount++;
            highlightedHTML += `<span class="bg-green-500/20 text-green-400 px-1 rounded mx-0.5 font-bold">${w}</span>`;
        } else if (isNeg) {
            negCount++;
            highlightedHTML += `<span class="bg-red-500/20 text-red-400 px-1 rounded mx-0.5 font-bold">${w}</span>`;
        } else {
            highlightedHTML += w;
        }
    });

    nlpHighlighted.innerHTML = highlightedHTML;

    // Calculate scores
    let posScore = 33;
    let negScore = 33;
    let neuScore = 34;

    if (posCount > 0 || negCount > 0) {
        // Simple heuristic for demo
        const totalSentiments = posCount + negCount;
        const posRatio = posCount / totalSentiments;
        const negRatio = negCount / totalSentiments;

        // Base confidence based on how many sentiment words relative to total words
        const confidence = Math.min((totalSentiments / Math.max(1, totalWords)) * 2, 0.8); 
        
        posScore = Math.round(posRatio * confidence * 100) + Math.round((1-confidence)*33);
        negScore = Math.round(negRatio * confidence * 100) + Math.round((1-confidence)*33);
        neuScore = 100 - posScore - negScore;
    }

    // Update bars
    document.getElementById('nlpNegBar').style.width = negScore + "%";
    document.getElementById('nlpNeuBar').style.width = neuScore + "%";
    document.getElementById('nlpPosBar').style.width = posScore + "%";

    document.getElementById('nlpNegScore').textContent = negScore + "%";
    document.getElementById('nlpNeuScore').textContent = neuScore + "%";
    document.getElementById('nlpPosScore').textContent = posScore + "%";
});


// Initialization
activateVisual(1);
