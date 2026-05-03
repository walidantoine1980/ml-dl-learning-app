// ==========================================
// SCROLLYTELLING LOGIC (Intersection Observer)
// ==========================================
const steps = document.querySelectorAll('.step-content');
const visLayers = document.querySelectorAll('.vis-layer');

let currentActiveStep = 1;

const observerOptions = {
    root: document.getElementById('scroll-container'),
    rootMargin: '-40% 0px -40% 0px', // Trigger when element is roughly in the middle
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
    
    // Dim inactive text slightly
    steps.forEach(s => {
        if(parseInt(s.getAttribute('data-step')) === stepId) {
            s.classList.remove('opacity-40');
        } else {
            s.classList.add('opacity-40');
        }
    });

    // Crossfade Visuals
    visLayers.forEach((layer, index) => {
        const layerId = index + 1;
        if (layerId === stepId) {
            layer.classList.remove('opacity-0', 'pointer-events-none');
            layer.classList.add('opacity-100', 'pointer-events-auto');
            
            // Trigger specific animations when coming into view
            if(stepId === 1) triggerVennAnimation();
            if(stepId === 4 && clusteringChart) resetKMeansAnimation();
            if(stepId === 5) drawNNConnections();
            
        } else {
            layer.classList.remove('opacity-100', 'pointer-events-auto');
            layer.classList.add('opacity-0', 'pointer-events-none');
        }
    });
}

// ==========================================
// VIS 1: VENN DIAGRAM ANIMATION
// ==========================================
function triggerVennAnimation() {
    const ia = document.getElementById('venn-ia');
    const ml = document.getElementById('venn-ml');
    const dl = document.getElementById('venn-dl');
    
    [ia, ml, dl].forEach(el => {
        el.classList.remove('opacity-100', 'scale-100');
        el.classList.add('opacity-0', 'scale-90');
    });

    setTimeout(() => { ia.classList.remove('opacity-0', 'scale-90'); ia.classList.add('opacity-100', 'scale-100'); }, 100);
    setTimeout(() => { ml.classList.remove('opacity-0', 'scale-90'); ml.classList.add('opacity-100', 'scale-100'); }, 600);
    setTimeout(() => { dl.classList.remove('opacity-0', 'scale-90'); dl.classList.add('opacity-100', 'scale-100'); }, 1100);
}

// ==========================================
// VIS 2: LINEAR REGRESSION
// ==========================================
const ctxReg = document.getElementById('regressionChart').getContext('2d');
const wSlider = document.getElementById('wSlider');
const bSlider = document.getElementById('bSlider');
const wValue = document.getElementById('wValue');
const bValue = document.getElementById('bValue');
const lossValue = document.getElementById('lossValue');

const dataPointsReg = [
    {x: 20, y: 50}, {x: 40, y: 90}, {x: 60, y: 120}, 
    {x: 80, y: 160}, {x: 100, y: 210}, {x: 120, y: 230}
];

let currentW = parseFloat(wSlider.value);
let currentB = parseFloat(bSlider.value);

const chartReg = new Chart(ctxReg, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Maisons réelles',
            data: dataPointsReg,
            backgroundColor: '#60a5fa', borderColor: '#3b82f6',
            borderWidth: 2, pointRadius: 6, pointHoverRadius: 8
        },
        {
            label: 'Modèle IA (y = wx + b)',
            type: 'line', data: [],
            borderColor: '#f472b6', borderWidth: 3, fill: false,
            pointRadius: 0, borderDash: [5, 5]
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
            x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Taille (m²)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 140 },
            y: { title: { display: true, text: 'Prix (k€)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 300 }
        },
        plugins: { legend: { labels: { color: '#e2e8f0' } } }
    }
});

function updateRegressionLine() {
    currentW = parseFloat(wSlider.value);
    currentB = parseFloat(bSlider.value);
    wValue.textContent = currentW.toFixed(2);
    bValue.textContent = currentB.toFixed(0);

    const lineData = [{ x: 0, y: currentB }, { x: 140, y: currentW * 140 + currentB }];

    let mse = 0;
    dataPointsReg.forEach(pt => {
        const pred = currentW * pt.x + currentB;
        mse += Math.pow(pt.y - pred, 2);
    });
    mse = Math.sqrt(mse / dataPointsReg.length); // RMSE for better scaling
    
    lossValue.textContent = mse.toFixed(1);
    if (mse < 15) lossValue.className = "text-green-400 font-mono text-sm font-bold";
    else if (mse < 30) lossValue.className = "text-yellow-400 font-mono text-sm";
    else lossValue.className = "text-red-400 font-mono text-sm";

    chartReg.data.datasets[1].data = lineData;
    chartReg.update();
}

wSlider.addEventListener('input', updateRegressionLine);
bSlider.addEventListener('input', updateRegressionLine);
updateRegressionLine();


// ==========================================
// VIS 3: CLASSIFICATION (ANTI-SPAM)
// ==========================================
const ctxClass = document.getElementById('classificationChart').getContext('2d');
// Generate random clusters
const spams = Array.from({length: 30}, () => ({x: 60 + Math.random()*40, y: 50 + Math.random()*50}));
const hams = Array.from({length: 30}, () => ({x: 10 + Math.random()*40, y: 10 + Math.random()*40}));

new Chart(ctxClass, {
    type: 'scatter',
    data: {
        datasets: [
            { label: 'Spams (Frauduleux)', data: spams, backgroundColor: '#ef4444', borderColor: '#dc2626', pointRadius: 5 },
            { label: 'Emails Valides', data: hams, backgroundColor: '#22c55e', borderColor: '#16a34a', pointRadius: 5 },
            {
                label: 'Frontière de Décision IA', type: 'line',
                data: [{x: 0, y: 90}, {x: 100, y: 10}],
                borderColor: '#a855f7', borderWidth: 3, fill: false, pointRadius: 0, borderDash: [10, 5]
            }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
            x: { title: { display: true, text: 'Mots Suspects', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 100 },
            y: { title: { display: true, text: 'Points d\'exclamation !!!', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 100 }
        },
        plugins: { legend: { labels: { color: '#e2e8f0' } } }
    }
});


// ==========================================
// VIS 4: CLUSTERING (K-MEANS)
// ==========================================
const ctxCluster = document.getElementById('clusteringChart').getContext('2d');

// Raw uncolored data
const rawData = [
    ...Array.from({length: 15}, () => ({x: 20 + Math.random()*20, y: 20 + Math.random()*20})),
    ...Array.from({length: 15}, () => ({x: 70 + Math.random()*20, y: 30 + Math.random()*20})),
    ...Array.from({length: 15}, () => ({x: 45 + Math.random()*20, y: 70 + Math.random()*20}))
];

let clusteringChart = new Chart(ctxCluster, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Clients (Habitudes)',
            data: rawData, backgroundColor: '#94a3b8', pointRadius: 6
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
            x: { title: { display: true, text: 'Fréquence d\'achat', color: '#94a3b8' }, grid: { color: '#334155' }, min: 0, max: 100 },
            y: { title: { display: true, text: 'Montant moyen', color: '#94a3b8' }, grid: { color: '#334155' }, min: 0, max: 100 }
        },
        plugins: { legend: { labels: { color: '#e2e8f0' } } }
    }
});

let kmeansStep = 0;
document.getElementById('runKmeansBtn').addEventListener('click', () => {
    kmeansStep++;
    if(kmeansStep === 1) {
        // Step 1: Assign colors based on clusters
        clusteringChart.data.datasets = [
            { label: 'Groupe 1 (Économes)', data: rawData.slice(0,15), backgroundColor: '#3b82f6', pointRadius: 6 },
            { label: 'Groupe 2 (Impulsifs)', data: rawData.slice(15,30), backgroundColor: '#ec4899', pointRadius: 6 },
            { label: 'Groupe 3 (Premium)', data: rawData.slice(30,45), backgroundColor: '#eab308', pointRadius: 6 }
        ];
        clusteringChart.update();
        document.getElementById('runKmeansBtn').textContent = 'Relancer';
    } else {
        // Reset
        kmeansStep = 0;
        resetKMeansAnimation();
        document.getElementById('runKmeansBtn').textContent = 'Étape Suivante';
    }
});

function resetKMeansAnimation() {
    clusteringChart.data.datasets = [{
        label: 'Clients (Non triés)',
        data: rawData, backgroundColor: '#94a3b8', pointRadius: 6
    }];
    clusteringChart.update();
}

// ==========================================
// VIS 5: NEURAL NETWORK (CANVAS)
// ==========================================
const nnCanvas = document.getElementById('nnCanvas');
const nnCtx = nnCanvas.getContext('2d');

function drawNNConnections() {
    const container = document.getElementById('nn-container');
    if(!container) return;
    nnCanvas.width = container.clientWidth;
    nnCanvas.height = container.clientHeight;
    
    const inputs = document.querySelectorAll('.input-neuron');
    const hiddens = document.querySelectorAll('.hidden-neuron');
    const outputs = document.querySelectorAll('.output-neuron');

    nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);
    nnCtx.lineWidth = 1;
    nnCtx.strokeStyle = 'rgba(51, 65, 85, 0.5)'; // slate-700

    function drawLines(fromNodes, toNodes) {
        fromNodes.forEach(from => {
            const fRect = from.getBoundingClientRect();
            const cRect = nnCanvas.getBoundingClientRect();
            const x1 = fRect.right - cRect.left;
            const y1 = fRect.top + fRect.height/2 - cRect.top;

            toNodes.forEach(to => {
                const tRect = to.getBoundingClientRect();
                const x2 = tRect.left - cRect.left;
                const y2 = tRect.top + tRect.height/2 - cRect.top;
                
                nnCtx.beginPath();
                nnCtx.moveTo(x1, y1);
                nnCtx.lineTo(x2, y2);
                nnCtx.stroke();
            });
        });
    }

    drawLines(inputs, hiddens);
    drawLines(hiddens, outputs);
}

window.addEventListener('resize', drawNNConnections);

window.fireNetwork = function(inputIndex) {
    const inputs = document.querySelectorAll('.input-neuron');
    const hiddens = document.querySelectorAll('.hidden-neuron');
    const outputs = document.querySelectorAll('.output-neuron');
    
    inputs.forEach(el => el.classList.remove('active-purple', 'active-pink'));
    inputs[inputIndex].classList.add('active-purple');

    setTimeout(() => {
        hiddens.forEach(el => {
            el.classList.add('active-purple');
            el.textContent = (Math.random() * 0.9 + 0.1).toFixed(1);
        });
    }, 400);

    setTimeout(() => {
        hiddens.forEach(el => el.classList.remove('active-purple'));
        outputs.forEach(el => {
            el.classList.add('active-pink');
            el.textContent = (Math.random() > 0.5 ? 'Cat' : 'Dog');
        });
    }, 1000);

    setTimeout(() => {
        outputs.forEach(el => el.classList.remove('active-pink'));
        inputs[inputIndex].classList.remove('active-purple');
        hiddens.forEach(el => el.textContent = '');
        outputs.forEach(el => el.textContent = '?');
    }, 2500);
};

// ==========================================
// VIS 7: NLP WORD EMBEDDINGS
// ==========================================
const ctxNlp = document.getElementById('nlpChart').getContext('2d');
new Chart(ctxNlp, {
    type: 'scatter',
    data: {
        datasets: [
            { label: 'Royauté', data: [{x: 80, y: 80}], backgroundColor: '#f472b6', pointRadius: 8, pointStyle: 'rectRounded' },
            { label: 'Royauté', data: [{x: 85, y: 75}], backgroundColor: '#f472b6', pointRadius: 8, pointStyle: 'rectRounded' },
            { label: 'Animaux', data: [{x: 20, y: 20}], backgroundColor: '#3b82f6', pointRadius: 8, pointStyle: 'circle' },
            { label: 'Animaux', data: [{x: 25, y: 15}], backgroundColor: '#3b82f6', pointRadius: 8, pointStyle: 'circle' },
            { label: 'Tech', data: [{x: 80, y: 20}], backgroundColor: '#22c55e', pointRadius: 8, pointStyle: 'triangle' },
            { label: 'Tech', data: [{x: 85, y: 25}], backgroundColor: '#22c55e', pointRadius: 8, pointStyle: 'triangle' }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const idx = context.dataIndex;
                        const dsIdx = context.datasetIndex;
                        const words = [
                            ['Roi'], ['Reine'], ['Chien'], ['Chat'], ['Ordinateur'], ['Algorithme']
                        ];
                        return words[dsIdx][idx];
                    }
                }
            }
        },
        scales: {
            x: { display: false, min: 0, max: 100 },
            y: { display: false, min: 0, max: 100 }
        },
        animation: {
            onComplete: function() {
                const chartInstance = this;
                const ctx = chartInstance.ctx;
                ctx.font = "bold 14px Inter";
                ctx.fillStyle = "#cbd5e1";
                ctx.textAlign = "center";

                const texts = ["Roi", "Reine", "Chien", "Chat", "Ordinateur", "Algorithme"];
                chartInstance.data.datasets.forEach((dataset, i) => {
                    const meta = chartInstance.getDatasetMeta(i);
                    meta.data.forEach((bar, index) => {
                        ctx.fillText(texts[i], bar.x, bar.y - 15);
                    });
                });
            }
        }
    }
});

// Init
activateVisual(1);
