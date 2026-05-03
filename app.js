// State
let currentStep = 1;
const totalSteps = 3;

// DOM Elements
const stepBtns = document.querySelectorAll('.step-btn');
const chapters = document.querySelectorAll('.chapter');
const visContainers = document.querySelectorAll('.vis-container');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

// Navigation Logic
function goToStep(step) {
    if (step < 1 || step > totalSteps) return;
    currentStep = step;

    // Update UI
    stepBtns.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.step-btn[data-step="${step}"]`).classList.add('active');

    chapters.forEach(c => c.classList.remove('active'));
    document.getElementById(`chap${step}`).classList.add('active');

    visContainers.forEach(v => v.classList.remove('active'));
    document.getElementById(`vis${step}`).classList.add('active');

    // Buttons
    if (step === 1) {
        prevBtn.classList.add('hidden');
    } else {
        prevBtn.classList.remove('hidden');
    }

    if (step === totalSteps) {
        nextBtn.classList.add('hidden');
    } else {
        nextBtn.classList.remove('hidden');
    }

    // Trigger animations specific to step
    if (step === 1) triggerVennAnimation();
    if (step === 3) drawNNConnections();
}

stepBtns.forEach(btn => {
    btn.addEventListener('click', () => goToStep(parseInt(btn.dataset.step)));
});

prevBtn.addEventListener('click', () => goToStep(currentStep - 1));
nextBtn.addEventListener('click', () => goToStep(currentStep + 1));

// ==========================================
// VIS 1: VENN DIAGRAM ANIMATION
// ==========================================
function triggerVennAnimation() {
    const ia = document.getElementById('venn-ia');
    const ml = document.getElementById('venn-ml');
    const dl = document.getElementById('venn-dl');
    
    // Reset
    [ia, ml, dl].forEach(el => {
        el.classList.remove('opacity-100', 'scale-100');
        el.classList.add('opacity-0', 'scale-90');
    });

    // Animate in sequence
    setTimeout(() => { ia.classList.remove('opacity-0', 'scale-90'); ia.classList.add('opacity-100', 'scale-100'); }, 100);
    setTimeout(() => { ml.classList.remove('opacity-0', 'scale-90'); ml.classList.add('opacity-100', 'scale-100'); }, 600);
    setTimeout(() => { dl.classList.remove('opacity-0', 'scale-90'); dl.classList.add('opacity-100', 'scale-100'); }, 1100);
}

// Trigger initially
triggerVennAnimation();

// ==========================================
// VIS 2: LINEAR REGRESSION (CHART.JS)
// ==========================================
const ctx = document.getElementById('regressionChart').getContext('2d');
const wSlider = document.getElementById('wSlider');
const bSlider = document.getElementById('bSlider');
const wValue = document.getElementById('wValue');
const bValue = document.getElementById('bValue');
const lossValue = document.getElementById('lossValue');

// Fake Data (House sizes vs Price)
const dataPoints = [
    {x: 20, y: 50}, {x: 40, y: 90}, {x: 60, y: 120}, 
    {x: 80, y: 160}, {x: 100, y: 210}, {x: 120, y: 230}
];

let currentW = parseFloat(wSlider.value);
let currentB = parseFloat(bSlider.value);

const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Maisons réelles',
            data: dataPoints,
            backgroundColor: '#60a5fa',
            borderColor: '#3b82f6',
            borderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8
        },
        {
            label: 'Modèle IA (y = wx + b)',
            type: 'line',
            data: [],
            borderColor: '#f472b6', // pink-400
            borderWidth: 3,
            fill: false,
            pointRadius: 0,
            borderDash: [5, 5]
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Turn off for real-time slider feeling
        scales: {
            x: { 
                type: 'linear', position: 'bottom', title: { display: true, text: 'Taille (m²)', color: '#94a3b8' },
                grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 140
            },
            y: { 
                title: { display: true, text: 'Prix (k€)', color: '#94a3b8' },
                grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 300
            }
        },
        plugins: {
            legend: { labels: { color: '#e2e8f0' } }
        }
    }
});

function updateRegressionLine() {
    currentW = parseFloat(wSlider.value);
    currentB = parseFloat(bSlider.value);
    
    wValue.textContent = currentW.toFixed(2);
    bValue.textContent = currentB.toFixed(0);

    // Calculate line points
    const lineData = [
        { x: 0, y: currentB },
        { x: 140, y: currentW * 140 + currentB }
    ];

    // Calculate MSE (Loss)
    let mse = 0;
    dataPoints.forEach(pt => {
        const pred = currentW * pt.x + currentB;
        mse += Math.pow(pt.y - pred, 2);
    });
    mse = mse / dataPoints.length;
    
    lossValue.textContent = mse.toFixed(0);
    
    // Color code loss (red if high, green if low)
    if (mse < 500) lossValue.className = "text-green-400 font-mono text-sm";
    else if (mse < 2000) lossValue.className = "text-yellow-400 font-mono text-sm";
    else lossValue.className = "text-red-400 font-mono text-sm";

    chart.data.datasets[1].data = lineData;
    chart.update();
}

wSlider.addEventListener('input', updateRegressionLine);
bSlider.addEventListener('input', updateRegressionLine);
// Initial draw
updateRegressionLine();

// ==========================================
// VIS 3: NEURAL NETWORK ANIMATION
// ==========================================
const nnCanvas = document.getElementById('nnCanvas');
const nnCtx = nnCanvas.getContext('2d');

function drawNNConnections() {
    // Resize canvas to match container
    const container = document.getElementById('nn-container');
    nnCanvas.width = container.clientWidth;
    nnCanvas.height = container.clientHeight;
    
    const inputs = document.querySelectorAll('.input-neuron');
    const hiddens = document.querySelectorAll('.hidden-neuron');
    const outputs = document.querySelectorAll('.output-neuron');

    nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);
    nnCtx.lineWidth = 1;
    nnCtx.strokeStyle = 'rgba(51, 65, 85, 0.5)'; // slate-700

    // Draw lines input -> hidden
    inputs.forEach(inp => {
        const inpRect = inp.getBoundingClientRect();
        const canvasRect = nnCanvas.getBoundingClientRect();
        const x1 = inpRect.right - canvasRect.left;
        const y1 = inpRect.top + inpRect.height/2 - canvasRect.top;

        hiddens.forEach(hid => {
            const hidRect = hid.getBoundingClientRect();
            const x2 = hidRect.left - canvasRect.left;
            const y2 = hidRect.top + hidRect.height/2 - canvasRect.top;
            
            nnCtx.beginPath();
            nnCtx.moveTo(x1, y1);
            nnCtx.lineTo(x2, y2);
            nnCtx.stroke();
        });
    });

    // Draw lines hidden -> output
    hiddens.forEach(hid => {
        const hidRect = hid.getBoundingClientRect();
        const canvasRect = nnCanvas.getBoundingClientRect();
        const x1 = hidRect.right - canvasRect.left;
        const y1 = hidRect.top + hidRect.height/2 - canvasRect.top;

        outputs.forEach(out => {
            const outRect = out.getBoundingClientRect();
            const x2 = outRect.left - canvasRect.left;
            const y2 = outRect.top + outRect.height/2 - canvasRect.top;
            
            nnCtx.beginPath();
            nnCtx.moveTo(x1, y1);
            nnCtx.lineTo(x2, y2);
            nnCtx.stroke();
        });
    });
}

// Handle window resize for canvas
window.addEventListener('resize', () => {
    if (currentStep === 3) drawNNConnections();
});

window.fireNetwork = function(inputIndex) {
    const inputs = document.querySelectorAll('.input-neuron');
    const hiddens = document.querySelectorAll('.hidden-neuron');
    const outputs = document.querySelectorAll('.output-neuron');
    
    // Light up clicked input
    inputs.forEach(el => el.classList.remove('active-purple', 'active-pink'));
    inputs[inputIndex].classList.add('active-purple');

    // Simulate forward propagation delays
    setTimeout(() => {
        hiddens.forEach(el => {
            el.classList.add('active-purple');
            el.textContent = (Math.random() * 0.9 + 0.1).toFixed(1); // Fake activation
        });
    }, 400);

    setTimeout(() => {
        hiddens.forEach(el => el.classList.remove('active-purple'));
        outputs.forEach(el => {
            el.classList.add('active-pink');
            el.textContent = (Math.random() > 0.5 ? 'Cat' : 'Dog'); // Fake classification
        });
    }, 1000);

    setTimeout(() => {
        outputs.forEach(el => el.classList.remove('active-pink'));
        inputs[inputIndex].classList.remove('active-purple');
        hiddens.forEach(el => el.textContent = '');
        outputs.forEach(el => el.textContent = '?');
    }, 2500);
};
