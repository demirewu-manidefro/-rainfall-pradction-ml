/* script.js */

// Navigation Scroll Effect
window.addEventListener('scroll', function () {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Mobile Menu Toggle
function toggleMenu() {
    const menu = document.getElementById('mobileMenu');
    menu.classList.toggle('active');

    const hamburger = document.querySelector('.hamburger');
    hamburger.classList.toggle('active');
}

async function handlePrediction(event) {
    event.preventDefault();

    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const data = {};

    // Convert FormData to JSON
    formData.forEach((value, key) => {
        data[key] = value;
    });

    // Show Loading
    const btn = document.getElementById('predictBtn');
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');

    btn.style.display = 'none';
    loading.style.display = 'block';
    resultSection.style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Update UI
        document.getElementById('predValue').innerText = result.prediction;
        document.getElementById('predNote').innerText = result.note;

        // Render Charts
        renderCharts(result.feature_importance);

        // Show Results
        loading.style.display = 'none';
        resultSection.style.display = 'block';
        btn.style.display = 'inline-block';

        // Smooth scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while predicting. Please try again.');
        loading.style.display = 'none';
        btn.style.display = 'inline-block';
    }
}

let importanceChartInstance = null;
let confidenceChartInstance = null;

function renderCharts(featureImportance) {
    const ctxImp = document.getElementById('importanceChart').getContext('2d');
    const ctxConf = document.getElementById('confidenceChart').getContext('2d');

    // Destory existing charts if any
    if (importanceChartInstance) importanceChartInstance.destroy();
    if (confidenceChartInstance) confidenceChartInstance.destroy();

    // Feature Importance Chart (Bar)
    const labels = Object.keys(featureImportance);
    const values = Object.values(featureImportance);

    importanceChartInstance = new Chart(ctxImp, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: values,
                // Using new palette color
                backgroundColor: 'rgba(6, 214, 160, 0.7)',
                borderColor: '#06d6a0',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#a8dadc' } },
                x: { grid: { display: false }, ticks: { color: '#a8dadc' } }
            }
        }
    });

    // Confidence Chart (Doughnut)
    confidenceChartInstance = new Chart(ctxConf, {
        type: 'doughnut',
        data: {
            labels: ['Confidence', 'Uncertainty'],
            datasets: [{
                data: [85, 15],
                backgroundColor: ['#06d6a0', '#073b4c'],
                borderWidth: 0,
                cutout: '75%'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#a8dadc' } }
            }
        }
    });
}
