// Smooth scrolling for navigation links and buttons
function scrollToSection(sectionId) {
    const section = document.querySelector(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Navigation link smooth scrolling
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const sectionId = link.getAttribute('href');
        scrollToSection(sectionId);
    });
});

// Hero section button scrolling
function scrollToDemo() {
    scrollToSection('#demo');
}

function scrollToFeatures() {
    scrollToSection('#features');
}

// Stats counter animation
document.addEventListener('DOMContentLoaded', () => {
    const statNumbers = document.querySelectorAll('.stat-number');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const stat = entry.target;
                const target = parseInt(stat.getAttribute('data-target'));
                let count = 0;
                const speed = 200; // Adjust speed of counting animation
                
                const updateCounter = () => {
                    const increment = target / speed;
                    count += increment;
                    if (count < target) {
                        stat.textContent = Math.ceil(count);
                        requestAnimationFrame(updateCounter);
                    } else {
                        stat.textContent = target;
                    }
                };
                
                updateCounter();
                observer.unobserve(stat);
            }
        });
    }, {
        threshold: 0.5
    });
    
    statNumbers.forEach(stat => {
        observer.observe(stat);
    });
});

// Demo form handling
const predictionForm = document.getElementById('predictionForm');
const demoResult = document.getElementById('demo-result');

predictionForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    // Get form values
    const studentName = document.getElementById('student-name').value || 'Étudiant';
    const age = parseInt(document.getElementById('age').value) || 15;
    const sex = document.getElementById('sex').value;
    const averageGrade = parseFloat(document.getElementById('average-grade').value) || 10;
    const absenceRate = parseFloat(document.getElementById('absence-rate').value) || 5;
    
    // Simulate AI prediction (simplified logic for demo)
    let performanceCategory;
    let recommendation;
    
    if (averageGrade >= 15 && absenceRate <= 5) {
        performanceCategory = 'Haut potentiel';
        recommendation = `Félicitations ! ${studentName} montre un excellent potentiel. Continuez à maintenir un haut niveau d'engagement et envisagez des activités d'enrichissement comme des projets avancés ou des compétitions académiques.`;
    } else if (averageGrade >= 10 && absenceRate <= 15) {
        performanceCategory = 'Potentiel moyen';
        recommendation = `${studentName} montre un potentiel solide. Pour améliorer les performances, envisagez un soutien supplémentaire dans les matières clés et une meilleure gestion du temps pour réduire les absences.`;
    } else {
        performanceCategory = 'À risque';
        recommendation = `Attention : ${studentName} pourrait bénéficier d'un soutien ciblé. Nous recommandons un tutorat personnalisé et un plan pour améliorer l'assiduité.`;
    }
    
    // Display result with animation
    demoResult.innerHTML = `
        <div class="result-content">
            <h3>Résultat pour ${studentName}</h3>
            <p><strong>Catégorie de performance :</strong> ${performanceCategory}</p>
            <p><strong>Recommandation :</strong> ${recommendation}</p>
            <div class="result-visual">
                <div class="performance-bar">
                    <div class="bar-fill" style="width: ${averageGrade * 5}%"></div>
                </div>
                <p>Note moyenne : ${averageGrade}/20</p>
            </div>
        </div>
    `;
    
    // Add animation class
    demoResult.classList.add('result-animated');
    
    // Reset form
    predictionForm.reset();
});

// Add CSS for result animation and performance bar (inlined in JS for simplicity)
const style = document.createElement('style');
style.textContent = `
    .result-content {
        text-align: left;
        color: var(--text-primary);
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.5s ease, transform 0.5s ease;
    }
    
    .result-animated .result-content {
        opacity: 1;
        transform: translateY(0);
    }
    
    .performance-bar {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .bar-fill {
        height: 100%;
        background: var(--gradient-1);
        transition: width 1s ease-in-out;
    }
`;
document.head.appendChild(style);