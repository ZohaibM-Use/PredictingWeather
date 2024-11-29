function updatePredictionClass(predictionTemp) {
    const predictionBox = document.getElementById('prediction');

    predictionBox.classList.remove('cold', 'warm', 'hot');

    if (predictionTemp < 50) {
        predictionBox.classList.add('cold');  
    } else if (predictionTemp >= 50 && predictionTemp <= 70) {
        predictionBox.classList.add('warm');  
    } else {
        predictionBox.classList.add('hot');  
    }
}

document.addEventListener("DOMContentLoaded", function () {
    const predictionTemp = window.predictionTemp;  
    updatePredictionClass(predictionTemp);  
});
