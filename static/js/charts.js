function createCharts(plotData) {
    plotData.forEach((coin, index) => {
        const trace = {
            x: coin.prices.map(p => new Date(p.date).toISOString().split('T')[0]),
            y: coin.prices.map(p => p.price),
            type: 'scatter',
            mode: 'lines',
            name: coin.name
        };

        const layout = {
            title: coin.name + ' Price Trend',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price (USD)' },
            margin: { t: 40 }
        };

        Plotly.newPlot('chart_' + (index + 1), [trace], layout);
    });
}

// Initialize charts when data is available
document.addEventListener('DOMContentLoaded', function() {
    const plotData = {{ plot_data|tojson }};
    createCharts(plotData);
});
