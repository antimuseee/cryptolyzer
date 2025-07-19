// Matrix rain background effect for body
(function() {
    const canvas = document.createElement('canvas');
    canvas.id = 'matrix-bg-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.pointerEvents = 'none';
    canvas.style.opacity = '0.18'; // Faint
    canvas.style.zIndex = '0';
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    const fontSize = 22;
    const columns = Math.floor(width / fontSize);
    const drops = Array(columns).fill(1);
    const matrixChars = 'アァカサタナハマヤャラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズヅブプエェケセテネヘメレヱゲゼデベペオォコソトノホモヨョロヲゴゾドボポヴABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

    function draw() {
        ctx.fillStyle = 'rgba(26,16,42,0.13)'; // match bg, faded
        ctx.fillRect(0, 0, width, height);
        ctx.font = fontSize + 'px monospace';
        ctx.fillStyle = '#00fff7';
        for (let i = 0; i < drops.length; i++) {
            const text = matrixChars[Math.floor(Math.random() * matrixChars.length)];
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
            if (Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
            if (drops[i] * fontSize > height) {
                drops[i] = 0;
            }
        }
    }

    let anim;
    function animate() {
        draw();
        anim = requestAnimationFrame(animate);
    }
    animate();

    window.addEventListener('resize', () => {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
    });
})();
