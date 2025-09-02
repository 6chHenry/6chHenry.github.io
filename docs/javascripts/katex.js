document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
        delimiters: [
            {left: "$$", right: "$$", display: true},
            {left: "\\[", right: "\\]", display: true},
            {left: "$", right: "$", display: false},
            {left: "\\(", right: "\\)", display: false}
        ],
        throwOnError: false,
        macros: {
            "\\R": "\\mathbb{R}",
            "\\N": "\\mathbb{N}",
            "\\Z": "\\mathbb{Z}",
            "\\Q": "\\mathbb{Q}",
            "\\C": "\\mathbb{C}",
            "\\P": "\\mathbb{P}",
            "\\E": "\\mathbb{E}",
            "\\Var": "\\mathrm{Var}",
            "\\Cov": "\\mathrm{Cov}",
            "\\argmax": "\\operatorname{argmax}",
            "\\argmin": "\\operatorname{argmin}"
        }
    });
    
    // 处理多行公式的显示
    const katexElements = document.querySelectorAll('.katex-display');
    katexElements.forEach(element => {
        element.style.overflowX = 'auto';
        element.style.overflowY = 'hidden';
        element.style.margin = '1em 0';
    });
});