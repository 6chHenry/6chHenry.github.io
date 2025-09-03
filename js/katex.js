document.addEventListener("DOMContentLoaded", function() {
    // 首先尝试使用 KaTeX 渲染
    try {
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
                "\\argmin": "\\operatorname{argmin}",
                // 添加 align 环境的映射
                "\\begin{align}": "\\begin{aligned}",
                "\\end{align}": "\\end{aligned}",
                "\\begin{align*}": "\\begin{aligned}",
                "\\end{align*}": "\\end{aligned}"
            }
        });
        
        // 处理多行公式的显示
        const katexElements = document.querySelectorAll('.katex-display');
        katexElements.forEach(element => {
            element.style.overflowX = 'auto';
            element.style.overflowY = 'hidden';
            element.style.margin = '1em 0';
        });
        
        // 检查是否有未渲染的数学公式
        setTimeout(() => {
            const unrenderedMath = document.querySelectorAll('script[type="math/tex"], .arithmatex');
            if (unrenderedMath.length > 0) {
                // 如果有未渲染的数学公式，加载 MathJax
                loadMathJax();
            }
        }, 1000);
    } catch (error) {
        console.error("KaTeX rendering failed:", error);
        // 如果 KaTeX 失败，加载 MathJax
        loadMathJax();
    }
});

// 加载 MathJax 的函数
function loadMathJax() {
    // 检查是否已经加载了 MathJax
    if (window.MathJax) {
        return;
    }
    
    // 创建 MathJax 配置
    window.MathJax = {
        tex: {
            inlineMath: [["\\(", "\\)"]],
            displayMath: [["\\[", "\\]"], ["$$", "$$"]],
            processEscapes: true,
            processEnvironments: true,
            tags: "ams",
            macros: {
                R: "\\mathbb{R}",
                N: "\\mathbb{N}",
                Z: "\\mathbb{Z}",
                Q: "\\mathbb{Q}",
                C: "\\mathbb{C}",
                P: "\\mathbb{P}",
                E: "\\mathbb{E}",
                Var: "\\mathrm{Var}",
                Cov: "\\mathrm{Cov}",
                argmax: "\\operatorname{argmax}",
                argmin: "\\operatorname{argmin}"
            },
            packages: {
                '[+]': ['ams', 'newcommand', 'configmacros', 'amscd', 'amsthm', 'color', 'autoload']
            },
            autoload: {
                color: [],
                colorV2: ['color']
            }
        },
        options: {
            ignoreHtmlClass: ".*|",
            processHtmlClass: "arithmatex"
        },
        startup: {
            ready: () => {
                MathJax.startup.defaultReady();
                MathJax.startup.promise.then(() => {
                    // 处理多行公式
                    const mathElements = document.querySelectorAll('.MathJax, .MathJax_Display');
                    mathElements.forEach(element => {
                        if (element.textContent.includes('\\\\')) {
                            element.style.overflowX = 'auto';
                            element.style.overflowY = 'hidden';
                            element.style.whiteSpace = 'nowrap';
                        }
                    });
                    
                    // 处理公式换行问题
                    const displayMathElements = document.querySelectorAll('.MathJax_Display');
                    displayMathElements.forEach(element => {
                        element.style.margin = '1em 0';
                    });
                });
            }
        }
    };
    
    // 动态加载 MathJax
    const mathJaxScript = document.createElement('script');
    mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    mathJaxScript.async = true;
    document.head.appendChild(mathJaxScript);
}