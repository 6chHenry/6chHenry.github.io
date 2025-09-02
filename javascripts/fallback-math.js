// 备用数学公式渲染方案
document.addEventListener("DOMContentLoaded", function() {
    // 检查 KaTeX 是否已成功渲染
    setTimeout(function() {
        const katexElements = document.querySelectorAll('.katex, .katex-display');
        const mathElements = document.querySelectorAll('script[type="math/tex"], .arithmatex');
        
        // 如果 KaTeX 未渲染或者有未渲染的数学元素，则使用 MathJax
        if (katexElements.length === 0 || mathElements.length > 0) {
            // 动态加载 MathJax
            const mathJaxScript = document.createElement('script');
            mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
            mathJaxScript.async = true;
            
            mathJaxScript.onload = function() {
                // 配置 MathJax
                window.MathJax = {
                    tex: {
                        inlineMath: [["\\(", "\\)"]],
                        displayMath: [["\\[", "\\]"], ["$$", "$$"]],
                        processEscapes: true,
                        processEnvironments: true,
                        tags: "ams",
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
                
                // 重新加载 MathJax
                MathJax.typesetPromise();
            };
            
            document.head.appendChild(mathJaxScript);
        }
    }, 1000); // 等待 1 秒检查 KaTeX 是否成功渲染
});