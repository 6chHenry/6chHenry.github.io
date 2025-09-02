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
        '[+]': ['ams', 'newcommand', 'configmacros']
      },
      autoload: {
        color: [],
        colorV2: ['color']
      },
      formatError: (jax, err) => {
        console.error('MathJax error:', err);
        jax.formatError(err);
      }
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex",
      renderActions: {
        addMenu: [0, '', '']
      }
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
    },
    loader: {
      load: ['[tex]/ams', '[tex]/newcommand', '[tex]/configmacros', '[tex]/noundefined', '[tex]/colorV2']
    }
  };
  
  document$.subscribe(() => { 
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })