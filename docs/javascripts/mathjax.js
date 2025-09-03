window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '0.8em',
    multlineWidth: '85%',
    useLabelIds: true,
    packages: {
      '[+]': ['ams', 'newcommand', 'noerrors', 'noundefined', 'boldsymbol']
    },
    macros: {
      RR: "\\mathbb{R}",
      CC: "\\mathbb{C}",
      NN: "\\mathbb{N}",
      ZZ: "\\mathbb{Z}",
      EE: "\\mathbb{E}",
      PP: "\\mathbb{P}",
      cA: "\\mathcal{A}",
      cF: "\\mathcal{F}",
      cG: "\\mathcal{G}",
      cH: "\\mathcal{H}",
      cK: "\\mathcal{K}",
      cL: "\\mathcal{L}",
      cO: "\\mathcal{O}",
      cU: "\\mathcal{U}",
      argmin: "\\operatorname{arg\\,min}",
      argmax: "\\operatorname{arg\\,max}",
      rank: "\\operatorname{rank}",
      tr: "\\operatorname{tr}",
      diag: "\\operatorname{diag}",
      sgn: "\\operatorname{sgn}",
      sinc: "\\operatorname{sinc}",
      rect: "\\operatorname{rect}",
      norm: ["\\left\\lVert #1 \\right\\rVert", 1],
      abs: ["\\left\\lvert #1 \\right\\rvert", 1],
      inner: ["\\langle #1, #2 \\rangle", 2],
      E: ["\\mathbb{E}\\left[ #1 \\right]", 1],
      Var: ["\\text{Var}\\left[ #1 \\right]", 1],
      Cov: ["\\text{Cov}\\left[ #1 \\right]", 1]
    }
  },
  loader: {
    load: ['[tex]/ams', '[tex]/newcommand', '[tex]/noerrors', '[tex]/noundefined', '[tex]/boldsymbol']
  },
  svg: {
    fontCache: 'global',
    displayAlign: 'center',
    displayIndent: '0em'
  },
  options: {
    processHtmlClass: 'arithmatex'
  }
};