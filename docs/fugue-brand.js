/* fugue-brand.js — shared between fugue and fugue-evo (keep byte-identical).
 *
 * Injects a sibling-site link into the menu bar so fugue.run and
 * evo.fugue.run point at each other. Which site we're on is detected by
 * hostname; local previews can force it with
 * <html data-fugue-site="fugue"|"evo">. */
(function () {
    'use strict';

    function siblingFor(site) {
        return site === 'evo'
            ? { label: 'Fugue', href: 'https://fugue.run', title: 'Fugue — probabilistic programming for Rust' }
            : { label: 'Evo', href: 'https://evo.fugue.run', title: 'Fugue Evo — evolutionary computation for Rust' };
    }

    function detectSite() {
        var forced = document.documentElement.getAttribute('data-fugue-site');
        if (forced === 'fugue' || forced === 'evo') return forced;
        if (window.location.hostname.indexOf('evo.') === 0) return 'evo';
        // Local preview fallback: the evo book titles itself "Fugue-Evo …".
        if (/evo/i.test(document.title)) return 'evo';
        return 'fugue';
    }

    function inject() {
        var buttons = document.querySelector('#menu-bar .right-buttons');
        if (!buttons || buttons.querySelector('.fugue-brand-link')) return;
        var sib = siblingFor(detectSite());
        var a = document.createElement('a');
        a.className = 'fugue-brand-link';
        a.href = sib.href;
        a.title = sib.title;
        a.setAttribute('aria-label', sib.title);
        var label = document.createElement('span');
        label.textContent = sib.label;
        var arrow = document.createElement('span');
        arrow.className = 'fugue-brand-arrow';
        arrow.textContent = '↗';
        a.appendChild(label);
        a.appendChild(arrow);
        buttons.insertBefore(a, buttons.firstChild);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', inject);
    } else {
        inject();
    }
})();
