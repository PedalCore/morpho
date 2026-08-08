// Interactive circle-of-fifths wheel, overlaid on the network view.
// Shows the current key (gold), pulses when a modulator moves it (with the
// rule that fired), and lets the performer click any key to set it manually.

import { KEY_NAMES } from '../sim/lab.js';

const NS = 'http://www.w3.org/2000/svg';

export class FifthsWheel {
  constructor(parent, onSelect) {
    this.onSelect = onSelect;
    this.current = 0;
    const size = 150;
    const c = size / 2;
    const R = 56;

    this.root = document.createElement('div');
    this.root.className = 'fifths';
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    this.svg.setAttribute('width', size);
    this.svg.setAttribute('height', size);

    this.pulse = document.createElementNS(NS, 'circle');
    this.pulse.setAttribute('cx', c);
    this.pulse.setAttribute('cy', c);
    this.pulse.setAttribute('r', 20);
    this.pulse.setAttribute('class', 'pulsering');
    this.svg.appendChild(this.pulse);

    this.nodes = KEY_NAMES.map((name, i) => {
      const a = (i / 12) * Math.PI * 2 - Math.PI / 2;
      const x = c + Math.cos(a) * R;
      const y = c + Math.sin(a) * R;
      const g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'node');
      const circle = document.createElementNS(NS, 'circle');
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', 10.5);
      const label = document.createElementNS(NS, 'text');
      label.setAttribute('x', x);
      label.setAttribute('y', y + 3);
      label.textContent = name;
      g.appendChild(circle);
      g.appendChild(label);
      g.addEventListener('click', () => this.onSelect && this.onSelect(i));
      this.svg.appendChild(g);
      return g;
    });

    this.center = document.createElementNS(NS, 'text');
    this.center.setAttribute('x', c);
    this.center.setAttribute('y', c + 6);
    this.center.setAttribute('class', 'centerkey');
    this.svg.appendChild(this.center);

    this.rule = document.createElementNS(NS, 'text');
    this.rule.setAttribute('x', c);
    this.rule.setAttribute('y', c + 22);
    this.rule.setAttribute('class', 'rulelabel');
    this.svg.appendChild(this.rule);

    this.root.appendChild(this.svg);
    parent.appendChild(this.root);
    this.setKey(0);
  }

  setKey(i, ruleText = '') {
    this.current = i;
    this.nodes.forEach((g, k) => g.setAttribute('class', k === i ? 'node active' : 'node'));
    this.center.textContent = KEY_NAMES[i];
    this.rule.textContent = ruleText;
    if (ruleText) {
      this.rule.setAttribute('class', 'rulelabel show');
      clearTimeout(this._ruleTimer);
      this._ruleTimer = setTimeout(() => this.rule.setAttribute('class', 'rulelabel'), 2600);
      // restart the pulse animation
      this.pulse.setAttribute('class', 'pulsering');
      void this.pulse.getBoundingClientRect();
      this.pulse.setAttribute('class', 'pulsering animate');
    }
  }
}
