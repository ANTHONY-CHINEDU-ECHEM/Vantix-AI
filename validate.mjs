// Parse and render every Mermaid diagram in a Markdown file, in both the default and dark
// themes, with HTML labels enabled exactly as GitHub renders them. Fails when a diagram does
// not parse or render, when any node or subgraph shape lacks an inline fill:none, or when an
// edge label background box is present.
//
// Usage: node validate.mjs PATH_TO_README
import fs from 'node:fs';
import { JSDOM } from 'jsdom';

const path = process.argv[2] ?? '../../README.md';
const markdown = fs.readFileSync(path, 'utf8');
const blocks = [...markdown.matchAll(/```mermaid\n([\s\S]*?)```/g)].map((m) => m[1]);
if (blocks.length === 0) {
  console.error('No Mermaid diagrams found');
  process.exit(1);
}

const dom = new JSDOM('<!doctype html><html><body></body></html>', { pretendToBeVisual: true });
const w = dom.window;
Object.assign(globalThis, {
  window: w,
  document: w.document,
  Element: w.Element,
  HTMLElement: w.HTMLElement,
  SVGElement: w.SVGElement,
  CSSStyleSheet: w.CSSStyleSheet ?? class { replaceSync() {} },
});
// jsdom has no layout engine; approximate text metrics so Mermaid can lay diagrams out.
w.SVGElement.prototype.getBBox = function getBBox() {
  const length = (this.textContent ?? '').length;
  return { x: 0, y: 0, width: 8 * length + 10, height: 20 };
};
w.SVGElement.prototype.getComputedTextLength = function getComputedTextLength() {
  return 8 * (this.textContent ?? '').length;
};
globalThis.DOMPurify = (await import('dompurify')).default(w);
const mermaid = (await import('mermaid')).default;

let failures = 0;
for (const theme of ['default', 'dark']) {
  mermaid.initialize({ startOnLoad: false, theme, securityLevel: 'strict' });
  for (const [index, source] of blocks.entries()) {
    const label = `diagram ${index + 1} (${theme} theme)`;
    try {
      await mermaid.parse(source);
      const { svg } = await mermaid.render(`v${theme}${index}`, source);
      const nodes = [...svg.matchAll(/<g class="node[^"]*"[^>]*>([\s\S]*?)<g class="label/g)];
      const unfilledNodes = nodes.filter((m) => !m[1].replace(/\s/g, '').includes('fill:none'));
      const clusters = [...svg.matchAll(/<g class="cluster[^"]*"[^>]*>\s*<rect([^>]*)>/g)];
      const unfilledClusters = clusters.filter((m) => !m[1].replace(/\s/g, '').includes('fill:none'));
      const labelBoxes = (svg.match(/class="labelBkg"/g) ?? []).length;
      const problems = [];
      if (unfilledNodes.length) problems.push(`${unfilledNodes.length} node(s) with a background`);
      if (unfilledClusters.length) problems.push(`${unfilledClusters.length} subgraph(s) with a background`);
      if (labelBoxes) problems.push(`${labelBoxes} edge label background box(es)`);
      if (problems.length) {
        failures += 1;
        console.error(`FAIL ${label}: ${problems.join(', ')}`);
      } else {
        console.log(`ok   ${label}: ${nodes.length} nodes and ${clusters.length} subgraphs, no backgrounds`);
      }
    } catch (error) {
      failures += 1;
      console.error(`FAIL ${label}: ${String(error?.message ?? error).slice(0, 400)}`);
    }
  }
}
process.exit(failures ? 1 : 0);
