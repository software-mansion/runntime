/** d1 triages a message as you type: editable questions, answered live. */
import type { Question } from '../../src/zoo/d1/prompt.ts';
import { loadDecider, renderAnswer, STYLE } from './shared.ts';

const EXAMPLES = [
  'I was charged twice this month, please refund one of them.',
  'The app crashes every time I open settings on my Pixel 8. Since yesterday.',
  'Someone logged into my account from another country and changed my email!!',
  'Love the new update, the dark mode looks great. Any plans for widgets?',
  'this is the third time im writing, if nobody answers today im cancelling',
];

const QUESTIONS = {
  team: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: {
      billing: 'Charges, refunds, invoices',
      technical: 'App or site faults',
      security: 'Account takeover or suspected unauthorised use',
      product: 'Feedback and feature requests',
    },
  },
  urgency: {
    type: 'score',
    instructions: 'How urgent is this?',
    criteria: ['Can wait', 'Today', 'Blocking the customer now'],
  },
  angry: { type: 'noul', instructions: 'Is the customer angry or frustrated?' },
  churn: { type: 'noul', instructions: 'Is the customer threatening to leave or cancel?' },
  refund: { type: 'noul', instructions: 'Is the customer asking for money back?' },
} satisfies Record<string, Question>;

document.head.insertAdjacentHTML(
  'beforeend',
  `<style>${STYLE}
  main { display: grid; grid-template-columns: minmax(280px, 1fr) minmax(320px, 1fr); gap: 24px; max-width: 1100px; }
  @media (max-width: 760px) { main { grid-template-columns: 1fr; } }
  textarea { width: 100%; box-sizing: border-box; background: #222228; color: #eee; border: 1px solid #444; border-radius: 8px; padding: 10px; font: inherit; }
  #msg { height: 120px; } #qs { height: 320px; font: 12px ui-monospace, monospace; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 16px; }
  .chips button { background: #2a2a33; color: #ccc; border: 1px solid #444; border-radius: 12px; padding: 3px 10px; cursor: pointer; font-size: 12px; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  label { color: #999; font-size: 13px; }
</style>`,
);
document.body.innerHTML = `
  <h1>d1-omni-600M triages your message</h1>
  <p class="sub">One trunk pass per question, zero generated tokens, on your GPU. Edit the questions too. <a href="./">all demos</a></p>
  <main>
    <div>
      <label>Message</label>
      <textarea id="msg"></textarea>
      <div class="chips" id="chips"></div>
      <label>Questions (Decision Index JSON)</label>
      <textarea id="qs" spellcheck="false"></textarea>
    </div>
    <div><div class="status" id="status"></div><div id="answers"></div></div>
  </main>`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const msg = $<HTMLTextAreaElement>('msg');
const qs = $<HTMLTextAreaElement>('qs');
const status = (s: string) => ($('status').textContent = s);
msg.value = EXAMPLES[0]!;
qs.value = JSON.stringify(QUESTIONS, null, 2);
$('chips').innerHTML = EXAMPLES.map((e, i) => `<button data-i="${i}">${e}</button>`).join('');

async function main() {
  const decider = await loadDecider(status);
  let dirty = true;
  const mark = () => (dirty = true);
  msg.oninput = mark;
  qs.oninput = mark;
  $('chips').onclick = (e) => {
    const i = (e.target as HTMLElement).dataset.i;
    if (i === undefined) return;
    msg.value = EXAMPLES[Number(i)]!;
    mark();
  };
  for (;;) {
    if (!dirty) {
      await new Promise((r) => setTimeout(r, 50));
      continue;
    }
    dirty = false;
    let questions: Record<string, Question>;
    try {
      questions = JSON.parse(qs.value);
    } catch {
      status('questions: invalid JSON');
      continue;
    }
    const t = performance.now();
    try {
      const { answers, inputTokens } = await decider.systemOne(msg.value, questions);
      $('answers').innerHTML = Object.entries(answers)
        .map(([n, a]) => {
          const q = questions[n];
          return renderAnswer(n, a, q?.type === 'score' ? q.criteria : undefined);
        })
        .join('');
      status(`${(performance.now() - t).toFixed(0)} ms · ${inputTokens} tokens read · 0 generated`);
    } catch (err) {
      status(`ERROR ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

main().catch((err) => status(`ERROR ${err instanceof Error ? err.message : String(err)}`));
