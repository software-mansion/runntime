/** d1 as an auto-moderator: a stream of comments, each one tagged as it
 *  arrives, toxic and spam ones removed. */
import type { Answer, Question } from '../../src/zoo/d1/prompt.ts';
import { loadDecider } from './shared.ts';

const COMMENTS: [string, string][] = [
  ['quiet_gpu', 'Wait, this runs entirely in the browser? No server at all?'],
  ['tensor_tom', 'Tried it on my M2 Air and it loads in like 6 seconds. Impressive.'],
  ['deals4u_2024', 'GET 90% OFF GPU CLOUD CREDITS!!! link in my bio, limited time only'],
  ['meh_dev', 'Cool demo but I still do not get why I would use this over an API call.'],
  [
    'angry_bob',
    'This is the dumbest thing I have seen all week, you clearly have no idea what you are doing.',
  ],
  ['webgpu_fan', 'WebGPU finally getting real use cases. Love to see it.'],
  ['curious_cat', 'How big is the download? Does it cache the weights between visits?'],
  ['lurker_42', 'ok'],
  ['crypto_kev', 'Forget AI, my new token is going 100x this month. DM me to get in early.'],
  [
    'sad_intern',
    'I spent all weekend trying to get onnxruntime-web working and gave up. This gives me hope.',
  ],
  ['nitpicker', 'Calling a 600M model "small" is a stretch but sure.'],
  ['kind_reviewer', 'Really clean writeup, thanks for sharing the code.'],
  ['troll_master', 'lol nobody asked, go back to your basement'],
  ['ml_phd', 'What is the calibration like on out-of-distribution inputs?'],
  ['shop_bot_88', 'Best cheap phone cases here >>> totally-not-spam dot shop <<< free shipping'],
  ['frontend_fred', 'Does it work on Firefox or only Chrome?'],
  ['grumpy_ops', 'Another JS framework for AI. Just what the world needed.'],
  ['happy_hacker', 'Just got it running on my gaming PC, 40 ms per question. This is wild.'],
  ['skeptic_sam', 'I will believe the benchmarks when someone other than the authors runs them.'],
  ['rude_rita', 'Whoever wrote this code should be banned from GitHub, absolute garbage.'],
  ['newbie_nina', 'Sorry if this is a dumb question but what does "zero output tokens" mean?'],
  ['data_dan', 'Using this for triaging support tickets would save my team hours every day.'],
  ['followme_pls', 'Follow my channel for daily AI tips!! Subscribe and hit the bell!!'],
  ['chill_dev', 'Neat. Bookmarked for the weekend.'],
  ['edge_lord', 'Imagine needing a GPU to classify text. Regex has worked for 50 years.'],
  ['privacy_pat', 'Love that nothing leaves the device. That is the real selling point for me.'],
  ['bug_hunter', 'Found a bug: the progress bar jumps back to 0 if you refresh mid-load.'],
  ['toxic_tim', 'Shut up, you are all idiots for upvoting this.'],
  ['question_quinn', 'Can I run my own fine-tuned model with this or only the zoo ones?'],
  ['wholesome_will', 'This made my day, great work to everyone involved!'],
];

const QUESTIONS: Record<string, Question> = {
  toxic: { type: 'noul', instructions: 'Is this comment rude, insulting or toxic?' },
  spam: { type: 'noul', instructions: 'Is this comment spam, an ad or self-promotion?' },
  // As a yes/no this misses most real questions; as a choice it catches them.
  kind: {
    type: 'choice',
    instructions: 'What kind of comment is this?',
    criteria: {
      question: 'The commenter asks something and wants an answer',
      statement: 'The commenter states something, no question',
    },
  },
  mood: {
    type: 'choice',
    instructions: 'What is the tone of this comment?',
    criteria: {
      positive: 'Positive or excited',
      neutral: 'Neutral',
      negative: 'Negative or critical',
    },
  },
};

document.head.insertAdjacentHTML(
  'beforeend',
  `<style>
  :root { color-scheme: dark; --bg: #0b0c10; --panel: #15171d; --card: #181b22; --line: #262a34; --text: #e9ebf0; --dim: #868c9a;
    --accent: #7c8cff; --red: #ff5c5c; --amber: #ffb547; --green: #3ddc84; --blue: #5cb8ff; }
  * { box-sizing: border-box; }
  body { margin: 0; background: radial-gradient(1000px 500px at 80% -10%, #1a1f35, var(--bg)); color: var(--text);
    font: 14px/1.45 Inter, system-ui, sans-serif; min-height: 100vh; padding: 28px 16px; }
  .app { width: min(1100px, 100%); margin: 0 auto; display: grid; grid-template-columns: 1fr 320px; gap: 24px; }
  @media (max-width: 860px) { .app { grid-template-columns: 1fr; } aside { order: -1; } }
  header { grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; }
  h1 { margin: 0; font-size: 22px; font-weight: 650; } h1 span { color: var(--dim); font-weight: 400; }
  header a { color: var(--dim); text-decoration: none; font-size: 13px; }
  .feed { display: flex; flex-direction: column; gap: 10px; min-height: 70vh; }
  .c { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px;
    animation: in .35s ease both; transition: opacity .4s, border-color .3s; }
  @keyframes in { from { opacity: 0; transform: translateY(-10px); } }
  .c .who { display: flex; align-items: center; gap: 8px; color: var(--dim); font-size: 12.5px; }
  .c .avatar { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: 11px; font-weight: 700; color: #0b0c10; }
  .c .who b { color: #c9cdd6; font-weight: 600; }
  .c .ms { margin-left: auto; font-variant-numeric: tabular-nums; }
  .c p { margin: 8px 0 10px; font-size: 15px; }
  .tags { display: flex; flex-wrap: wrap; gap: 6px; min-height: 24px; }
  .tag { font-size: 12px; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--dim); background: #1d2029; }
  .tag.red { color: var(--red); border-color: color-mix(in srgb, var(--red) 40%, transparent); background: color-mix(in srgb, var(--red) 10%, transparent); }
  .tag.amber { color: var(--amber); border-color: color-mix(in srgb, var(--amber) 40%, transparent); background: color-mix(in srgb, var(--amber) 10%, transparent); }
  .tag.blue { color: var(--blue); border-color: color-mix(in srgb, var(--blue) 40%, transparent); background: color-mix(in srgb, var(--blue) 10%, transparent); }
  .tag.green { color: var(--green); border-color: color-mix(in srgb, var(--green) 40%, transparent); background: color-mix(in srgb, var(--green) 10%, transparent); }
  .pending .tags::after { content: 'thinking…'; color: var(--dim); font-size: 12px; animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: .3; } }
  .c.removed { border-color: color-mix(in srgb, var(--red) 35%, var(--line)); }
  .c.removed p { opacity: .35; text-decoration: line-through; text-decoration-color: color-mix(in srgb, var(--red) 60%, transparent); }
  .removed-label { margin-left: 8px; font-size: 11px; font-weight: 700; letter-spacing: .06em; color: var(--red); }
  aside { display: flex; flex-direction: column; gap: 14px; position: sticky; top: 20px; align-self: start; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px; }
  .panel h2 { margin: 0 0 12px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); font-weight: 600; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat b { display: block; font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; line-height: 1.1; }
  .stat span { color: var(--dim); font-size: 12px; }
  .stat.red b { color: var(--red); } .stat.amber b { color: var(--amber); } .stat.blue b { color: var(--blue); }
  .big { font-size: 34px !important; color: var(--accent); }
  textarea { width: 100%; height: 80px; resize: vertical; background: #0f1116; color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 10px; font: inherit; }
  button { margin-top: 8px; width: 100%; padding: 9px; border: 0; border-radius: 10px; background: var(--accent); color: #0b0c10; font-weight: 650; cursor: pointer; }
  button.ghost { background: #232633; color: var(--text); }
  .note { color: var(--dim); font-size: 12px; margin: 0; }
  .loading { padding: 40px; text-align: center; color: var(--dim); }
  .loading i { display: block; width: 240px; height: 4px; margin: 14px auto 0; background: var(--line); border-radius: 2px; overflow: hidden; }
  .loading i::after { content: ''; display: block; height: 100%; width: var(--p, 0%); background: var(--accent); transition: width .2s; }
</style>`,
);

document.body.innerHTML = `
  <div class="app">
    <header>
      <h1>AutoMod in your browser <span>· d1-omni-600M on WebGPU</span></h1>
      <a href="./">← all demos</a>
    </header>
    <div class="feed" id="feed"><div class="loading" id="loading"><span id="loadText">Starting…</span><i id="loadBar"></i></div></div>
    <aside>
      <div class="panel">
        <h2>Live stats</h2>
        <div class="stats">
          <div class="stat"><b id="sCount">0</b><span>comments checked</span></div>
          <div class="stat"><b id="sMs">–</b><span>ms per comment</span></div>
          <div class="stat red"><b id="sRemoved">0</b><span>removed</span></div>
          <div class="stat blue"><b id="sQuestions">0</b><span>need a reply</span></div>
          <div class="stat"><b class="big">0</b><span>tokens generated</span></div>
          <div class="stat amber"><b id="sSpam">0</b><span>spam caught</span></div>
        </div>
      </div>
      <div class="panel">
        <h2>Post a comment</h2>
        <textarea id="mine" placeholder="Try to sneak something past the mod…"></textarea>
        <button id="post">Post</button>
        <button class="ghost" id="pause">Pause stream</button>
      </div>
      <p class="note">4 questions per comment, answered in one pass each by a 600M decision model running on your GPU. No server, no API, nothing leaves the page.</p>
    </aside>
  </div>`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const feed = $('feed');
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const hue = (s: string) => [...s].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) % 360, 7);

function loading(text: string) {
  $('loadText').textContent = text;
  const pct = /(\d+)%/.exec(text)?.[1];
  if (pct) $('loadBar').style.setProperty('--p', `${pct}%`);
}

const stats = { count: 0, ms: 0, removed: 0, questions: 0, spam: 0 };
const yes = (a: Answer | undefined) => (a?.type === 'noul' ? a.noul : 0);

function card(user: string, text: string): HTMLElement {
  const el = document.createElement('div');
  el.className = 'c pending';
  el.innerHTML = `<div class="who"><span class="avatar" style="background:hsl(${hue(user)} 70% 65%)">${esc(user[0]!.toUpperCase())}</span><b>u/${esc(user)}</b><span class="ms"></span></div><p>${esc(text)}</p><div class="tags"></div>`;
  feed.prepend(el);
  while (feed.children.length > 40) feed.lastElementChild!.remove();
  return el;
}

function tag(label: string, p: number, color: string) {
  return `<span class="tag ${color}">${label} ${(p * 100).toFixed(0)}%</span>`;
}

function render(el: HTMLElement, answers: Record<string, Answer>, ms: number) {
  const toxic = yes(answers.toxic);
  const spam = yes(answers.spam);
  const kind = answers.kind;
  const question = kind?.type === 'choice' ? (kind.probabilities.question ?? 0) : 0;
  const mood = answers.mood;
  const tags: string[] = [];
  if (toxic >= 0.5) tags.push(tag('toxic', toxic, 'red'));
  if (spam >= 0.5) tags.push(tag('spam', spam, 'amber'));
  if (question >= 0.5) tags.push(tag('question', question, 'blue'));
  if (mood?.type === 'choice') {
    const color = mood.choice === 'positive' ? 'green' : mood.choice === 'negative' ? 'red' : '';
    tags.push(tag(mood.choice, mood.confidence, color));
  }
  el.classList.remove('pending');
  el.querySelector('.tags')!.innerHTML = tags.join('');
  el.querySelector('.ms')!.textContent = `${ms.toFixed(0)} ms`;
  const removed = toxic >= 0.5 || spam >= 0.5;
  if (removed) {
    el.classList.add('removed');
    el.querySelector('.who')!.insertAdjacentHTML(
      'beforeend',
      '<span class="removed-label">REMOVED</span>',
    );
  }
  stats.count++;
  stats.ms += ms;
  stats.removed += removed ? 1 : 0;
  stats.questions += question >= 0.5 ? 1 : 0;
  stats.spam += spam >= 0.5 ? 1 : 0;
  $('sCount').textContent = String(stats.count);
  $('sMs').textContent = (stats.ms / stats.count).toFixed(0);
  $('sRemoved').textContent = String(stats.removed);
  $('sQuestions').textContent = String(stats.questions);
  $('sSpam').textContent = String(stats.spam);
}

async function main() {
  const decider = await loadDecider(loading);
  $('loading').remove();
  const queue: [string, string][] = [];
  let paused = false;
  let next = 0;
  $('post').onclick = () => {
    const text = $<HTMLTextAreaElement>('mine').value.trim();
    if (!text) return;
    queue.unshift(['you', text]);
    $<HTMLTextAreaElement>('mine').value = '';
  };
  $('pause').onclick = () => {
    paused = !paused;
    $('pause').textContent = paused ? 'Resume stream' : 'Pause stream';
  };
  for (;;) {
    if (queue.length === 0 && !paused) queue.push(COMMENTS[next++ % COMMENTS.length]!);
    const item = queue.shift();
    if (!item) {
      await new Promise((r) => setTimeout(r, 100));
      continue;
    }
    const el = card(...item);
    await new Promise((r) => setTimeout(r, 250)); // let the card land before it gets tagged
    const t = performance.now();
    const { answers } = await decider.systemOne(item[1], QUESTIONS);
    render(el, answers, performance.now() - t);
    await new Promise((r) => setTimeout(r, 650));
  }
}

main().catch((err) => loading(`Error: ${err instanceof Error ? err.message : String(err)}`));
