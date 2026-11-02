'use strict';

const view = document.getElementById('view');
const jobBox = document.getElementById('job');

// ---------------------------------------------------------------- helpers
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pct = (p, digits = 1) => (p == null ? '–' : (p * 100).toFixed(digits) + '%');
const num = (v, digits = 2) => (v == null ? '–' : Number(v).toFixed(digits));
const regionName = (r) => (r === 'central' ? '中央' : '地方');
const store = {
  get(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ } },
};

async function api(path, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const response = await fetch('/api/' + path, options);
  const data = await response.json().catch(() => ({error: '応答を読めません。'}));
  if (!response.ok) throw new Error(data.error || `エラー（${response.status}）`);
  return data;
}

function todayJst() {
  const now = new Date(Date.now() + 9 * 3600 * 1000);
  return now.toISOString().slice(0, 10);
}

function dateLabel(day) {
  const week = '日月火水木金土'[new Date(day + 'T12:00:00Z').getUTCDay()];
  const year = day.slice(0, 4) === todayJst().slice(0, 4) ? '' : `${day.slice(0, 4)}年`;
  return `${year}${Number(day.slice(5, 7))}月${Number(day.slice(8, 10))}日（${week}）`;
}

function timeLabel(iso) {
  if (!iso) return '';
  return `${Number(iso.slice(5, 7))}/${Number(iso.slice(8, 10))} ${iso.slice(11, 16)}`;
}

function surfaceName(surface) {
  return ({'芝': '芝', 'ダ': 'ダート'})[surface] || surface || '';
}

// ---------------------------------------------------------------- jobs
const watched = new Map();

function watch(job, onDone) {
  watched.set(job.id, onDone);
  showJob(job);
  poll();
}

let polling = null;
async function poll() {
  if (polling) return;
  polling = setInterval(async () => {
    if (!watched.size) { clearInterval(polling); polling = null; return; }
    for (const [id, onDone] of [...watched]) {
      try {
        const job = await api('job?id=' + id);
        showJob(job);
        if (job.status === 'done' || job.status === 'error') {
          watched.delete(id);
          if (job.status === 'done') { setTimeout(() => { if (!watched.size) jobBox.hidden = true; }, 2500); }
          onDone && onDone(job);
        }
      } catch (e) { watched.delete(id); }
    }
  }, 1500);
}

function showJob(job) {
  jobBox.hidden = false;
  jobBox.classList.toggle('error', job.status === 'error');
  jobBox.textContent = job.status === 'error' ? `${job.label}：${job.error}` : `${job.label}：${job.message}`;
  jobBox.title = jobBox.textContent;
}

// ---------------------------------------------------------------- router
async function route() {
  const parts = location.hash.replace(/^#\/?/, '').split('/');
  const page = parts[0] || 'day';
  document.querySelectorAll('[data-nav]').forEach((a) => a.classList.toggle('active', a.dataset.nav === (page === 'race' ? 'day' : page)));
  try {
    if (page === 'race') await raceView(parts[1], parts[2]);
    else if (page === 'records') await recordsView();
    else if (page === 'status') await statusView();
    else await dayView(parts[1] || store.get('day', todayJst()));
  } catch (error) {
    view.innerHTML = `<p class="error-text">${esc(error.message)}</p>`;
  }
}
window.addEventListener('hashchange', route);

// ---------------------------------------------------------------- day
async function dayView(day) {
  store.set('day', day);
  const data = await api('day?date=' + day);
  let region = store.get('region', 'central');
  const draw = () => {
    const races = data.races.filter((race) => race.region === region);
    const venues = new Map();
    races.forEach((race) => {
      if (!venues.has(race.course)) venues.set(race.course, []);
      venues.get(race.course).push(race);
    });
    const counts = {central: 0, local: 0};
    data.races.forEach((race) => { counts[race.region] += 1; });
    view.innerHTML = `
      <div class="datebar">
        <span class="date-title">${dateLabel(day)}</span>
        <a href="#/day/${data.previous}"><button type="button">前日</button></a>
        <input type="date" id="pick" value="${day}" aria-label="日付">
        <a href="#/day/${data.next}"><button type="button">翌日</button></a>
        <a href="#/day/${todayJst()}"><button type="button">今日</button></a>
      </div>
      <div class="row">
        <div class="segments" role="group" aria-label="中央・地方">
          <button type="button" data-region="central" aria-pressed="${region === 'central'}">中央 ${counts.central || ''}</button>
          <button type="button" data-region="local" aria-pressed="${region === 'local'}">地方 ${counts.local || ''}</button>
        </div>
        <span class="spacer"></span>
        <button type="button" id="discover">レース一覧を取得</button>
        <button type="button" id="all" class="primary" ${data.races.length ? '' : 'disabled'}>この日の全レースを予想</button>
      </div>
      ${races.length ? `<div class="venues">${[...venues].map(([course, list]) => `
        <section class="venue"><h3>${esc(course)}</h3><ol>${list.map((race) => `
          <li class="${race.supported ? '' : 'unsupported'}"><a href="#/race/${race.region}/${race.race_id}">
            <span class="rno">${race.race_number}R</span>
            <span class="time">${esc(race.start_time || '')}</span>
            <span class="name">${esc(race.race_name || (race.supported ? '' : '対象外'))}</span>
            <span class="mark">${race.predictions ? '予想済' : ''}</span>
          </a></li>`).join('')}</ol></section>`).join('')}</div>`
        : `<p class="empty">${data.races.length ? `この日の${regionName(region)}のレースはありません。` : 'この日のレースはまだ取得していません。「レース一覧を取得」を押すと、netkeibaの日別一覧から取得します。'}</p>`}`;
    view.querySelector('#pick').addEventListener('change', (event) => { location.hash = '#/day/' + event.target.value; });
    view.querySelectorAll('[data-region]').forEach((button) => button.addEventListener('click', () => {
      region = button.dataset.region; store.set('region', region); draw();
    }));
    view.querySelector('#discover').addEventListener('click', async (event) => {
      event.target.disabled = true;
      try { watch(await api('day/discover', {date: day}), () => route()); } catch (e) { alertJob(e); event.target.disabled = false; }
    });
    view.querySelector('#all').addEventListener('click', async (event) => {
      event.target.disabled = true;
      try { watch(await api('day/predict', {date: day}), () => route()); } catch (e) { alertJob(e); event.target.disabled = false; }
    });
  };
  draw();
}

function alertJob(error) {
  showJob({status: 'error', label: '操作', error: error.message});
}

// ---------------------------------------------------------------- race
const raceState = {sort: 'number', open: new Set(), selected: 0};

async function raceView(region, raceId) {
  const data = await api(`race?region=${region}&race_id=${raceId}`);
  const race = data.race;
  if (raceState.raceId !== raceId) { raceState.raceId = raceId; raceState.open = new Set(); raceState.selected = 0; }
  const predictions = data.predictions;
  const prediction = predictions[raceState.selected] || predictions[0] || null;
  const surface = race.race_surface || prediction?.surface || '';
  const course = race.race_course || prediction?.course || '';
  const number = race.race_number || prediction?.race_number || Number(raceId.slice(-2));
  const day = race.race_date || prediction?.race_date || '';
  const facts = [surface && race.race_distance_m ? `${surfaceName(surface)}${race.race_distance_m}m` : '',
    race.race_track_condition ? `馬場 ${race.race_track_condition}` : '',
    race.race_weather ? `天候 ${race.race_weather}` : '',
    race.race_start_time ? `${race.race_start_time}発走` : '',
    race.race_declared_field_size ? `${race.race_declared_field_size}頭` : ''].filter(Boolean).join('　');
  const runners = mergeRunners(data.card, prediction, data.results);
  const yMax = Math.max(0.05, ...runners.flatMap((r) => r.distribution || []));
  const notices = [];
  if (prediction) {
    if (prediction.timing === 'after') notices.push('発走後に作った予想です。記録の集計には含めません。');
    if (prediction.history_through && day && prediction.history_through < shiftDay(day, -1)) {
      notices.push(`この予想に使った成績は${dateLabel(prediction.history_through)}までです。それ以降の成績は入っていません。`);
    }
    if (region === 'local' && !prediction.bodyweight_known) notices.push('馬体重が発表される前の出馬表で予想しています。');
  }
  const sorted = [...runners].sort(raceState.sort === 'win'
    ? (a, b) => (b.win_probability ?? -1) - (a.win_probability ?? -1) || a.horse_number - b.horse_number
    : (a, b) => a.horse_number - b.horse_number);
  const leader = Math.max(...runners.map((r) => r.win_probability ?? 0));
  const hasResults = !!data.results;
  view.innerHTML = `
    <div class="band ${surface === 'ダ' ? 'dirt' : ''}">
      <div class="where"><a href="#/day/${day}">${day ? dateLabel(day) : ''}</a>　${esc(course)} ${number}R　${regionName(region)}</div>
      <h1>${esc(race.race_name || `${course}${number}R`)}</h1>
      <div class="facts">${esc(facts)}</div>
    </div>
    <div class="controls">
      <button type="button" class="primary" id="refresh">最新の出馬表で予想</button>
      <button type="button" id="saved">保存済みの出馬表で予想</button>
      <span class="spacer"></span>
      ${predictions.length ? `<label class="small muted">予想
        <select id="history">${predictions.map((p, i) => `<option value="${i}" ${i === raceState.selected ? 'selected' : ''}>
          ${timeLabel(p.created_at)}${p.timing === 'after' ? '（発走後）' : ''}${p.origin === 'nightly' ? '（自動）' : ''}</option>`).join('')}</select></label>` : ''}
    </div>
    ${notices.map((text) => `<div class="notice">${esc(text)}</div>`).join('')}
    ${prediction ? '' : '<p class="muted">まだ予想していません。「最新の出馬表で予想」を押すと、出馬表を取得して予想します（1〜3分ほどかかります）。</p>'}
    ${runners.length ? `
    <div class="card-wrap"><table class="card">
      <thead><tr>
        <th class="opt">枠</th><th class="num sortable" data-sort="number">馬番</th><th>馬名</th><th class="opt">騎手・厩舎</th><th class="num opt">馬体重</th>
        <th class="num sortable" data-sort="win">勝率</th><th class="num">3着内</th><th class="num opt">期待着順</th>
        <th>着順の確率</th>${hasResults ? '<th>着順</th>' : ''}
      </tr></thead>
      <tbody>${sorted.map((r) => runnerRow(r, yMax, leader, hasResults)).join('')}</tbody>
    </table></div>
    ${prediction ? `<h2>着順ごとの確率</h2><div class="heat-wrap">${heatmap(runners)}</div>
    <p class="small muted">色が濃いほど確率が高い着順です。${hasResults ? '枠で囲んだマスが実際の着順です。' : ''}右端は競走中止など完走しない確率です。</p>` : ''}
    ${prediction?.outcome?.status === 'scored' ? outcomeBox(prediction.outcome) : ''}
    ${prediction ? `<p class="small muted">モデル：${esc(prediction.model_train_end)}より前の成績で学習　作成：${timeLabel(prediction.created_at)}</p>` : ''}`
    : ''}`;
  view.querySelector('#refresh').addEventListener('click', () => predict(region, raceId, true));
  view.querySelector('#saved').addEventListener('click', () => predict(region, raceId, false));
  view.querySelector('#history')?.addEventListener('change', (e) => { raceState.selected = Number(e.target.value); raceView(region, raceId); });
  view.querySelectorAll('[data-sort]').forEach((th) => th.addEventListener('click', () => { raceState.sort = th.dataset.sort; raceView(region, raceId); }));
  view.querySelectorAll('tr.runner').forEach((tr) => tr.addEventListener('click', () => {
    const id = tr.dataset.horse;
    raceState.open.has(id) ? raceState.open.delete(id) : raceState.open.add(id);
    raceView(region, raceId);
  }));
}

async function predict(region, raceId, refresh) {
  view.querySelectorAll('.controls button').forEach((b) => { b.disabled = true; });
  try {
    const job = await api('race/predict', {region, race_id: raceId, refresh});
    watch(job, (done) => {
      if (done.status === 'done') raceState.selected = 0;
      if (location.hash.includes(raceId)) raceView(region, raceId);
    });
  } catch (error) {
    alertJob(error);
    view.querySelectorAll('.controls button').forEach((b) => { b.disabled = false; });
  }
}

function shiftDay(day, days) {
  const d = new Date(day + 'T12:00:00Z');
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function mergeRunners(card, prediction, results) {
  const predicted = new Map((prediction?.runners || []).map((r) => [r.horse_id, r]));
  const base = card.length ? card : (prediction?.runners || []);
  return base.map((row) => {
    const p = predicted.get(row.horse_id) || {};
    const merged = {...p, ...row, distribution: p.distribution || null, win_probability: p.win_probability,
      top3_probability: p.top3_probability, expected_rank: p.expected_rank, finish_probability: p.finish_probability};
    merged.status = row.finish_status === 'scratched' || row.finish_status === 'excluded' || p.status === 'scratched' || p.status === 'excluded' ? 'scratched' : 'starter';
    if (results && results[row.horse_id]) merged.result = results[row.horse_id];
    return merged;
  });
}

function weightText(r) {
  if (!r.weight_kg) return '';
  const change = r.weight_change_kg == null ? '' : `（${r.weight_change_kg > 0 ? '+' : ''}${r.weight_change_kg}）`;
  return `${r.weight_kg}${change}`;
}

function finishText(result) {
  if (!result) return '';
  if (result.finish_status === 'finished') return result.finish_position;
  return ({did_not_finish: '中止', disqualified: '失格', scratched: '取消', excluded: '除外'})[result.finish_status] || '–';
}

function runnerRow(r, yMax, leader, hasResults) {
  const open = raceState.open.has(r.horse_id);
  const sexAge = `${r.sex || ''}${r.age || ''}`;
  const actual = r.result?.finish_status === 'finished' ? r.result.finish_position - 1
    : (r.result && r.result.finish_status !== 'scratched' && r.result.finish_status !== 'excluded' ? (r.distribution ? r.distribution.length - 1 : null) : null);
  const finish = finishText(r.result);
  const row = `
    <tr class="runner ${r.status === 'scratched' ? 'scratched' : ''}" data-horse="${esc(r.horse_id)}" aria-expanded="${open}">
      <td class="opt"><span class="frame f${r.frame || 0}">${r.frame || ''}</span></td>
      <td class="num"><span class="hn f${r.frame || 0}">${r.horse_number}</span></td>
      <td><div class="horse">${esc(r.horse_name || r.horse_id)}</div><div class="sub">${esc(sexAge)}　${r.carried_weight_kg ?? ''}kg<span class="narrow-only">${esc(r.jockey_name || '')}　${esc(weightText(r))}</span></div></td>
      <td class="opt"><div>${esc(r.jockey_name || '')}</div><div class="sub">${esc(r.trainer_name || '')}</div></td>
      <td class="num opt">${esc(weightText(r))}</td>
      <td class="num prob ${r.win_probability && r.win_probability === leader ? 'lead' : ''}">${r.status === 'scratched' ? '取消' : pct(r.win_probability)}</td>
      <td class="num">${pct(r.top3_probability)}</td>
      <td class="num opt">${num(r.expected_rank, 1)}</td>
      <td class="dist">${r.distribution ? miniChart(r.distribution, yMax, actual) : ''}</td>
      ${hasResults ? `<td class="finish ${typeof finish === 'number' && finish <= 3 ? 'top3' : ''}">${esc(finish)}</td>` : ''}
    </tr>`;
  if (!open || !r.distribution) return row;
  return row + `<tr class="detail"><td colspan="${hasResults ? 10 : 9}"><div class="detail-grid">
    ${bigChart(r.distribution, actual)}
    <div class="facts">
      <div>勝率 <b>${pct(r.win_probability)}</b></div>
      <div>3着内 <b>${pct(r.top3_probability)}</b></div>
      <div>完走したときの期待着順 <b>${num(r.expected_rank, 1)}</b></div>
      <div>完走しない確率 <b>${pct(1 - (r.finish_probability ?? 1))}</b></div>
      ${r.result ? `<div>実際の着順 <b>${esc(finish)}</b></div>` : ''}
    </div></div></td></tr>`;
}

function miniChart(dist, yMax, actual) {
  const n = dist.length - 1, w = 6, g = 2, h = 30, gap = 5;
  const width = n * (w + g) + gap + w;
  const bars = dist.map((p, i) => {
    const x = i < n ? i * (w + g) : n * (w + g) + gap;
    const bh = Math.max(p > 0.002 ? 1 : 0, (p / yMax) * (h - 2));
    const hit = i === actual ? `<rect class="actual" x="${x - 1}" y="0.75" width="${w + 2}" height="${h - 1.5}" rx="1"/>` : '';
    return `<rect class="bar ${i === n ? 'nonfinish' : ''}" x="${x}" y="${h - bh}" width="${w}" height="${bh}"/>${hit}`;
  }).join('');
  const label = dist.slice(0, n).map((p, i) => `${i + 1}着${(p * 100).toFixed(0)}%`).join(' ');
  return `<svg class="mini" width="${width}" height="${h}" viewBox="0 0 ${width} ${h}" preserveAspectRatio="none" role="img" aria-label="${esc(label)}">${bars}</svg>`;
}

function bigChart(dist, actual) {
  const n = dist.length - 1, w = 24, g = 6, h = 150, top = 16, bottom = 20, gap = 12;
  const width = n * (w + g) + gap + w + 8;
  const yMax = Math.max(...dist, 0.05);
  const bars = dist.map((p, i) => {
    const x = 4 + (i < n ? i * (w + g) : n * (w + g) + gap);
    const bh = (p / yMax) * (h - top - bottom);
    const y = h - bottom - bh;
    const label = p >= 0.005 ? (p >= 0.1 ? (p * 100).toFixed(0) : (p * 100).toFixed(1)) : '';
    const hit = i === actual ? `<rect class="actual" x="${x - 2}" y="${top - 12}" width="${w + 4}" height="${h - bottom - top + 14}" rx="2"/>` : '';
    return `<rect class="bar ${i === n ? 'nonfinish' : ''}" x="${x}" y="${y}" width="${w}" height="${Math.max(bh, p > 0 ? 1 : 0)}" rx="2"/>
      <text class="value" x="${x + w / 2}" y="${y - 3}" text-anchor="middle">${label}</text>
      <text x="${x + w / 2}" y="${h - 5}" text-anchor="middle" font-size="11">${i < n ? i + 1 : '中止'}</text>${hit}`;
  }).join('');
  return `<svg width="${width}" height="${h}" viewBox="0 0 ${width} ${h}" role="img" aria-label="着順ごとの確率（%）">
    <line class="axis" x1="0" x2="${width}" y1="${h - bottom}" y2="${h - bottom}"/>${bars}</svg>`;
}

function heatmap(runners) {
  const active = runners.filter((r) => r.distribution).sort((a, b) => a.horse_number - b.horse_number);
  if (!active.length) return '';
  const n = active[0].distribution.length - 1;
  const max = Math.max(...active.flatMap((r) => r.distribution));
  const cell = (p, hit) => {
    const alpha = Math.min(1, p / max);
    const dark = alpha > 0.55;
    return `<td class="${hit ? 'hit' : ''}" style="background: color-mix(in srgb, var(--bar) ${(alpha * 100).toFixed(0)}%, var(--surface)); color: ${dark ? 'var(--surface)' : 'var(--ink)'}">${p >= 0.05 ? (p * 100).toFixed(0) : ''}</td>`;
  };
  const head = [...Array(n).keys()].map((i) => `<th>${i + 1}</th>`).join('') + '<th>中止</th>';
  const rows = active.map((r) => {
    const actual = r.result?.finish_status === 'finished' ? r.result.finish_position - 1 : (r.result ? n : -1);
    return `<tr><th class="who"><span class="frame f${r.frame || 0}" style="width:20px;height:20px;font-size:11px">${r.horse_number}</span> ${esc(r.horse_name || '')}</th>
      ${r.distribution.map((p, i) => cell(p, i === actual)).join('')}</tr>`;
  }).join('');
  return `<table class="heat"><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function outcomeBox(o) {
  return `<h2>このレースの採点</h2><div class="figures">
    <div class="figure"><div class="value">${num(o.outcome_log_loss, 3)}</div><div class="label">順位分布の対数損失</div><div class="compare">全馬同じ確率なら ${num(o.uniform_log_loss, 3)}</div></div>
    <div class="figure"><div class="value">${o.pick_position ?? '–'}着</div><div class="label">3着内率1位の馬（${o.pick_horse_number}番）</div></div>
    <div class="figure"><div class="value">${o.favourite_won ? '的中' : '外れ'}</div><div class="label">勝率1位の馬（${o.favourite_horse_number}番）の1着</div></div>
  </div>`;
}

// ---------------------------------------------------------------- records
const recordState = {days: store.get('records-days', '30'), region: '', after: false};

async function recordsView() {
  const query = `records?days=${recordState.days}&region=${recordState.region}&after=${recordState.after ? 1 : 0}`;
  const data = await api(query);
  const s = data.summary;
  const better = s.outcome_log_loss != null && s.uniform_log_loss != null ? s.uniform_log_loss - s.outcome_log_loss : null;
  view.innerHTML = `
    <h1>予想の記録</h1>
    <div class="filters">
      <div class="segments" role="group" aria-label="期間">${[['7', '7日'], ['30', '30日'], ['90', '90日'], ['all', '全期間']].map(([v, l]) =>
        `<button type="button" data-days="${v}" aria-pressed="${recordState.days === v}">${l}</button>`).join('')}</div>
      <select id="region" aria-label="中央・地方">
        <option value="" ${recordState.region === '' ? 'selected' : ''}>中央と地方</option>
        <option value="central" ${recordState.region === 'central' ? 'selected' : ''}>中央</option>
        <option value="local" ${recordState.region === 'local' ? 'selected' : ''}>地方</option>
      </select>
      <label class="small"><input type="checkbox" id="after" ${recordState.after ? 'checked' : ''}> 発走後に作った予想も含める</label>
    </div>
    ${s.races ? `
    <div class="figures">
      <div class="figure"><div class="value">${s.races}</div><div class="label">採点したレース</div><div class="compare">結果待ち ${s.pending_races}レース</div></div>
      <div class="figure"><div class="value">${num(s.outcome_log_loss, 3)}</div><div class="label">順位分布の対数損失（小さいほど良い）</div>
        <div class="compare">全馬同じ確率なら ${num(s.uniform_log_loss, 3)}${better != null ? `（${Math.abs(better).toFixed(3)}${better >= 0 ? '良い' : '悪い'}）` : ''}</div></div>
      <div class="figure"><div class="value">${pct(s.pick_top3_rate)}</div><div class="label">3着内率1位の馬が3着以内</div></div>
      <div class="figure"><div class="value">${pct(s.favourite_win_rate)}</div><div class="label">勝率1位の馬が1着</div></div>
      <div class="figure"><div class="value">${num(s.win_log_loss, 3)}</div><div class="label">1着の対数損失</div></div>
    </div>
    ${data.daily.length >= 2 ? `<div class="chart-box"><div class="legend"><span>予想</span><span class="uniform">全馬同じ確率</span></div>${lineChart(data.daily)}</div>` : ''}
    <h2>レースごと</h2>
    <div class="list-wrap"><table class="list">
      <thead><tr><th>日付</th><th>レース</th><th>3着内率1位の馬</th><th class="num">着順</th><th class="num">対数損失</th><th class="num">全馬同確率</th></tr></thead>
      <tbody>${data.races.map((r) => `<tr>
        <td>${dateLabel(r.race_date)}</td>
        <td><a href="#/race/${r.region}/${r.race_id}">${esc(r.course)} ${r.race_number}R</a> <span class="muted small">${esc(r.race_name || '')}</span></td>
        <td>${r.pick_horse_number ?? ''} ${esc(r.pick_horse_name || '')}</td>
        <td class="num">${r.pick_position ?? '–'}</td>
        <td class="num">${num(r.outcome_log_loss, 3)}</td>
        <td class="num muted">${num(r.uniform_log_loss, 3)}</td></tr>`).join('')}</tbody>
    </table></div>` : `<p class="empty">この条件で採点済みの予想はまだありません。予想したレースは、結果が出たあとの自動処理で採点されます（結果待ち ${s.pending_races}レース）。</p>`}`;
  view.querySelectorAll('[data-days]').forEach((b) => b.addEventListener('click', () => { recordState.days = b.dataset.days; store.set('records-days', b.dataset.days); recordsView(); }));
  view.querySelector('#region').addEventListener('change', (e) => { recordState.region = e.target.value; recordsView(); });
  view.querySelector('#after').addEventListener('change', (e) => { recordState.after = e.target.checked; recordsView(); });
}

function lineChart(daily) {
  const w = Math.max(300, Math.min(1150, view.clientWidth - 58)), h = 200, left = 40, right = 24, top = 10, bottom = 24;
  const values = daily.flatMap((d) => [d.model, d.uniform]).filter((v) => v != null);
  let lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo) * 0.1 || 0.1; lo -= pad; hi += pad;
  const x = (i) => left + (daily.length === 1 ? 0 : (i / (daily.length - 1)) * (w - left - right));
  const y = (v) => top + (1 - (v - lo) / (hi - lo)) * (h - top - bottom);
  const path = (key) => daily.map((d, i) => d[key] == null ? '' : `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(d[key]).toFixed(1)}`).join('');
  const ticks = [lo + pad, (lo + hi) / 2, hi - pad].map((v) => `<line class="axis" x1="${left}" x2="${w - right}" y1="${y(v)}" y2="${y(v)}"/>
    <text x="${left - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11">${v.toFixed(2)}</text>`).join('');
  const labels = [0, Math.floor((daily.length - 1) / 2), daily.length - 1].filter((v, i, a) => a.indexOf(v) === i)
    .map((i) => `<text x="${x(i)}" y="${h - 6}" text-anchor="middle" font-size="11">${Number(daily[i].day.slice(5, 7))}/${Number(daily[i].day.slice(8, 10))}</text>`).join('');
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="日ごとの対数損失">${ticks}${labels}
    <path class="line uniform" d="${path('uniform')}"/><path class="line" d="${path('model')}"/></svg>`;
}

// ---------------------------------------------------------------- status
async function statusView() {
  const [data, log] = await Promise.all([api('status'), api('log')]);
  const st = data.state;
  const stepNames = {collect: '成績の収集', prepare: 'DB更新', score: '照合', fit: '学習し直し', precompute: '当日の予想'};
  const statusNames = {ok: '完了', skipped: '不要', error: '失敗', incomplete: '未完了', running: '実行中', warning: '一部未完了', busy: '重複'};
  view.innerHTML = `
    <h1>状態</h1>
    <h2>データとモデル</h2>
    <dl class="kv">
      <dt>成績の収集</dt><dd>${st.collected_through ? dateLabel(st.collected_through) + 'まで' : '–'}</dd>
      <dt>予想に使うDB</dt><dd>${data.dataset?.last_day ? dateLabel(data.dataset.last_day) + 'までの成績' : '–'}<div class="small muted">${esc(st.dataset || '')}</div></dd>
      <dt>モデル</dt><dd>${data.model ? `${dateLabel(data.model.train_end)}より前の成績で学習（作成 ${timeLabel(data.model.created_at)}）` : '–'}<div class="small muted">${esc(st.model_dir || '')}</div></dd>
      <dt>学習し直し</dt><dd>${st.last_fit ? dateLabel(st.last_fit) : '–'}に実行。${st.retrain_days}日ごとに自動で実行します。</dd>
    </dl>
    <div class="controls">
      <button type="button" class="primary" id="update" ${data.nightly_running ? 'disabled' : ''}>今すぐ成績を追加</button>
      <button type="button" id="refit" ${data.nightly_running ? 'disabled' : ''}>今すぐ学習し直す</button>
      ${data.nightly_running ? '<span class="small muted">更新処理を実行中です。</span>' : ''}
    </div>
    <p class="small muted">成績の追加は、収集・DB更新・照合・当日の予想まで行います（1〜2時間）。学習し直しは、それに加えてモデルを作り直します（さらに30分ほど）。</p>
    <h2>自動処理の履歴</h2>
    ${data.runs.length ? `<div class="list-wrap"><table class="list"><thead><tr><th>開始</th><th>結果</th><th>段階</th></tr></thead><tbody>
      ${data.runs.map((run) => `<tr><td>${timeLabel(run.started_at)}</td><td>${statusNames[run.status] || esc(run.status)}</td>
        <td><div class="steps">${run.steps.map((s) => `<span class="step ${s.status}">${stepNames[s.step] || s.step} ${statusNames[s.status] || s.status}</span>`).join('')}</div></td></tr>`).join('')}
    </tbody></table></div>` : '<p class="muted">まだ自動処理を実行していません。</p>'}
    ${log.lines.length ? `<h2>最新のログ</h2><pre class="log">${esc(log.lines.join('\n'))}</pre>` : ''}`;
  const start = async (body) => {
    try { await api('nightly', body); statusView(); } catch (e) { alertJob(e); }
  };
  view.querySelector('#update').addEventListener('click', () => start({}));
  view.querySelector('#refit').addEventListener('click', () => start({force_fit: true}));
}

route();
