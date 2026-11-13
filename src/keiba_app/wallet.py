"""Users' bankrolls: who uses the app, their deposits and withdrawals, the bets they bought and how those paid.

Kept in app.sqlite3 next to the prediction log.  Every change of money is one ledger row (deposit, withdraw,
bet, payout, refund, adjust), so a balance is the sum of a user's ledger and its history is the running sum.
A bet is recorded when the user says they bought it (from a proposal or by hand) and settled from the race's
payout table once results are saved: a ticket pays ``stake / 100 * payout``; a ticket that includes a
取消 / 除外 horse is refunded (枠連 only when every horse of the frame is out).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .state import now, today

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    login TEXT PRIMARY KEY, name TEXT NOT NULL, admin INTEGER NOT NULL DEFAULT 0,
    kelly_fraction REAL NOT NULL DEFAULT 0.5, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT, login TEXT NOT NULL REFERENCES users(login) ON DELETE CASCADE,
    at TEXT NOT NULL, kind TEXT NOT NULL, amount INTEGER NOT NULL, bet_id INTEGER, note TEXT
);
CREATE INDEX IF NOT EXISTS ledger_user ON ledger(login, at, id);
CREATE TABLE IF NOT EXISTS user_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, login TEXT NOT NULL REFERENCES users(login) ON DELETE CASCADE,
    created_at TEXT NOT NULL, prediction_id INTEGER, region TEXT NOT NULL, race_id TEXT NOT NULL,
    race_date TEXT NOT NULL, course TEXT, race_number INTEGER, race_name TEXT, label TEXT,
    tickets_json TEXT NOT NULL, cost INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'open',
    payout INTEGER, refund INTEGER, settled_at TEXT, detail_json TEXT
);
CREATE INDEX IF NOT EXISTS user_bets_user ON user_bets(login, race_date);
CREATE INDEX IF NOT EXISTS user_bets_open ON user_bets(status, race_date);
"""

BET_KINDS = ('win', 'place', 'bracket', 'quinella', 'wide', 'exacta', 'trio', 'trifecta')
MONEY_KINDS = dict(deposit='入金', withdraw='出金', bet='購入', payout='払戻', refund='返還', adjust='調整')
MAX_YEN = 100_000_000


def _yen(value, name='金額'):
    try:
        amount = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}は整数（円）で指定してください。') from exc
    if amount <= 0 or amount > MAX_YEN:
        raise ValueError(f'{name}は1円以上{MAX_YEN:,}円以下にしてください。')
    return amount


def normalize_ticket(ticket):
    """{bet, key, stake} with the key in the app's canonical form ('5', '3-7', '7>3')."""
    from keiba_ai.bets import combo_key
    bet = ticket.get('bet')
    if bet not in BET_KINDS:
        raise ValueError(f'券種が不正です: {bet}')
    raw = str(ticket.get('key', '')).strip()
    ordered = bet in ('exacta', 'trifecta')
    parts = [part for part in raw.replace('→', '>').replace('-', ' ').replace('>', ' ').split() if part]
    size = 1 if bet in ('win', 'place') else 3 if bet in ('trio', 'trifecta') else 2
    if len(parts) != size or not all(part.isdigit() and 0 < int(part) <= 18 for part in parts):
        raise ValueError(f'買い目が不正です: {raw}')
    if bet != 'bracket' and len(set(parts)) != size:
        raise ValueError(f'同じ馬番が重複しています: {raw}')
    if ordered and '-' in raw:
        raise ValueError(f'馬単・3連単は着順を「>」でつないでください: {raw}')
    stake = int(ticket.get('stake') or 0)
    if stake < 100 or stake % 100 or stake > MAX_YEN:
        raise ValueError('各買い目の金額は100円単位で指定してください。')
    return dict(bet=bet, key=combo_key(bet, parts), stake=stake)


def settle_tickets(tickets, payouts, runners):
    """(payout, refund, per-ticket results) or None while the payout table is not available.

    ``runners``: results rows (horse_number, frame, finish_status).  Bet types missing from the table (not sold,
    e.g. 枠連 with 8 or fewer runners) are refunded."""
    if not payouts:
        return None
    out = {str(row.get('horse_number')) for row in runners if row.get('finish_status') in ('scratched', 'excluded')}
    frames = {}
    for row in runners:
        frames.setdefault(str(row.get('frame')), []).append(row.get('finish_status') in ('scratched', 'excluded'))
    payout = refund = 0
    results = []
    for ticket in tickets:
        numbers = ticket['key'].replace('>', '-').split('-')
        if ticket['bet'] == 'bracket':
            void = any(all(frames.get(number) or [True]) for number in numbers)
        else:
            void = any(number in out for number in numbers)
        table = payouts.get(ticket['bet'])
        if void or not table:
            refund += ticket['stake']
            results.append(dict(ticket, result='refund', amount=ticket['stake']))
            continue
        paid = table.get(ticket['key'])
        amount = ticket['stake'] // 100 * paid if paid else 0
        payout += amount
        results.append(dict(ticket, result='hit' if paid else 'miss', amount=amount))
    return payout, refund, results


class Wallet:
    def __init__(self, path):
        self.path = Path(path)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA journal_mode=WAL')
            with connection:
                yield connection
        finally:
            connection.close()

    def change_token(self):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT (SELECT COALESCE(MAX(id),0)||':'||COUNT(*)||':'||COALESCE(SUM(amount),0) FROM ledger),"
                "(SELECT COUNT(*)||':'||COALESCE(GROUP_CONCAT(name||admin||kelly_fraction),'') FROM users),"
                "(SELECT COUNT(*) FROM user_bets WHERE status='open')").fetchone()
        return '|'.join(str(value) for value in row)

    # -- users ---------------------------------------------------------
    def users(self):
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT u.*,COALESCE((SELECT SUM(amount) FROM ledger l WHERE l.login=u.login),0) AS balance,'
                "(SELECT COUNT(*) FROM user_bets b WHERE b.login=u.login AND b.status='open') AS open_bets "
                'FROM users u ORDER BY u.created_at, u.login').fetchall()
        return [dict(row) for row in rows]

    def user(self, login):
        if not login:
            return None
        return next((user for user in self.users() if user['login'] == login), None)

    def add_user(self, login, name, *, deposit=0, admin=False):
        login = str(login or '').strip()
        name = str(name or '').strip() or login
        if not login or len(login) > 200 or len(name) > 100:
            raise ValueError('ログイン名（Tailscaleのメールアドレスなど）を指定してください。')
        at = now().isoformat(timespec='seconds')
        with self._connect() as connection:
            if connection.execute('SELECT 1 FROM users WHERE login=?', (login,)).fetchone():
                raise ValueError(f'{login} はすでに登録されています。')
            connection.execute('INSERT INTO users (login,name,admin,created_at) VALUES (?,?,?,?)',
                               (login, name, int(bool(admin)), at))
            if deposit:
                connection.execute("INSERT INTO ledger (login,at,kind,amount,note) VALUES (?,?,'deposit',?,?)",
                                   (login, at, _yen(deposit, '初期資金'), '初期資金'))
        return self.user(login)

    def update_user(self, login, *, name=None, admin=None, kelly_fraction=None):
        changes, values = [], []
        if name is not None:
            name = str(name).strip()
            if not name or len(name) > 100:
                raise ValueError('表示名が不正です。')
            changes.append('name=?')
            values.append(name)
        if admin is not None:
            changes.append('admin=?')
            values.append(int(bool(admin)))
        if kelly_fraction is not None:
            fraction = float(kelly_fraction)
            if not 0 < fraction <= 1:
                raise ValueError('ケリー係数は0より大きく1以下にしてください。')
            changes.append('kelly_fraction=?')
            values.append(fraction)
        if changes:
            with self._connect() as connection:
                if connection.execute(f'UPDATE users SET {",".join(changes)} WHERE login=?',
                                      values + [login]).rowcount == 0:
                    raise KeyError('ユーザーがいません。')
        return self.user(login)

    def remove_user(self, login):
        with self._connect() as connection:
            if connection.execute('DELETE FROM users WHERE login=?', (login,)).rowcount == 0:
                raise KeyError('ユーザーがいません。')

    def balance(self, login):
        with self._connect() as connection:
            return int(connection.execute('SELECT COALESCE(SUM(amount),0) FROM ledger WHERE login=?',
                                          (login,)).fetchone()[0])

    # -- money -----------------------------------------------------------
    def move(self, login, kind, amount, note=None):
        """Deposit, withdraw or adjust (adjust may be negative)."""
        if kind not in ('deposit', 'withdraw', 'adjust'):
            raise ValueError('種類が不正です。')
        if kind == 'adjust':
            value = int(amount)
            if value == 0 or abs(value) > MAX_YEN:
                raise ValueError('調整額が不正です。')
        else:
            value = _yen(amount) * (-1 if kind == 'withdraw' else 1)
        if self.user(login) is None:
            raise KeyError('ユーザーがいません。')
        with self._connect() as connection:
            connection.execute('INSERT INTO ledger (login,at,kind,amount,note) VALUES (?,?,?,?,?)',
                               (login, now().isoformat(timespec='seconds'), kind, value, (note or '')[:200] or None))
        return self.balance(login)

    def delete_entry(self, login, entry_id):
        """Remove a deposit / withdraw / adjust row entered by mistake (bet rows go with their bet)."""
        with self._connect() as connection:
            row = connection.execute('SELECT kind FROM ledger WHERE id=? AND login=?', (entry_id, login)).fetchone()
            if row is None:
                raise KeyError('記録がありません。')
            if row['kind'] not in ('deposit', 'withdraw', 'adjust'):
                raise ValueError('馬券の記録は馬券の取り消しから消してください。')
            connection.execute('DELETE FROM ledger WHERE id=?', (entry_id,))

    # -- bets ------------------------------------------------------------
    def record_bet(self, login, race, tickets, *, label=None, prediction_id=None):
        """Record bought tickets of one race and take their cost from the balance."""
        if self.user(login) is None:
            raise KeyError('ユーザーがいません。')
        tickets = [normalize_ticket(ticket) for ticket in tickets or []]
        if not tickets:
            raise ValueError('買い目がありません。')
        if len(tickets) > 200:
            raise ValueError('買い目が多すぎます。')
        merged = {}
        for ticket in tickets:  # the same ticket twice is one bigger stake
            key = (ticket['bet'], ticket['key'])
            merged[key] = dict(ticket, stake=merged[key]['stake'] + ticket['stake']) if key in merged else ticket
        tickets = list(merged.values())
        cost = sum(ticket['stake'] for ticket in tickets)
        at = now().isoformat(timespec='seconds')
        with self._connect() as connection:
            identifier = connection.execute(
                'INSERT INTO user_bets (login,created_at,prediction_id,region,race_id,race_date,course,race_number,'
                'race_name,label,tickets_json,cost) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (login, at, prediction_id, race['region'], race['race_id'], race['race_date'], race.get('course'),
                 race.get('race_number'), race.get('race_name'), (label or '')[:100] or None,
                 json.dumps(tickets, ensure_ascii=False), cost)).lastrowid
            connection.execute("INSERT INTO ledger (login,at,kind,amount,bet_id) VALUES (?,?,'bet',?,?)",
                               (login, at, -cost, identifier))
        return self.bet(identifier)

    def bet(self, identifier):
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM user_bets WHERE id=?', (identifier,)).fetchone()
        return None if row is None else _bet(row)

    def cancel_bet(self, login, identifier):
        """Delete a bet recorded by mistake (only while unsettled) and give its cost back."""
        with self._connect() as connection:
            row = connection.execute('SELECT status FROM user_bets WHERE id=? AND login=?', (identifier, login)).fetchone()
            if row is None:
                raise KeyError('馬券の記録がありません。')
            if row['status'] != 'open':
                raise ValueError('精算済みの馬券は取り消せません（「返還にする」を使ってください）。')
            connection.execute('DELETE FROM ledger WHERE bet_id=?', (identifier,))
            connection.execute('DELETE FROM user_bets WHERE id=?', (identifier,))

    def void_bet(self, login, identifier, note='返還（手動）'):
        """Refund a whole bet (e.g. the race was cancelled); undoes an earlier settlement first."""
        at = now().isoformat(timespec='seconds')
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM user_bets WHERE id=? AND login=?', (identifier, login)).fetchone()
            if row is None:
                raise KeyError('馬券の記録がありません。')
            connection.execute("DELETE FROM ledger WHERE bet_id=? AND kind IN ('payout','refund')", (identifier,))
            connection.execute("INSERT INTO ledger (login,at,kind,amount,bet_id,note) VALUES (?,?,'refund',?,?,?)",
                               (login, at, row['cost'], identifier, note))
            connection.execute("UPDATE user_bets SET status='void',payout=0,refund=?,settled_at=?,detail_json=NULL "
                               'WHERE id=?', (row['cost'], at, identifier))

    def bets(self, login, *, limit=300):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM user_bets WHERE login=? ORDER BY status!='open', race_date DESC, id DESC LIMIT ?",
                (login, limit)).fetchall()
        return [_bet(row) for row in rows]

    def open_bets(self, through=None):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM user_bets WHERE status='open' AND race_date<=? ORDER BY id",
                                      (through or today(),)).fetchall()
        return [_bet(row) for row in rows]

    def settle(self, saved, *, through=None):
        """Settle open bets of races up to ``through`` whose payouts are saved; returns how many were settled."""
        settled = 0
        cache = {}
        for bet in self.open_bets(through):
            key = (bet['region'], bet['race_id'])
            if key not in cache:
                payouts, _ = saved.payouts(*key)
                cache[key] = (payouts, saved.snapshot(*key, 'results') or [])
            payouts, runners = cache[key]
            outcome = settle_tickets(bet['tickets'], payouts, runners)
            if outcome is None:
                continue
            payout, refund, details = outcome
            at = now().isoformat(timespec='seconds')
            with self._connect() as connection:
                if connection.execute("UPDATE user_bets SET status='settled',payout=?,refund=?,settled_at=?,detail_json=? "
                                      "WHERE id=? AND status='open'",
                                      (payout, refund, at, json.dumps(details, ensure_ascii=False), bet['id'])).rowcount:
                    for kind, amount in (('payout', payout), ('refund', refund)):
                        if amount:
                            connection.execute('INSERT INTO ledger (login,at,kind,amount,bet_id) VALUES (?,?,?,?,?)',
                                               (bet['login'], at, kind, amount, bet['id']))
                    settled += 1
        return settled

    # -- history ---------------------------------------------------------
    def history(self, login):
        """The ledger oldest first with the running balance, a daily balance series, and summary figures."""
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(
                'SELECT l.*,b.race_date,b.course,b.race_number,b.race_name,b.label FROM ledger l '
                'LEFT JOIN user_bets b ON b.id=l.bet_id WHERE l.login=? ORDER BY l.at, l.id', (login,))]
            bets = [dict(row) for row in connection.execute(
                "SELECT status,cost,payout,refund,race_date FROM user_bets WHERE login=?", (login,))]
        balance = 0
        daily = {}
        for row in rows:
            balance += row['amount']
            row['balance'] = balance
            row['kind_name'] = MONEY_KINDS.get(row['kind'], row['kind'])
            day = row['at'][:10]  # the day the money moved (a payout: the day it was settled)
            daily[day] = daily.get(day, 0) + row['amount']
        series, running = [], 0
        for day in sorted(daily):
            running += daily[day]
            series.append(dict(day=day, balance=running, change=daily[day]))
        settled = [bet for bet in bets if bet['status'] == 'settled']
        staked = sum(bet['cost'] - (bet['refund'] or 0) for bet in settled)
        returned = sum(bet['payout'] or 0 for bet in settled)
        deposits = sum(row['amount'] for row in rows if row['kind'] in ('deposit', 'withdraw', 'adjust'))
        open_cost = sum(bet['cost'] for bet in bets if bet['status'] == 'open')
        summary = dict(balance=balance, net_deposit=deposits, profit=balance + open_cost - deposits, open_cost=open_cost,
                       settled_bets=len(settled), staked=staked, returned=returned,
                       roi=returned / staked if staked else None,
                       hit_bets=sum(1 for bet in settled if (bet['payout'] or 0) > 0))
        return dict(entries=list(reversed(rows)), series=series, summary=summary)


def _bet(row):
    item = dict(row)
    item['tickets'] = json.loads(item.pop('tickets_json') or '[]')
    item['details'] = json.loads(item.pop('detail_json') or 'null')
    return item
