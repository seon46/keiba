"""Paths and the small JSON state of the app (current database, model, collection progress)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

JST = timezone(timedelta(hours=9))
DEFAULTS = dict(dataset=None, model_dir=None, collected_through=None, data_start='2021-10-11', train_start=None,
                last_fit=None, retrain_days=7, threads=2, keep_databases=2, keep_models=3)


def now():
    return datetime.now(JST)


def today():
    return now().date().isoformat()


def day_after(value, days=1):
    return (date.fromisoformat(value) + timedelta(days=days)).isoformat()


class App:
    """``data_dir`` is the scraper's data folder (keiba.sqlite3, raw/); the app lives in ``data_dir/app``."""

    def __init__(self, data_dir='data', home=None):
        self.data_dir = Path(data_dir).resolve()
        self.home = Path(home).resolve() if home else self.data_dir / 'app'
        self.home.mkdir(parents=True, exist_ok=True)
        for name in ('db', 'models', 'logs'):
            (self.home / name).mkdir(exist_ok=True)

    @property
    def state_path(self):
        return self.home / 'state.json'

    @property
    def logbook_path(self):
        return self.home / 'app.sqlite3'

    @property
    def scraper_db(self):
        return self.data_dir / 'keiba.sqlite3'

    def read_state(self):
        try:
            stored = json.loads(self.state_path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            stored = {}
        return dict(DEFAULTS, **stored)

    def write_state(self, **changes):
        state = self.read_state()
        state.update(changes)
        handle, name = tempfile.mkstemp(prefix='.state-', dir=self.home)
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
        os.replace(name, self.state_path)
        return state

    def require_ready(self):
        state = self.read_state()
        if not state['dataset'] or not state['model_dir']:
            raise ValueError('DBとモデルが未登録です。keiba_app init を実行してください。')
        return state
