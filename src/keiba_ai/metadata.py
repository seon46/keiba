"""Read static race conditions from existing, identified HTML observations."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib


CSV_CONDITION_FIELDS={
    'race_class':'race_class','race_grade':'race_grade','weight_rule':'race_weight_rule',
    'age_min':'race_age_min','age_max':'race_age_max','first_prize_man':'race_first_prize_man',
    'conditions_raw':'race_conditions_raw',
}


class RaceMetadata:
    """Optional enrichment source; no HTTP, original DB writes, or row guesses."""

    def __init__(self,database):
        self.path=Path(database).resolve()
        if not self.path.is_file():
            raise ValueError(f'収集DBがありません: {self.path}')
        self.connection=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        self.connection.row_factory=sqlite3.Row
        self.connection.execute('PRAGMA query_only=ON')
        self.counts=Counter()
        self.warnings=[]
        self._cached=lru_cache(maxsize=128)(self._lookup)

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()

    def close(self):
        self._cached.cache_clear()
        self.connection.close()

    def _warning(self,key,snapshot,identifier,region,reason):
        self.counts[key]+=1
        if len(self.warnings)<20:
            self.warnings.append(dict(snapshot_id=snapshot,race_id=identifier,region=region,reason=reason))

    def _lookup(self,snapshot,identifier,region):
        from keiba_scraper.conditions import extract_conditions,parse_conditions
        self.counts['snapshot_lookups']+=1
        row=self.connection.execute(
            'SELECT r.race_id,r.region,r.page,r.metadata_json,o.raw_path,o.sha256,o.encoding,o.status '
            'FROM races r JOIN observations o ON o.id=r.snapshot_id WHERE r.snapshot_id=?',(snapshot,)).fetchone()
        if row is None:
            self._warning('missing_snapshot',snapshot,identifier,region,'保存済みsnapshotが見つかりません。')
            return {}
        if (row['race_id'],row['region'],row['page'],row['status'])!=(identifier,region,'results',200):
            self._warning('identity_mismatch',snapshot,identifier,region,'snapshotのレース・地域・ページが一致しません。')
            return {}
        try:
            metadata=json.loads(row['metadata_json'])
            if not isinstance(metadata,dict):
                raise ValueError('保存済みレースメタデータがオブジェクトではありません。')
            if metadata.get('conditions_raw'):
                result=parse_conditions(name=metadata.get('race_name',''),title=metadata.get('title',''),
                                        conditions=metadata['conditions_raw'],region=region)
                # Persisted verified grade/class can include evidence from an
                # empty HTML grade icon, which plain text cannot recover.
                for key in CSV_CONDITION_FIELDS:
                    if metadata.get(key) is not None:
                        result[key]=metadata[key]
                self.counts['from_metadata']+=1
            else:
                raw=(self.path.parent/row['raw_path']).resolve()
                if not raw.is_relative_to(self.path.parent/'raw'):
                    raise ValueError('原本の保存先がdata/rawの外です。')
                body=raw.read_bytes()
                if raw.suffix=='.gz': body=gzip.decompress(body)
                if hashlib.sha256(body).hexdigest()!=row['sha256']:
                    raise ValueError('原本のチェックサムが一致しません。')
                result=extract_conditions(body.decode(row['encoding']),region=region)
                self.counts['from_html']+=1
                if not result.get('conditions_raw'):
                    self.counts['header_not_found']+=1
        except (OSError,ValueError,UnicodeError,EOFError,zlib.error,LookupError) as exc:
            self._warning('unusable_raw',snapshot,identifier,region,str(exc))
            return {}
        self.counts['class_known']+=int(result.get('race_class') is not None)
        self.counts['grade_known']+=int(result.get('race_grade') is not None)
        return {CSV_CONDITION_FIELDS[key]:value for key,value in result.items()
                if key in CSV_CONDITION_FIELDS and value is not None and value!=''}

    def __call__(self,row):
        self.counts['input_rows']+=1
        snapshot=row.get('snapshot_id')
        identifier=row.get('race_id')
        region=row.get('region')
        if not snapshot:
            self.counts['rows_without_snapshot']+=1
            return {}
        # Return a copy: the importer may adjust optional fields without
        # changing this cache for later horses in the same source observation.
        return dict(self._cached(snapshot,identifier,region))

    def summary(self):
        info=self._cached.cache_info()
        return dict(source_database=str(self.path),**dict(self.counts),cache_hits=info.hits,
                    warning_examples=list(self.warnings),warning_limit=20)
