# keiba

netkeibaの公開データから、中央・地方の過去レース結果を収集してSQLite・CSV・JSONLに保存するPython CLIです。日別一覧から実際のレースIDを発見し、期間内の全レースを順番に取得します。Python 3.11以上が必要です。

## セットアップ

```bash
cd /workspace/keiba
bash scripts/setup.sh
```

依存バージョンは `requirements.txt` に固定しています。Linux・macOSでは以降の例の `.venv/bin/keiba-scraper` を使います。

### Windows / PowerShell

GitとPython 3.11以上をインストールし、PowerShellで実行します。

```powershell
# 初回のみ。すでにclone済みなら、そのkeibaフォルダーでgit pullする
git clone https://github.com/seon46/keiba.git
cd keiba

# Windows PowerShell 5.1 / PowerShell 7向けセットアップ
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# 中央・地方の過去5年間を収集
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper backfill --years 5 --region both
```

仮想環境のActivateは不要です。セットアップのPython自動検出は `py -3`、`python`、`python3` の順です。使用するPythonを明示する場合はセットアップコマンドに `-Python "C:\path with spaces\python.exe"` を追加します。実行ポリシーの指定はセットアップ用プロセスだけに適用されます。

以降のLinux向け例の `.venv/bin/keiba-scraper` は、Windowsでは `.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper` に置き換えます。

```powershell
# 別のPowerShellで進捗確認。同じkeibaフォルダーで実行する
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper status

# 中断後、表示されたjob_idを指定して再開（JOB_IDを実際のIDに置換）
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper backfill --resume JOB_ID

# CSVはPythonから直接UTF-8で出力
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper export --page results --output data\results.csv
```

取得中はPowerShellを開いたままにし、PCのスリープを無効にしてください。Ctrl+Cで停止すると保存済みデータと進捗は残ります。Windowsの省電力設定をこのプログラムが変更することはありません。

## 過去5年間を収集する

```bash
# 日本時間で昨日までの5年間、中央・地方の確定結果を収集
.venv/bin/keiba-scraper backfill --years 5 --region both
```

2026年10月11日に実行すると **2021年10月11日〜2026年10月10日（両端を含む）** が対象です。1826日×2地域の3652件の開催日確認を予定に登録し、一覧に掲載されている各レースを取得します。レースIDを推測して総当たりする処理はありません。

**コードの完成と、5年分の取得完了は別です。** プログラム実行中に取得した分から保存されます。既定で全HTTP要求の間隔を10秒空けるため、全期間の収集は数日〜数週間を見込んでください。仮に10万件のHTTP要求なら、間隔だけで約11.6日必要です。開催日一覧・robots・通信時間・再試行も加わります。レース総数は一覧を集めるまで確定しません。

新規収集は確定結果だけが既定です。レース結果には出走馬ごとの情報が含まれるので、期間内の馬の過去成績を馬IDでまとめられます。これが5年分の基本データセットです。出馬表も収集する場合は `--pages results entries`、馬ページの全戦績を別途収集する場合は `--include-horses` を追加します。馬ページは対象期間外・海外の戦績も含むため、出力時に期間を絞ってください。

### 期間を固定し、中断後に再開する

```bash
# 期間を固定して予定だけを保存。通信なし
.venv/bin/keiba-scraper backfill \
  --start-date 2021-10-11 --end-date 2026-10-10 --region both --plan-only

# 上記条件のジョブを収集・再開（この条件のジョブID）
.venv/bin/keiba-scraper backfill --resume 64bca8f8ad301623

# 進捗と失敗原因を確認。通信なし
.venv/bin/keiba-scraper status --job-id 64bca8f8ad301623
```

最初の出力の `job_id` を控えてください。期間・地域・ページ・馬戦績の条件が同じなら同じジョブになり、完了したタスクは再実行しません。`--years 5` の対象期間は実行日で変わるので、翌日以降は `--resume JOB_ID` を使って元の期間を再開します。`--resume` と期間等の変更は併用できません。

Ctrl+Cで停止すると保存済みの原本・解析結果・収集予定を残し、処理中のタスクを待機状態に戻します。通信待ちの間もタスクの貸出期限を1分ごとに延長し、強制終了時は最後の延長から通常15分で回復します。クラウド環境の停止中は収集も止まります。全期間を取得するには、実行を継続できるマシンと十分なディスク容量を用意してください。

```bash
# 今回は2時間だけ実行して一時停止。その後、同じジョブを再開できる
.venv/bin/keiba-scraper backfill --resume 64bca8f8ad301623 --max-hours 2

# 今回は10タスクだけ処理。タスクには開催日確認も含まれる
.venv/bin/keiba-scraper backfill --resume 64bca8f8ad301623 --max-tasks 10
```

空き容量が既定で1GiB未満になると次のタスクの前に一時停止します。閾値は `--min-free-mb` で指定できます。原本をgzipで圧縮して保存し、CSV出力は全件をメモリに載せず行います。通常は1プロセスで実行してください。同じ保存先を使うプロセスは取得間隔を共有し、タスクの貸出で重複処理を抑えます。異なる保存先での並列収集は取得間隔を共有しません。

### 完了条件と失敗時の復旧

`status` の `complete=true` は、対象日すべての一覧確認と、発見された全レース・要求した馬戦績の保存が完了したことを示します。未取得・実行中・失敗・取得制限が残れば `complete=false` です。`tasks` に種類別の件数、`discovered_races` に発見レース数、`result_runner_rows` に結果の出走行数、`problems` に失敗の先頭20件を表示します。全失敗はSQLiteの `collection_tasks` に保存されます。

一覧の全カードとIDを照合し、結果のレースID・開催日・掲載頭数・一覧の頭数を照合します。取消・除外の行も頭数に含みます。日付やIDが照合できないページ、未確定結果、未知の形式は失敗として残します。開催中止等の未対応ページも推測で成功扱いにしません。

正常な「開催なし」応答は原本と根拠を保存して完了します。中央の空一覧には日付が記載されない形式があるため、その場合は要求URLの開催日を使い、`day_verified=false` と根拠を記録します。該当件数は `empty_days_without_date_metadata` に表示します。サイト自体が一覧からレースを省いた場合まで完全性を保証するものではありません。

接続の失敗・タイムアウト・5xxはHTTP側で最大2回再試行し、それでも失敗したタスクは待機時間を空けて最大3回の実行に制限します。解析失敗は同じ原本の再試行を繰り返さず、失敗として残します。HTTP 403/429・robots拒否があればその場で収集全体を停止します。

```bash
# 原因を修正して、失敗タスクだけを再実行待ちに戻す。通信なし
.venv/bin/keiba-scraper retry --job-id 64bca8f8ad301623

# 未確定ページやHTTP200のエラーページが原本に残っている場合は、再取得を予約
.venv/bin/keiba-scraper --refresh retry --job-id 64bca8f8ad301623

# 予約したタスクを含めて収集再開
.venv/bin/keiba-scraper backfill --resume 64bca8f8ad301623
```

パーサーを修正した場合は通常の `retry` で原本を再解析できます。ページ自体が更新・復旧した場合は `--refresh retry` を使います。元の観測は削除しません。取得制限を受けた場合は条件が解消してから再開してください。`retry` は保存した通信停止期間を解除しません。

### 収集後の診断と取り漏らしの補完

`status` の `problems` は失敗の先頭20件です。日別一覧の失敗が先に並ぶと、レース結果の失敗理由は表示されません。全件を確認するには `diagnose` を使います。通信せず、原因別の件数、全失敗タスクのURL、保存済みHTMLの有無、未公開開催の情報をUTF-8 JSONへ保存します。認証情報やHTML本文は含めません。

```powershell
# Windows: keibaフォルダーで最新版へ更新（収集は停止しておく）
git pull

# 全失敗を診断し、data/recovery_report.jsonへ保存
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper diagnose --job-id 64bca8f8ad301623 --output data\recovery_report.json --overwrite

# 通信エラーだけを再試行待ちに戻す場合
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper retry --job-id 64bca8f8ad301623 --error-type FetchError

# 通常の全失敗retryや、ページ更新時の--refresh retryも引き続き使える
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper backfill --resume 64bca8f8ad301623
```

`--error-type` は複数指定でき、`--kind day/race/horse` でも絞れます。HTTP 503等はサイト復旧後に取得できる可能性がありますが、解析未対応や開催未公開の問題は再取得だけでは解消しません。失敗を再予約する前に診断を保存すると、元の失敗理由を残せます。

日別一覧に一部の競馬場だけ「公開予定」と表示されている場合は、公開済みのレースを収集し、未公開の競馬場を `unavailable_meetings` に記録します。その日の一覧タスクは失敗のまま保持するので、`complete=true` にはなりません。空区画を開催中止・開催なしと推測して埋めません。

以前のバージョンで成功したレースにも、地方の通過順や英数字の騎手・調教師ID等の取り漏らしがあります。更新後、保存済みの成功レース原本を `reparse` で再解析できます。

```powershell
# 収集・他のreparseを停止し、単独で実行する。ネットワーク通信なし
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper reparse --job-id 64bca8f8ad301623
```

原本の取得時刻・snapshot_idは変えず、ID・開催日・頭数を照合してから解析結果を更新します。解析・検証が失敗した場合は元の解析結果を残し、`reparse_error` を表示します。成功した原本はパーサーの内容ごとに記録するため、Ctrl+C等で止めても同じコマンドで残りから再開できます。保存直後に強制終了して記録が残らなかった原本は再解析しますが、同じsnapshotを更新するため重複行は作りません。`--max-tasks`で今回の件数を制限できます。

`reparse_summary.complete` は成功レース原本の再解析が終わったかを示します。収集全体の `status.complete`、未取得レースの再試行とは別に確認します。`reparse` は未取得ページをダウンロードしません。収集と再解析を同時に動かさないでください。開始時の実行中タスク確認は排他ロックではありません。

地方の別表から補完したコーナー通過順では、並走グループはグループ先頭の順位を共有します。元の原文・並走グループ・corner番号も `race_corner_passages`、`corner_positions`、`corner_group_ids` 等に保持します。既存の馬ごとの通過順は上書きせず、未知の馬番・破損した表・不足したcornerは推測しません。問題がある別表は `race_parse_warnings` に残します。

## データセットを出力する

```bash
# 5年分の確定結果をCSVに出力（取得済み分だけ）
.venv/bin/keiba-scraper export --page results \
  --start-date 2021-10-11 --end-date 2026-10-10 --output data/results_5years.csv

# 中央だけ、JSONL
.venv/bin/keiba-scraper export --page results --region central \
  --start-date 2021-10-11 --end-date 2026-10-10 \
  --format jsonl --output data/central_5years.jsonl

# 別途取得した馬ページの戦績
.venv/bin/keiba-scraper export --dataset horse-runs \
  --start-date 2021-10-11 --end-date 2026-10-10 --output data/horse_runs_5years.csv
```

出力はUTF-8で、1行が1頭の出走です。通常は地域・レース・ページごとの最新観測だけを出力します。全取得履歴は `--all-snapshots`、既存ファイルを置き換える場合は `--overwrite` を指定します。収集中にも出力でき、CSVの列と行は同じDBの状態から読みます。書き込み途中のファイルで既存出力を置き換えません。

IDは文字列、欠損は `null`（CSVでは空欄）です。取消等は順位を `null`、状態を `scratched` / `excluded` などで保存します。ばんえいは `race_type=banei` とし、英字付きID・直線200m・水分量などを保持します。ばんえいと平地競馬は特徴量・モデルで区別してください。

### 保存内容

- `data/raw/`: HTML原本を `.html.gz` で保存。SHA-256は展開前の圧縮データではなく、取得した元bytesで計算・照合します。既存の `.html` 原本も引き続き読めます。
- `data/keiba.sqlite3`: 原文URL、実際の取得時刻、文字コード、チェックサム、解析結果、収集キュー。
- `observations`: 取得履歴。`races` / `runners`: レース・出走馬。`horses` / `horse_runs`: 別途取得した馬の戦績。
- `collection_jobs` / `collection_tasks` / `collection_races`: 期間、進捗、失敗原因、発見したレースと期待頭数。

データは `.gitignore` でGitの対象外です。バックアップする場合は収集を停止して `data/` 全体をコピーしてください。DBだけではHTML原本が含まれません。動作中のDBファイルだけを単純コピーせず、SQLiteのbackup API等を使ってください。

## 単発取得とキャッシュ

```bash
.venv/bin/keiba-scraper fetch --region central --race-id 202405021211 --page results
.venv/bin/keiba-scraper fetch --region local --race-id 202642062411 --page results
.venv/bin/keiba-scraper fetch --region central --race-id 202605040411 --page entries
.venv/bin/keiba-scraper list --region central --date 2024-05-26
.venv/bin/keiba-scraper horse --horse-id 2019102383
```

国内レースIDはURLの `race_id=` の後の12桁、通常の馬IDは10桁、ばんえいの馬IDは `B` と9桁です。地域は明示し、IDの桁から推測しません。複数対象は `--race-id` / `--horse-id` を繰り返し指定します。馬戦績の海外レースIDは英字も文字列のまま保存します。

一度取得したURLは保存済みHTMLを再利用し、取得時刻を変更しません。共通オプションはサブコマンドより前に指定します。

```bash
.venv/bin/keiba-scraper --refresh fetch --region central --race-id 202405021211 --page results
.venv/bin/keiba-scraper --offline fetch --region central --race-id 202405021211 --page results
.venv/bin/keiba-scraper --data-dir data --interval 15 backfill --resume 64bca8f8ad301623
```

当日の情報更新は明示的に再取得してください。`backfill` の完了タスクは `--refresh` でも実行対象に戻りません。更新するレースは `--refresh fetch` を使います。

新規HTTP要求はrobots.txtと再試行も含めて既定10秒以上の間隔を空けます。robotsの規則は24時間で再確認し、拒否・取得条件を確認できない応答・ページのリダイレクトで停止します。403/429はその取得先への通信を最低1時間停止し、長い `Retry-After` があればその期間を使います。TLS検証は有効です。

アクセス間隔は許可された取得頻度を保証するものではありません。[公式の案内](https://support.keiba.netkeiba.com/hc/ja/articles/39720493823129)と[利用規約](https://www.netkeiba.com/info/kiyaku.html)の用途・負荷の条件を確認してください。

## 学習・予想へ使うとき

`retrieved_at` は実際の取得時刻（UTC）、掲載のレース日・発走時刻は日本時間です。情報の公開時刻を推定しません。過去ページを今取得しても、当時の発走前の情報を再現できるとは限りません。

結果の着順・時計・確定オッズを、そのレースの予想入力に使わないでください。過去戦績は予想対象より前の出走に絞って特徴量にし、学習・評価も時系列で分割します。予想時刻の出馬表・オッズは、その時点で `--refresh fetch` して観測を蓄積してください。

この段階で生成するのは公開のレース・出走データセットです。調教、会員向け指数、払戻金の構造化、ログイン、予測モデルの学習は未実装です。

## 検証

```bash
.venv/bin/python -m unittest discover -s tests -v
```

中断・再開、強制終了後のタスク回復、取得制限、収集漏れの完了判定、日付・ID・頭数不一致、開催なし、5年の期間境界、圧縮原本と旧DBの互換、同時書き込み中のCSV出力、出力失敗時のファイル保護をネットワークなしで検証します。中央・地方・ばんえいの2021年の実ページでも解析・ID・日付・頭数を確認しています。
