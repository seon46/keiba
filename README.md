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

`status` の `complete=true` は、対象日すべての一覧確認と、発見された全レース・要求した馬戦績の保存が完了したことを示します。未取得・実行中・失敗・取得制限が残れば `complete=false` です。`tasks` に種類別の件数、`discovered_races` に発見レース数、`result_runner_rows` に結果の出走行数、`cancelled_races` に明示的な開催中止の件数、`problems` に失敗の先頭20件を表示します。全失敗はSQLiteの `collection_tasks` に保存されます。

一覧の全カードとIDを照合し、結果のレースID・開催日・掲載頭数・一覧の頭数を照合します。取消・除外の行も頭数に含みます。開催日はtitle末尾の開催情報から読み、レース名に含まれる別の日付は使いません。日付やIDが照合できないページ、未確定結果、未知の形式は失敗として残します。

空の結果表に加え、その結果欄に「取り止めとなりました」等の明示的な告知があり、原本のID・開催日も一致した場合だけ、開催中止として保存・完了します。`race_status=cancelled` と告知原文・根拠をレースに残し、出走馬の行は作りません。空表だけでは中止と推測せず失敗のままです。中止レースは1頭1行のCSVには出ず、SQLiteの `races.metadata_json` と `status.cancelled_races` で確認できます。

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

更新済みコードで今回の診断にある失敗を復旧する場合、まず `--refresh retry --job-id 64bca8f8ad301623 --error-type ValueError`、続けて通常の `retry --job-id 64bca8f8ad301623`、最後に `backfill --resume 64bca8f8ad301623` を実行します。これで一覧・開催日エラーは再取得を予約し、0行結果は保存済み原本で中止の証拠を再解析し、HTTP失敗分は取得し直します。再予約済みの再取得フラグは通常のretryで消えません。取得完了済みタスクは再実行しません。

GitHubの名前解決に失敗するPCには、別途渡すGit bundleを `keiba-recovery.bundle` としてkeibaフォルダーに保存してから、`git fetch .\keiba-recovery.bundle HEAD` と `git merge --ff-only FETCH_HEAD` でコードを更新できます。両コマンドが成功したことを確認してから復旧を実行してください。この更新にGitHubへの接続は不要で、`data/` と `.venv/` は保持します。ローカル変更などで更新が止まった場合は強制リセットせず、表示されたエラーを確認してください。

日別一覧に一部の競馬場だけ「公開予定」と表示されている場合は、公開済みのレースを収集し、未公開の競馬場を `unavailable_meetings` に記録します。その日の一覧タスクは失敗のまま保持するので、`complete=true` にはなりません。空区画を開催中止・開催なしと推測して埋めません。

以前のバージョンで成功したレースにも、地方の通過順や英数字の騎手・調教師ID等の取り漏らしがあります。更新後、保存済みの成功レース原本を `reparse` で再解析できます。

```powershell
# 収集・他のreparseを停止し、単独で実行する。ネットワーク通信なし
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper reparse --job-id 64bca8f8ad301623
```

原本の取得時刻・snapshot_idは変えず、ID・開催日・頭数を照合してから解析結果を更新します。解析・検証が失敗した場合は元の解析結果を残し、`reparse_error` を表示します。成功した原本はパーサーの内容ごとに記録するため、Ctrl+C等で止めても同じコマンドで残りから再開できます。保存直後に強制終了して記録が残らなかった原本は再解析しますが、同じsnapshotを更新するため重複行は作りません。`--max-tasks`で今回の件数を制限できます。

`reparse_summary.complete` は成功レース原本の再解析が終わったかを示します。収集全体の `status.complete`、未取得レースの再試行とは別に確認します。`reparse` は未取得ページをダウンロードしません。収集と再解析を同時に動かさないでください。開始時の実行中タスク確認は排他ロックではありません。

Windows向けに別途配布する補完専用ZIPには、Git bundleと `reparse_saved_data.py` / `reparse_saved_data.cmd` を同梱します。ZIPをすべて展開してから `reparse_saved_data.cmd` を実行すると、`%USERPROFILE%\keiba` をローカルbundleからfast-forward更新し、既存の仮想環境で上記ジョブの成功済み原本だけを再解析します。Downloads等に展開したまま実行でき、bundleの移動は不要です。Gitの更新に失敗すれば補完を開始しません。`retry` / `backfill` やサイトへの再取得は実行しません。

この起動用ファイルは進捗を `data/reparse_console.log`、最終集計を `data/reparse_report.json` に保存します。空き容量不足などで一時停止した場合も完了とは表示しません。同じファイルで残りから再開できます。補完で更新するのはSQLiteの解析結果です。既存CSVを使う場合は、完了後にexportで出力し直してください。

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
- `pedigrees`: 馬ごとの5代血統。原本の観測ID・取得時刻と、父系・母系の経路ごとの祖先を保持。
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

収集するのは公開のレース・出走データセットです。調教、会員向け指数、払戻金の構造化、ログインは未実装です。以下の初版モデルは公開の出走データを使います。

## 保存済みレースに登場する馬の5代血統を取得する

`pedigree` は、公開血統ページから父・母・母父と5代分の祖先62枠を取得します。血統の取得・保存専用で、予測モデルの入力や学習方法は変更しません。追加ライブラリの導入は不要です。

既存の5年間の収集ジョブ `64bca8f8ad301623` を指定すると、そのジョブに属する保存済みレース結果から馬IDを重複除去して対象を決めます。中央・地方をまとめて扱い、同じ馬をレースごとに取得しません。結果の取消・除外欄に掲載されている馬も対象です。祖先の情報は同じ1ページ内から読み、祖先ページを再帰的に収集しません。元のレース収集ジョブは保持して、`ped-` から始まる別の血統ジョブを作ります。

PowerShellでコードを更新し、対象頭数と予定だけ確認します。この段階ではサイトへアクセスしません。

```powershell
& {
    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath (Join-Path $env:USERPROFILE "keiba")
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Code update failed." }
    .\.venv\Scripts\python.exe -X utf8 -m keiba_scraper pedigree --from-job-id 64bca8f8ad301623 --plan-only
    if ($LASTEXITCODE -ne 0) { throw "Pedigree planning failed." }
}
```

表示の `total_horses` が対象頭数、`completed_horses` が取得済み、`remaining_horses` が残りです。取得を開始するコマンドは次です。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper pedigree --from-job-id 64bca8f8ad301623
```

レース結果を追加していなければ、**中断後もこの同じコマンドで残りから再開できます**。完了済みタスクは再取得しません。既存キャッシュがあれば原本を再利用します。Ctrl+Cでは進捗を保存し、PCの再起動などによる強制終了後は、処理中だった1件の貸出期限が切れるまで通常最大15分待ってから復帰します。取得中はPowerShellを開き、PCをスリープさせないでください。初回は1万頭につき既定10秒の間隔だけで約28時間かかり、通信・再試行の時間も加わります。

今回だけ10頭まで、または2時間まで実行する場合は、それぞれ `--max-tasks 10` / `--max-hours 2` を付けます。特定の1頭なら `pedigree --horse-id 2019102383` を使えます。`--horse-id` は複数回指定できます。

進捗を別のPowerShellから確認する場合は、全ジョブを表示する次のコマンドを使います。血統ジョブは `config.type="pedigree"` と `ped-` で始まる `job_id` で識別できます。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper status
```

対象一覧は計画時点で固定します。後からレース結果が増えると、同じ `--from-job-id` で新しく計画した際は対象の異なる別の血統ジョブになり、保存済みHTMLは共有します。元の対象一覧を厳密に再開する場合は、血統側のIDを使う `pedigree --resume ped-...` を指定してください。レース側のIDを `--resume` に使うことはできません。

403/429・robots拒否では収集全体を停止し、進捗に `blocked` を残します。タイムアウト・5xx等は既存の制限付き再試行を使います。血統表がない、別馬のページ、壊れた表などは失敗として保持します。失敗は `diagnose --job-id ped-... --output data\pedigree_report.json` で確認でき、原因解消後は `retry --job-id ped-... --kind pedigree` で待機へ戻せます。保存済みページ自体がエラーの場合は `--refresh retry` で再取得を予約します。ここでの `ped-...` は表示された血統ジョブIDへ置き換えます。

保存先は既存の `data\keiba.sqlite3` の `pedigrees` テーブルと、`data\raw\` のHTML原本です。祖先名が不明な枠はnull、名前だけ判明してIDがない祖先は名前を保存します。世代を推測して埋めず、同じ祖先が父系・母系の複数箇所に現れる場合も経路ごとに残します。**血統ジョブの `complete=true` は全対象ページの保存完了を示し、62枠すべて判明したこととは別です。** 不明な祖先名がある頭数は `partial_pedigrees`、各馬の不明箇所は `missing_paths` / `missing_id_paths` に記録します。

取得済み分をCSVまたはJSONLへ出力できます。出力だけなら通信しません。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper export --dataset pedigrees --output data\pedigrees.csv
.\.venv\Scripts\python.exe -X utf8 -m keiba_scraper export --dataset pedigrees --format jsonl --output data\pedigrees.jsonl
```

出力は1頭1行で、`horse_id`、馬名、`sire_id/name`、`dam_id/name`、`damsire_id/name`、既知祖先数、原本のURL・snapshot_id・取得時刻などを含みます。`ancestors` に62枠の祖先をJSONで保持し、`path` のSが父、Dが母、DSが母父、SSが父父です。各祖先には名前・ID・参照URL・明示生年・原文を保存します。既存の出力を置き換える場合だけ `--overwrite` を追加してください。血統は馬単位なのでレースの日付・地域・ページによる出力フィルターはありません。

## 3着以内の確率を学習する

補完後のCSVから、各馬の3着以内の確率を推定します。中央平地・中央障害・地方平地・ばんえいは別モデルです。モデルの精度は自分のデータで出した `evaluation.json` を確認してください。サンプルで処理が動いたことは、本番での予測精度の証明にはなりません。

Windowsでは、収集・再解析を終えてから次を実行します。CPUで動作し、GPUは不要です。依存ライブラリの導入後、学習と予測はネットワーク通信を使いません。

```powershell
Set-Location -LiteralPath (Join-Path $env:USERPROFILE "keiba")
& {
    $ErrorActionPreference = "Stop"
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Code update failed." }
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-ml.ps1
    if ($LASTEXITCODE -ne 0) { throw "ML setup failed." }
    .\.venv\Scripts\python.exe -X utf8 -m keiba_ai fit --input data\results_5years_reparsed.csv --output-dir data\ml_top3
}
```

Linuxでは `.venv/bin/python -m pip install -r requirements-ml.txt`、`.venv/bin/python -m pip install --no-deps --no-build-isolation -e .` の後、`.venv/bin/python -m keiba_ai fit --input data/results_5years_reparsed.csv --output-dir data/ml_top3` を実行します。

`fit` はCSVの検査、履歴特徴量の作成、学習・評価を順に行います。既存のCSV・収集DBは変更しません。約100万行では時間と追加ディスク容量が必要です。PCをスリープさせず、処理中の表示を確認してください。既存の出力先には上書きしません。

- `data/ml_top3/quality_report.json`: 読み込んだ行数、欠損、除外したレースと理由、期間・区分別の件数。
- `data/ml_top3/features.sqlite3`: 検査済みの元行と、過去日だけから作った学習入力。
- `data/ml_top3/models/evaluation.json`: 区分ごとの2025年検証・2026年最終評価、基準予測との比較、確率の校正表。
- `data/ml_top3/models/*.joblib`: 学習済みモデル。自分で学習して保存したファイルだけを読み込んでください。

CSVの検査だけから進める場合は `prepare --input ... --output-dir data/ml_top3` を使えます。prepare完了後に学習が中断・失敗した場合は、次で学習から再実行できます。学習途中の木からの再開は行わず、学習を最初からやり直します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai train --dataset data\ml_top3\features.sqlite3 --output-dir data\ml_top3\models
```

既定では2024年末までで候補を学習し、2025年の成績で設定を選びます。3着以内モデルの `--feature-set auto` は、初版34項目の `baseline`、時計・上がり・適性を含む52項目の `form`、過去通過順を追加した66項目の `enhanced`、保存済み条件を加えた96項目の `context`、平地では当日傾向を加えた104項目の `bias` を比較します。3着以内専用モデルでのbias設定は既定の縮小4レース・減衰0.8に固定し、順位モデルでは後述の3設定を比較します。2025年の `race_log_loss` が最も小さい設定を区分ごとに選び、同点なら少ない特徴量と小さい木を優先します。選んだ設定を2025年末までのデータで再学習し、選定後の1モデルについて2026年分を評価します。モデルの重みと候補の選定には2026年の正解を使いません。`--validation-start` / `--test-start` で境界を指定できます。各区分に学習200行・検証50行・最終評価50行と正例/負例の両方が必要です。不足した区分は理由付きで除外します。

入力は出走条件、年齢・性別・負担重量、前日までの馬・騎手・調教師の成績などです。今回レースの着順・時計・上がり・通過順・確定オッズ・人気・馬体重・天候・馬場状態は入力にしません。同日の全レースの特徴量を作ってから通常の履歴を更新します。`bias` の当日分だけは別の状態から、同日・同場・同芝ダの先行レースを使います。未取得の戦績は未知のままです。取得期間より前の成績や取得漏れまで復元するものではありません。品質レポートに月・区分別の件数を残しますが、CSVだけでは未開催の日と取り漏らした日を区別できません。

拡張特徴には、過去の速度、勝馬や同じレースの完走馬平均からの時計差、上がりとその平均との差、コース・距離帯別の出走数と3着内率を加えます。時計差は距離で換算し、同じ過去レース内で比較することで条件差を軽減します。時計の履歴と適性集計は中央平地・中央障害・地方平地・ばんえいごとに分けます。原本にない時計や競走中止・失格馬の時計を0にせず、欠損のまま扱います。

### 初版を学習済みの場合の更新

同じ補完済みCSVに時計・上がりも保存されているため、再収集は不要です。新しい出力先 `data/ml_top3_v2` で作り直し、初版の `data/ml_top3` を残してください。旧features.sqlite3には時計・上がりを保持していないので、更新後のtrainだけでは追加特徴を復元できません。

```powershell
& {
    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath (Join-Path $env:USERPROFILE "keiba")
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Code update failed." }
    .\.venv\Scripts\python.exe -X utf8 -m keiba_ai fit --input data\results_5years_reparsed.csv --output-dir data\ml_top3_v2 --feature-set auto
    if ($LASTEXITCODE -ne 0) { throw "Training did not complete." }
}
```

追加のライブラリ導入は不要です。`auto` では比較候補が増えるため、初版より学習に時間がかかります。時計・上がりの列がないCSVも読み込めますが、その場合は時計に関する追加特徴は欠損になります。`quality_report.json` の欠損数で確認してください。

新版の `models/evaluation.json` には、区分ごとの `validation_comparison` に従来版と拡張版それぞれの2025年成績、`selected_feature_set` に採用した特徴量、`test` に採用モデルの2026年成績を保存します。追加特徴が良いとは決めつけず、2025年の成績が良い方を採用します。予測時も学習時に採用された列だけを適用します。`--feature-set baseline` または `enhanced` で一方に固定することもできます。

以前共有した2026年の成績を今回も再評価するため、2026年分は完全に新しい未使用期間ではありません。改善案の比較・採用は2025年の成績で行い、2026年の数値を見ながら調整を繰り返さないでください。将来の未使用レースでの評価も必要です。

予測は発走前に出馬表と出走馬が確定した段階を想定します。その時点で既知の取消・除外を反映し、変更があれば再予測してください。取消・除外を除いた相手関係も使うため、過去の結果CSVでの評価は、この時点の情報が揃っていたという前提があります。結果ページを今取得したデータでは、当時の変更履歴や厳密な公開時刻までは再現できません。馬券購入時点での実証には、その時点の出馬表観測を別途蓄積してください。

3着以内は `finished` かつ着順3以内を正例、競走中止・失格を負例とし、取消・除外は対象外です。同着では正例が4頭以上になる場合もあります。確率をレース内で合計1に揃える処理は行いません。「3着以内」と「複勝の的中条件」は別で、少頭数などの複勝ルールや払戻金はこのモデルに含めません。

評価の `race_log_loss` / `race_brier` は小さいほど良く、各レースを等しく重み付けします。`top_pick_top3_rate` はモデル最上位の馬が3着以内だった割合、`precision_at_3` は推奨上位3頭のうち3着以内だった割合です。比較対象はモデルと同じ出走対象頭数からの `3/頭数` と、順位指標についての確定人気上位3頭です。確定人気は参考比較にだけ使います。回収率・利益は評価していません。

## 過去の通過順を使い、各順位の確率分布を学習する

`--target rank` は、各馬の1着〜出走頭数分の各順位と、未完走・失格の確率を出す順位モデルです。3着以内の確率も1〜3着の合計として出します。過去通過順を含める指定は `--feature-set enhanced` です。取得済みCSVから作り直せるので、再収集や追加ライブラリの導入は不要です。出力先は新しく `data/ml_rank_v1` とし、以前のモデルを残します。

```powershell
& {
    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath (Join-Path $env:USERPROFILE "keiba")
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Code update failed." }
    .\.venv\Scripts\python.exe -X utf8 -m keiba_ai fit --input data\results_5years_reparsed.csv --output-dir data\ml_rank_v1 --target rank --feature-set enhanced
    if ($LASTEXITCODE -ne 0) { throw "Rank training did not complete." }
}
```

過去通過順は、頭数で補正した最初・最後の観測位置、位置取りの変化、明示された3角→4角・4角→ゴールの順位改善、過去5走の前方率や相手の前方傾向など14項目として追加します。「最初の観測」は必ずしも1角とは限りません。角番号のない2値から3角・4角を推測せず、並走グループの共有先頭順位もそのまま使います。欠損・矛盾・不正な通過順だけを未知にして品質レポートに警告を残します。当日・未来・非完走馬の通過順を入力にせず、過去日の完走記録だけを使います。

順位モデルは完走馬の順位比を回帰し、別に完走確率を学習します。その予測値からPlackett–Luce順序をサンプルし、完走有無を確率計算で積分します。未完走馬に架空の最下位を付けません。各馬の全順位と未完走の合計は100%です。全馬完走の場合は各順位の全馬合計も100%。未完走の可能性がある場合、各順位の合計は「その順位まで完走馬がいる確率」になり、下位ほど100%未満になり得ます。

この順位分布は同着を仮定しません。同着や順位が連続しないレースは順位モデルの学習・評価から除外し、件数を `excluded_rows` / `excluded_races` に記録します。そのレースを翌日以降の過去成績として使う処理は継続します。既存の3着以内専用モデルでは、これまで通り同着も正例として扱います。

順位モデルも区分ごとに2025年だけで木の設定と分布の温度を選びます。`--feature-set auto` の場合は `enhanced`（66項目）、`context`（96項目）、平地では `bias`（104項目）を比較します。`--feature-set form` は従来どおり明示指定できます。選定後、2025年末までで再学習して2026年を評価します。順位回帰・完走予測・分布計算のため、3着以内専用モデルより時間がかかります。

`models/evaluation.json` の `race_outcome_log_loss` は各馬の実際の順位または未完走の周辺確率を採点し、レース単位で平均した値です。レース全体の着順の同時確率を採点した値ではありません。`conditional_expected_rank_mae` は完走した馬について完走条件付き予測平均順位と実着順の差、`nonfinish_brier` は未完走確率の誤差です。予測完走確率が0で条件付き平均を定義できないケースはMAEから除き、有効件数を `conditional_rank_rows` に記録します。実際に起きた事象へ確率0を割り当てた場合も、対数損失では1e-12へクリップして採点します。`top3` 内に従来の3着以内評価も保存します。同着等を除外しているため、旧モデルと件数が同じとは限りません。

学習時には `models/central_flat_test_rank_probabilities.csv` など区分別のCSVも保存します。評価対象の2026年レースについて、各馬の順位確率と実着順を確認できるファイルです。今後のレースの予測とは別で、対象日より前の履歴だけで計算しています。ファイル名は評価JSONの `test_predictions_file` にも記録します。

順序の計算は近似で、既定は候補選定128サンプル、最終評価1024サンプル、出馬表の予測4096サンプルです。完走有無は独立を仮定して厳密に積分するため、未完走確率や行・列の整合性はサンプル数に依存しません。順位の周辺確率にはサンプル誤差があり、同じレースIDとseedで再現します。0.1%を同じ完走確率の一様順序分布と混合して極端なゼロを抑えます。`--validation-samples` / `--evaluation-samples`、予測の `--samples` で回数を指定できます。一様順位の比較基準は乱数なしで厳密に計算します。

予測対象の出馬表CSVがある場合は次で順位分布を出力します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai predict --dataset data\ml_rank_v1\features.sqlite3 --model-dir data\ml_rank_v1\models --entries data\entries.csv --output data\rank_predictions.csv
```

CSVの `rank_1_probability` / `rank_2_probability` / … が各順位、`nonfinish_probability` が未完走・失格、`top3_probability` が3着以内です。値は0〜1なので、0.25は25%です。`expected_rank_given_finish` は完走する場合の平均順位、`starters` は取消・除外を除いた出走対象頭数です。異なる頭数のレースを含むCSVでは最大頭数まで順位列を揃え、その馬にとって存在しない順位は0にします。出力は3着以内確率の高い順です。

## 保存済みのオッズ・馬体重・馬場とクラスを使う

`--feature-set context` は、従来の通過順までの66項目に次の30項目を加えます。出力は各馬の順位確率分布のままです。

- 今回のクラス・格付け・斤量条件・対象年齢・1着賞金、前走のクラス・格付けと今回との一致。
- 前走と直近5走の馬体重、前走の増減と直近5走の増減絶対値の平均。
- 前走と直近5走の単勝オッズの対数、頭数で補正した人気、人気に対する着順の改善。
- 前走の天候・馬場状態、前走と直近5走の水分量、重・不良と良・稍重それぞれでの過去出走数・3着内率。
- 直近5走で馬体重・オッズが観測できた件数。

前走・直近5走と馬場別の成績は、中央平地・中央障害・地方平地・ばんえいごとに分け、予測日より前だけを使います。過去レースの競走中止・失格馬でも、記録済みの馬体重やオッズは履歴に残しますが、架空の着順は補いません。地方のクラスを中央のクラスへ換算せず、各ラベルをカテゴリとして扱います。複合クラスなど判定できないものは未知のままです。

今回レースの確定オッズ・人気・馬体重・天候・馬場状態は直接入力しません。結果ページから集めた値では予測時点で公表されていた値を再現できないため、今回は過去情報として活用します。今回のクラスや斤量条件など、事前に定まる開催条件は直接使います。当日情報を直接使う段階では、発走前の取得時刻付きデータを揃えて別に評価する必要があります。

以前のCSVにはクラス等の条件全文がないため、`--race-db data\keiba.sqlite3` を指定すると、CSVのsnapshot_idとレース・地域を照合し、保存済みHTMLのヘッダーから学習用データへ補完します。原本のチェックサムも確認します。再スクレイピング、血統収集、収集DBや既存CSVの書き換えは行いません。DBと同じdataフォルダーの `raw/` も必要です。原本がない・破損している場合は補完せず、`quality_report.json` の `quality.metadata_enrichment` に件数と警告を残します。任意項目の欠損だけでレース全体を捨てません。

学習済みの旧モデルを残して、PowerShellで次を実行します。追加ライブラリの導入は不要です。

```powershell
& {
    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath (Join-Path $env:USERPROFILE "keiba")
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Code update failed." }
    .\.venv\Scripts\python.exe -X utf8 -m keiba_ai fit --input data\results_5years_reparsed.csv --race-db data\keiba.sqlite3 --output-dir data\ml_rank_context_v1 --target rank --feature-set context
    if ($LASTEXITCODE -ne 0) { throw "Rank training did not complete." }
}
```

保存HTMLの読み込みもあるため、処理には時間がかかります。PCをスリープさせずに実行してください。`data\ml_rank_context_v1\models\evaluation.json` が新しい評価、`data\ml_rank_context_v1\quality_report.json` が欠損・補完の報告です。精度向上を保証するものではなく、評価の数値で確認します。すでに確認した2026年分の再評価なので、完全な未使用期間としては扱いません。

prepareだけ完了して学習が中断した場合は、作成済みの `features.sqlite3` を使い、次で学習からやり直せます。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai train --dataset data\ml_rank_context_v1\features.sqlite3 --output-dir data\ml_rank_context_v1\models --target rank --feature-set context
```

今後の予測には、この新しい特徴量DB・モデルと、更新済みスクレイパーで出力した出馬表を使います。出馬表にもクラス等を保存するようになっています。条件のない旧出馬表は、その項目が欠損になります。

## 当日の先行レースによるトラック傾向を使う

`--target rank --feature-set auto` は、中央平地・地方平地で当日傾向を含む104項目の `bias` も比較します。従来の96項目までの値は保持し、次の8項目を末尾に追加します。順位分布の出力方式と学習方法は引き続き決定木の勾配ブースティング＋PLです。

| 項目 | 内容 |
|---|---|
| `day_front_bias` | 人気に対する着順の上振れと最終通過位置から測った前有利の傾向 |
| `day_inner_bias` | 人気に対する着順の上振れと馬番位置から測った内有利の傾向 |
| `day_bias_races` | 測定に使えた当日の先行レース数 |
| `day_bias_prior_front` | 過去日だけから作った前有利の事前値 |
| `horse_front_bias_fit` | 当日の前有利度と、その馬の過去5走の最終通過位置との交互作用 |
| `horse_inner_bias_fit` | 当日の内有利度と、今回の馬番位置との交互作用 |
| `field_front_runner_count` | 出走馬のうち過去の前方率が0.5以上の頭数。全馬の履歴がなければ欠損 |
| `meeting_day_index` | 中央の既知レースIDを開催情報と照合した開催何日目 |

内外位置は枠番ではなく馬番を使い、取消で馬番が飛んでいても元の宣言頭数・全出馬表の馬番を基準にします。完走6頭以上、完走馬の70%以上かつ6頭以上で人気・通過順等が利用でき、位置に十分なばらつきがある先行レースを採用します。人気補正は能力差を完全には除きません。最終通過位置と着順の関係や展開の影響も含む傾向の指標であり、馬場の因果効果を直接推定した値ではありません。

前日までの馬・騎手・調教師の履歴は、引き続き日内で凍結します。当日の測定は独立した状態へ追加し、同じ日・地域・競馬場・芝ダの、対象より小さいレース番号だけを使います。先行レースの確定人気・通過順・着順は使いますが、対象自身・後続レースの値は使いません。中央は場コード、地方は場コードと月日を既存の日付・競馬場と照合し、順序を確認できない形式は当日測定に使いません。芝とダートを分け、ばんえい・中央障害は対象外です。この2区分の `auto` は従来の `enhanced` / `context` で選びます。`--feature-set bias` に固定した場合は、この2区分を理由付きで除外します。

長期事前値は競馬場・芝ダ・200m距離帯別の平均と分散を保持し、日末にだけ更新します。中央で翌暦日かつ同開催・連続開催日と確認できれば、前日の同条件の値も混ぜます。当日値はレース番号差の減衰・距離の近さ・馬場状態で重み付けし、事前値へ縮小します。対象の確定馬場は参照せず、直近の確定済み先行レースの馬場状態を基準に、それと異なる先行レースの重みを下げます。

順位モデルでは `(縮小レース数, 減衰率)` の **`(3, 0.7)` / `(4, 0.8)` / `(5, 0.9)`** の3組を、木の大きさ・分布の温度と一緒に2025年だけで選びます。全組合せの探索ではありません。2026年は採用後の確認だけに使います。当日結果がない場合も事前値は利用でき、`day_bias_races=0` になります。

**過去データでの評価は、レース番号順を結果の利用可能順とする仮定を置きます。** 保存済み結果には当時の確定・公開時刻がないため、発走遅延や審議などによる公開順の逆転までは再現できません。`quality_report.json` と `evaluation.json` に `intraday_availability_assumption` を記録します。実際の予測へ渡す当日CSVには、その時点ですでに確定した結果を含めてください。

PowerShellで新しい出力先へ学習します。CSV・収集DB・既存モデルを保持し、ネットワーク通信や追加ライブラリの導入なしで実行します。

```powershell
& {
    $ErrorActionPreference = "Stop"
    Set-Location -LiteralPath (Join-Path $env:USERPROFILE "keiba")
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Code update failed." }
    .\.venv\Scripts\python.exe -X utf8 -m keiba_ai fit --input data\results_5years_reparsed.csv --race-db data\keiba.sqlite3 --output-dir data\ml_rank_bias_v1 --target rank --feature-set auto
    if ($LASTEXITCODE -ne 0) { throw "Rank training did not complete." }
}
```

入力特徴量の版が変わるため、旧版の `features.sqlite3` やモデルを今回のコードでそのまま予測に使うことはできません。新しい `fit` が必要です。比較候補と保存する特徴量が増えるので、前回より時間・空き容量が必要です。既存の `features` 表の構造は保持し、別の `bias_variants` 表へ3設定分の8項目を保存します。学習時は1設定ずつ読み込み、全候補の大きな行列を同時には保持しません。処理中はPCをスリープさせないでください。

`data\ml_rank_bias_v1\models\evaluation.json` の `validation_comparison` が2025年の特徴量別比較、`bias_config_comparison` が3組の比較、`selected_feature_set` / `selected_bias_config` が採用結果です。2025年の検証と2026年の評価には、同じ予測確率から次の集計も作ります。

- `by_race_number`：1〜4R / 5〜8R / 9R以降 / 番号不明。
- `by_first_race`：1R / それ以降 / 番号不明。
- `by_intraday_support`：先行結果の測定0件 / 1件以上 / 不明。

評価で改善したかどうかは実データの数値で確認してください。2026年はすでに見た期間の再評価です。今回追加した値がレース後半で有効かの検討・設定選択には2025年を使います。

当日の結果を使う予測には `--same-day-results` を指定します。既存スクレイパーの `fetch --page results` / `export --page results` で、確定済みレースの全頭を含む結果CSVを作ってください。CSVへ出力する日付は、予測する出馬表の日付と一致させます。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai predict --dataset data\ml_rank_bias_v1\features.sqlite3 --model-dir data\ml_rank_bias_v1\models --entries data\entries.csv --same-day-results data\today_results.csv --output data\rank_predictions_bias.csv
```

当日CSVは一時DBへ検査・読み込みを行い、学習用DBや元のCSVへ書き込みません。不正・不完全なレースがあれば拒否します。他日・他場・芝ダの違うレース、対象自身と後続Rは、対象ごとに選別して当日状態に入れません。複数の予測対象Rの間でも当日状態を共有せず、後のRの状態が前のRに残ることを防ぎます。`--same-day-results` を省略した場合は、学習用DBに当日結果が入っていても使わず、事前値だけになります。

予測CSVにも `day_bias_races`、前・内の傾向値、採用特徴量、`intraday_bias_used`、`bias_prior_only`、設定ID、当日入力CSVのSHA-256を記録します。区分ごとに `auto` が旧特徴量を採用した場合、当日結果は順位予測へ影響しません。

## v3実験：現行の順位モデル（M0）とLightGBM＋Plackett–Luce（M1）の比較

`keiba_ai.experiment_v3` は、prepare済みの `features.sqlite3` をそのまま使い、2つの順位モデルを同じ条件で比較します。既存の学習・予測コマンドは変更しません。

- 特徴量：`context` の96項目から、過去走のオッズ・人気由来の7項目を除いた89項目。当日バイアス（人気基準のv1）も使いません。
- M0：現行の `train_rank` と同じ方式（順位比の回帰＋温度0.1/0.2/0.4）。
- M1：LightGBMで、レース内の着順のPlackett–Luce尤度を直接最適化します。木の本数は学習期間の最後の2か月で早期終了して決め、検証年は使いません。完走確率もLightGBMで学習します。段の重みは `all`（全着順）と `top5`（上位5着まで）を比較します。
- 検証：2022年4月以降で学習し、2023・2024・2025年をそれぞれ検証年にする3つのfoldで評価します（2021年10月〜2022年3月は履歴の構築だけに使います）。2026年以降の最終評価期間は読みません。
- 指標：各foldの `race_outcome_log_loss`（小さいほど良い）。`summary` にfold平均と、M1がM0より全foldで良いかを記録します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3 --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3_m0_m1 --threads 2
```

結果は `data\exp_v3_m0_m1\experiment.json` に保存されます。`lightgbm` が必要なので、初回は `scripts\setup-ml.ps1` を実行してください。

## v3実験B：段ごとの温度・中央地方の統合モデル・LightGBMの調整

`keiba_ai.experiment_v3b` は、v3実験のM1（LightGBM＋PL、全着順）を起点に、次の3点を同じ3つの検証foldで比較します。

- 構造：区分ごとのモデル（`separate`）と、中央平地・地方平地をまとめた1つのモデル（`pooled`、`is_central` を特徴量に追加）。
- ハイパーパラメータ：既定値と、Optunaによる探索（構造も探索の対象）。探索の目的関数は、検証年の着順のPL負の対数尤度の平均です（サンプリング不要で速い）。最初の2試行は既定値です。
- 温度：1つの温度と、1着・2着・3着・4着以降で別々の温度。どちらも学習期間末尾の早期終了用2か月で最尤推定し、検証年は使いません。

各設定をレース単位で採点し、`separate_default/single_t`（前回のM1に相当）との差を、レース単位の対応のあるブートストラップで95%区間付きで出します。レース単位の損失は `race_losses.csv` に保存します。2026年以降は読みません。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3b --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3b --threads 2 --tuning-trials 20 --tuning-hours 4
```

`lightgbm` と `optuna` が必要なので、初回は `scripts\setup-ml.ps1` を実行してください。`--tuning-trials 0` にすると調整を省いて温度と構造だけを比較します。

## v3実験C：発走前に公表される今回の馬体重・馬場状態

`keiba_ai.experiment_v3c` は、v3実験Bで最も良かった設定（区分ごとのLightGBM＋PL、調整済みパラメータ、4段の温度）を基準に、発走前に公表される値から作る特徴量の効果を測ります。

- `weight`：今回の馬体重と増減、直近5走の平均との差、普段の増減幅との差、斤量÷馬体重、休み明けでの増減、レース内での馬体重・増減の比較（8項目）。
- `going`：今回の馬場状態と天候、重・不良かどうか、今回と同じ馬場区分（良・稍重／重・不良）での本人の出走数と3着内率、前走からの馬場の変化（6項目）。
- `both`：両方。

今回レースの着順・時計・通過順・オッズ・人気・水分量は読みません。発表値が取れない場合に備え、学習期間ではレースごとに5%の確率でこれらの値を欠損にして学習します（`--announce-dropout`）。基準（`base`）との差を、レース単位の対応のあるブートストラップで95%区間付きで出します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3c --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3c --threads 2
```

## 学習済みモデルで出馬表を予測する

予測対象の出馬表を収集・CSV出力してから、次を実行します。entries CSVは同じスクレイパーの `export --page entries` の出力です。対象日の全頭を含めてください。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai predict --dataset data\ml_top3\features.sqlite3 --model-dir data\ml_top3\models --entries data\entries.csv --output data\top3_predictions.csv
```

出力の `top3_probability` が各馬の3着以内の確率です。学習期間内の日付は拒否し、履歴は常に予測日より前だけから再構築します。`history_last_date` と `days_since_history` に履歴の古さを表示します。元データが2026-10-10までなら、その後の戦績は自動では増えません。新しい成績を追加した場合は、新しい出力先でprepareと学習をやり直してください。コードと特徴量の版が変わった場合も再作成が必要です。

## 検証

```bash
.venv/bin/python -m unittest discover -s tests -v
```

中断・再開、強制終了後のタスク回復、取得制限、収集漏れの完了判定、日付・ID・頭数不一致、開催なし、5年の期間境界、圧縮原本と旧DBの互換、同時書き込み中のCSV出力、出力失敗時のファイル保護をネットワークなしで検証します。中央・地方・ばんえいの2021年の実ページでも解析・ID・日付・頭数を確認しています。
