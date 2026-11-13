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

## v3実験D：馬の強さから騎手の腕を分離する

`keiba_ai.experiment_v3d` は、これまでで最も良い設定（区分ごとのLightGBM＋PL、調整済みパラメータ、4段の温度、地方は今回の馬体重も使用）を基準に、騎手の扱い方を比べます。

1. 騎手の特徴量を使わない馬の強さモデル `f0` を学習します。
2. 検証年より前のレースは、四半期ごとに3つに分けたクロスフィッティングで強さを予測し、自分を学習に使ったモデルで自分の強さを予測しないようにします。
3. その強さをオフセットにしたPL尤度で、騎手ごとの上乗せ θ を日ごとに逐次推定します（半減期365日の長期版と90日の短期版）。各行には、その日より前の状態だけを使います。
4. 比較する4つの設定：`base`（今の騎手の通算率）、`nojockey`（騎手情報なし）、`theta_feature`（通算率の代わりに θ 由来の特徴量）、`theta_offset`（θ をオフセットとして馬の強さを学習し、予測は f＋γθ）。

温度と γ は早期終了用の期間で推定し、検証年は使いません。基準との差を、レース単位の対応のあるブートストラップで95%区間付きで出します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3d --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3d --threads 2
```

## v3実験E：馬場差とクラスを補正したスピード指数

`keiba_ai.speed` は、走破時計から因果的なスピード指数を作ります。時計は1000mあたりの秒数（ペース）で扱い、各レースの基準ペースは上位3頭の平均です。

- 基準ペース：競馬場×芝ダ×距離ごと（指数用）と、さらにクラス別（馬場差の測定用）を、指数移動平均で持ちます。
- 馬場差：その日・その競馬場×芝ダの各レースについて「レースのペース − クラス別基準」を求め、その中央値をレース数で縮小した値です。弱いクラスばかりの日を「遅い馬場」と誤認しないよう、クラス別の基準と比べます。
- 指数：−（馬のペース − 基準 − 馬場差）＋ 0.12×（斤量 − 55）。単位は秒/km（プラスが速い）。
- その日の指数は、すべての特徴量を作り終えてから計算するので、翌日以降の特徴量にしか入りません。

特徴量は、前走・直近5走平均・直近5走最高・同じ芝ダ×距離帯での最高・上昇傾向・指数の数・重不良での最高、レース内でのz値（平均・最高）、相手の最高との差、上位3頭の平均（11項目）です。

`keiba_ai.experiment_v3e` は、各区分のこれまでの最良設定（騎手θを除く。地方は今回の馬体重を含む）に指数を加えた場合を比べます。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3e --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3e --threads 2
```

## v3実験F：指数＋騎手θの組み合わせと、レース内の相対化

`keiba_ai.experiment_v3f` は、実験Eの `speed`（指数あり）を基準に比べます。

- `speed_rel`（両区分）：主要な能力の列（時計差、上がり、4角位置、コース・距離適性、騎手・調教師の3着内率、斤量、年齢、間隔、指数）について、同じレース内でのz値と順位を追加します（32項目）。同じレースの、すでに因果的な列だけを使います。
- `speed_theta`（地方のみ）：騎手の通算率に加えて、騎手θの特徴量を追加します。θ用の馬の強さは、騎手特徴量なし・指数ありのモデルを四半期ブロックでクロスフィッティングして求めます。
- `speed_theta_rel`（地方のみ）：両方。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3f --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3f --threads 2
```

## v3実験G：馬のレーティングと、相手の強さを考慮した成績

`keiba_ai.ratings` は、馬ごとの多人数レーティングを着順から日ごとに更新します（騎手θと同じPLのオンライン推定を、馬に当てはめたもの。半減期240日）。各行には、その日より前のレーティングだけを使います。

- レース前のレーティング、その不確かさ、短期版、レース内のz値、相手の最高との差、上位3頭の平均。
- 過去走の相手の強さ：その過去走の相手のレーティング平均（直近5走平均・前走）と、今回の相手との差（相手関係の上がり下がり）。
- 期待以上の走り（surprise）：相手のレーティングから求めた期待着順と実際の着順の差（直近5走平均・前走）。

`keiba_ai.experiment_v3g` は、これまでの最良設定（中央は指数あり、地方は指数＋騎手θ）にこれらを加えて比べます。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3g --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3g --threads 2
```

## v3実験H：段ごとの温度の推定方法

`keiba_ai.experiment_v3h` は、各区分の現在の最良設定（中央：指数＋レーティング、地方：馬体重＋指数＋騎手θ＋レーティング）を1回だけ学習し、温度の推定方法だけを変えて比べます。

- `stop_single` / `stop_4`（現行）/ `stop_4_shrunk`：学習期間の最後の2か月で、温度1つ・4段・4段（共通の値に強く寄せる）。
- `stop_power`：同じ2か月で、T_j = T_1 × j^β の2パラメータ。
- `oof_4` / `oof_6`：学習期間全体を四半期ブロックでクロスフィッティングした予測で、4段・6段。

どれも検証年より前のレースだけを使い、同じ検証予測を同じ乱数で採点します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3h --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3h --threads 2
```

## v3実験I：馬の強さを基準にしたトラックバイアス（v3）

`keiba_ai.bias_v3` は、レースごとに、馬の強さ（学習に使っていないクロスフィット予測）をオフセットに固定し、4角の位置と馬番の効果 β をPL尤度で推定します。競馬場×芝ダ×距離帯の事前値は前日までだけで更新し、当日の偏差は同じ日・同じ場・同じ馬場の先行レースから精度重み付きで縮小推定します。人気・オッズは使いません。

`keiba_ai.experiment_v3i` は、各区分の最良設定を基準に次を比べます。

- `bias_feature`：当日の偏差、事前値、予想位置・馬番との積（9項目）を特徴量に追加（案B）。
- `bias_offset`：学習データでは、自レース以外の当日のレースから求めた偏差×実際の位置を init_score にして強さを学び直し、予測は「強さ＋γ×バイアス項」（案A、γは早期終了用の期間で推定）。
- 温度は `stop_power` と、1着だけ自由で2着以降を関数形にした `stop_hybrid` の両方で採点します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3i --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\exp_v3i --threads 2
```

## v3実験J：血統の特徴量とニューラルネット

`keiba_ai.experiment_v3j` は、収集した5代血統（`data\keiba.sqlite3`）を使い、同じ検証（2023〜2025年、早期終了は直前2か月）で次を比べます。

- `A_lgbm`：現在の最良設定（LightGBM＋Plackett–Luce）。
- `B_lgbm_ped`：Aに血統の特徴量（父・母父・母の能力値。馬の強さを差し引いたうえで日ごとに更新。父は芝ダ別・距離帯別も）を追加。
- `C_nn`：ニューラルネット（PyTorch）。Aと同じ特徴量に、5代62頭の祖先・騎手・調教師の埋め込みと、同じレースの馬どうしの注意機構（Transformer）を加え、同じPlackett–Luceの損失で学習。
- `D_blend`：BとCの加重平均（重みは早期終了用の期間で推定）。`E_blend`：AとCの加重平均。

GPU版のPyTorchを入れてから実行します（`nvidia-smi` で確認できるNVIDIAのGPUが必要です）。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-nn.ps1
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3j --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --output-dir data\exp_v3j --threads 2
```

## v3実験K：ニューラルネットの改善

`keiba_ai.experiment_v3k` は、実験JのLightGBM部分（完走モデル・A・B・ニューラルネットの入力）を `--cache-dir` に保存し、2回目以降はそれを再利用してニューラルネットだけを学習し直します。比べるのは、実験Jの設定（`nn_j`）、学習率の山を最後まで下げる設定（`nn_sched`）、正則化を強めた設定（`nn_reg`）、その5本平均（`nn_reg_x5`）、foldごとの設定探索＋5本平均（`nn_search_x5`）と、それぞれのBとの混ぜ合わせ（`D_…`）です。基準はAと、実験Jで最良だった `D_nn_j` です。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3k --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --cache-dir data\v3k_cache --output-dir data\exp_v3k --threads 2
```

## v3実験L：TabM型のニューラルネット

`keiba_ai.experiment_v3l` は、実験Kと同じLightGBMの保存分（`--cache-dir`）を使い、次を比べます。

- `tabm`：TabM型（重みを共有した32個のモデルの平均）＋数値の周期埋め込み（各特徴量を学習された周波数のsin/cosでベクトル化）＋レース内の注意機構。
- `tabm_noattn`：注意機構なし（各馬を自分の特徴量だけで評価）。`tabm_noemb`：数値の埋め込みなし。
- 参考に `nn_j`（実験Jの設定）と `nn_sched`。それぞれ単体と、B（LightGBM＋血統）との混ぜ合わせ（`D_…`）で採点します。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3l --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --cache-dir data\v3k_cache --output-dir data\exp_v3l --threads 2
```

## v3実験M：TabMの設定探索と複数モデルの混ぜ合わせ

`keiba_ai.experiment_v3m` は、実験Kの保存分を使い、年ごとにTabMの設定（注意機構の有無、幅・層、正則化、数値の埋め込みの周波数、内部のモデル数など）を `--search-trials` 通り試し、早期終了用の期間で一番良い設定を1回（`tabm_search`）と3本平均（`tabm_search_x3`）で学習します。あわせて、B（LightGBM＋血統）と学習したすべてのネットワークを一度に混ぜる `D_multi` も採点します。GPUのメモリ不足になった設定は飛ばします。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.experiment_v3m --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --cache-dir data\v3k_cache --output-dir data\exp_v3m --threads 2
```

## v3パイプライン：最良設定での学習と出馬表の予測（オッズ・トラックバイアス不使用）

`keiba_ai.pipeline_v3` は、v3実験で最良だった設定で、中央平地・地方平地ごとのモデルを学習し、出馬表から各馬の順位分布を出します。

- 中央平地：89項目＋スピード指数＋馬のレーティング（騎手は従来の成績率）。
- 地方平地：上に加えて、今回の馬体重（学習時は5%のレースで隠す）と騎手θ（騎手特徴量なしの強さを四半期の交差推定で求め、それを基準に算出）。
- 強さはLightGBM＋Plackett–Luce（調整済みパラメータ）。`train-end` の直前2か月で早期終了し、温度（1着だけ自由、2着以降は a·j^b）も同じ2か月で推定します。
- 障害・ばんえいは対象外です（出馬表に含まれていれば飛ばします）。

学習（`--train-end` を省くと、データ最終日の翌日まで全部で学習します）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.pipeline_v3 fit --dataset data\ml_rank_bias_v2\features.sqlite3 --output-dir data\v3_model --threads 2
```

`--train-end 2025-07-01 --evaluate-until 2026-01-01` のように指定すると、学習期間の後を評価して `model.json` と `evaluation_<区分>.csv` に保存します。2026-01-01以降（最終評価期間）の評価には `--allow-test-period` が必要です。

予測（`train-end` 以降の日付のみ。履歴は予測日より前だけから作り直します）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.pipeline_v3 predict --dataset data\ml_rank_bias_v2\features.sqlite3 --model-dir data\v3_model --entries data\entries.csv --output data\v3_predictions.csv
```

学習期間より後の確定結果での採点（新しい成績を含めてprepareし直したfeatures.sqlite3を使います。特徴量の版が同じなら、モデルは学習し直さずに使えます）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.pipeline_v3 evaluate --dataset data\ml_rank_v3_new\features.sqlite3 --model-dir data\v3_model --output-dir data\v3_eval
```

各レースは前日までの成績だけで予測し（predictと同じ）、`evaluation.json` に全体・週ごとの指標と一様予測との差、`race_losses.csv` にレースごとの損失を保存します。

出力には `rank_k_probability`（k着の確率）、`nonfinish_probability`、`win_probability`、`top3_probability`、`expected_rank_given_finish`、`strength_score` が入ります。地方で馬体重が出馬表にあれば使い、なければ空欄扱いで予測します（`bodyweight_used`）。

## v4パイプライン（本番）：LightGBM＋血統 と TabM型ニューラルネットの混ぜ合わせ

`keiba_ai.pipeline_v4` は、実験Mで最良だった組み合わせ（`D_tabm_search_x3`）を本番用に学習・予測します。

- B：v3パイプラインの特徴量に、血統の特徴量（父・母父・母のθ。芝ダ別・距離帯別。基準の強さはv3モデルの交差推定）を加えたLightGBM＋Plackett–Luce。
- NN：TabM型ネットワーク（内部16本、数値の周期埋め込み、注意機構なし、祖先62頭・騎手・調教師の埋め込み）。設定は実験Mで2025年に中央・地方とも選ばれたものに固定し、乱数を変えた3本の平均。
- 強さ = w_B·B + w_NN·NN（重みと温度は直前2か月で推定）。予測にはGPUは不要です（学習はGPU推奨）。
- 予測時の履歴（指数・レーティング・騎手θ・血統θ）は日付・区分ごとに一度だけ作り、同じ日の2レース目以降は数秒で出ます。

学習（`--race-db` は血統を集めた収集DB）。確認用（2025年を採点し、実験Mの値と比べる）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.pipeline_v4 fit --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --output-dir data\v4_check --train-end 2025-01-01 --evaluate-until 2026-01-01 --threads 2
```

本番（データ最終日まで全部で学習）:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.pipeline_v4 fit --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --output-dir data\v4_model --threads 2
.\.venv\Scripts\python.exe -X utf8 -m keiba_ai.pipeline_v4 predict --dataset data\ml_rank_bias_v2\features.sqlite3 --race-db data\keiba.sqlite3 --model-dir data\v4_model --entries data\entries.csv --output data\v4_predictions.csv
```

出力はv3と同じ列に、`lightgbm_score`（B）と `network_score`（NN）が加わります。アプリに `keiba_app init --model-dir data\v4_model` で登録すると、毎晩の自動処理は新しく出てきた馬（直近60日のレース）の血統を集め、7日ごとにv4で学習し直します。予想するときに血統が未取得の馬がいれば、その場で取得してから予想します（取得に失敗しても、血統なしとして予想します）。

## 競馬AIアプリ（ブラウザで予想・記録・自動更新）

設計は `docs/app_design.md`。開催日 → 競馬場 → R でレースを選び、出馬表の並びで各馬の着順ごとの確率（小さな棒グラフ・クリックで拡大・ヒートマップ）を表示します。予想はすべて記録し、結果が出たら自動で照合します。外部サーバーは使わず、自分のPCで動きます。

初回（既存のDBと本番モデルを登録します。コピーはしません）:

```powershell
git pull
.\.venv\Scripts\python.exe -X utf8 -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -X utf8 -m keiba_app init --dataset data\ml_rank_bias_v2\features.sqlite3 --model-dir data\v3_model
.\.venv\Scripts\python.exe -X utf8 -m keiba_app serve
```

ブラウザで http://127.0.0.1:8765 を開きます。毎晩の自動処理（2:30）、朝のレース一覧の取得と全レースの予想（7:30。どちらもスリープ解除あり）、ログオン時の画面の起動を登録するには（時刻は `-NightlyTime 03:00 -MorningTime 08:00` のように変えられます）:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-app-tasks.ps1
```

登録後の画面はウィンドウなしで動くので、コードを更新したら次のコマンドで再起動します（手で `serve` を起動する必要はありません。もう1つ起動しても、ポートが使用中で止まります）:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\restart-app.ps1
```

`pipeline_v4 fit --segments` を付けると、区分（中央：芝ダート×距離×年齢×クラス、地方：距離×年齢×クラス。距離は短距離〜1399m・マイル1400〜1899m・中距離1900〜2399m・長距離2400m〜）ごとの補正も学習し、`--evaluate-until` の期間で区分なしのモデルと比べます（`model.json` の `segments` と `evaluation.segmented`）。補正は、全レースで学習したモデルを出発点に、その区分のレースだけでLightGBM＋血統の続きの木と各シードのNNの追加学習を行い、どちらもその区分の早期終了期間で止めます。良くならない区分は全体のモデルのままです。早期終了期間に50レース未満、学習期間に300レース未満の区分は作りません。保存するモデルは今までと同じです。

学習に使う期間の始まりは `pipeline_v4 fit --train-start 日付`（既定: 2022-04-01）で変えられます。毎晩の学習し直しで使う開始日は `keiba_app init ... --train-start 日付 --data-start 日付` で設定します。データを過去に足したときは、DBの最初の日から半年ほど後を学習開始日にします（過去の成績から作る特徴量が揃うまでの期間）。

画面の「この日の全レースを予想」と「レース一覧を取得」は、アプリとは別のプロセス（`keiba_app day --date 日付`・`keiba_app discover --date 日付`、手でも実行できます）で、低い優先度で動きます。

毎晩の自動処理・`pipeline_v4`・`experiment_*` は、CPUとディスクの優先度を下げて動きます（画面や1レースの処理が先に進みます。何も動いていなければ速さはほぼ同じです）。通常の優先度で動かすには、先に `$env:KEIBA_PRIORITY = "normal"` を実行します。

毎晩の自動処理（`keiba_app nightly`）は、前回の続きから昨日までの成績の収集 → 新しい馬の血統の収集 → 全期間のCSV出力とprepare（新しいフォルダーに作成してから切り替え） → 予想の照合 → 7日ごとの学習し直し（健全性を確認してから切り替え） → 当日の全レースの先行予想、の順に実行します。画面の「状態」から手動でも実行できます。データは `data\app\` に保存します（`state.json`、予想の記録 `app.sqlite3`、DB・モデルの世代、ログ）。

外出先の端末から見るには、PCとスマホなどにTailscaleを入れて同じアカウントでログインし、PCで一度だけ次を実行します。表示されたHTTPSのアドレスを、自分の端末から開けます（自分のアカウントの端末以外からは見えません）。

```powershell
tailscale serve --bg 8765
```

**買い目の提案と自信度**：予想したレースの画面に「買い目の提案」が出ます。順位分布と同じ着順のシミュレーション（2万回）から、全券種（単勝・複勝・枠連・馬連・ワイド・馬単・3連複・3連単）の的中率を計算し、券種ごとに「確率の高い順に上からN点」の買い方のうち、指定した的中率（既定20%以上）・点数（既定10点まで）に合うものを出します。オッズを取得すると、「どの買い目が当たっても払戻がほぼ同じ」になるよう100円単位で配分し（オッズの低い買い目ほど多く買う）、当たっても損になる買い方（トリガミ）は出しません。合計金額は、設定した資金（既定5万円）が一番速く増える額（ケリー基準：シミュレーションの着順で、log(1+(払戻−合計)/資金) の平均を最大にする額）を、1レースの上限（既定1,000円）までの中から選び、その「資金の伸び」（1レースあたりの対数成長率）の高い順に並べます。複勝・ワイドは低い方のオッズを使います。マイナスは、モデルの確率では買わない方が良いという意味です。モデルの確率は高めに出やすいので、資金を実際より多めに設定すると控えめな金額になります。その下の「組み合わせの提案」は、2〜3券種を組み合わせた買い方です。単独では的中率の低い買い目（3連単の数点など）も部品に入れ、券種ごとに資金の伸びの良い4つずつを候補に、2〜3券種の組み合わせを全部比べて、資金の伸びの高い順に出します。どの買い目が1つだけ当たっても合計金額以上が戻る配分で、複数当たれば払戻は足し算になります（的中率は、どれかが当たる確率で、指定した的中率は組み合わせ全体に適用）。オッズは予想には使いません。自信度は本命（勝率が一番高い馬）の勝率で、v4の検証年（2025年、`experiment_confidence`）の3等分の境目で分けます（中央：37.6%以上がA・24.8%未満がC、地方：46.8%以上がA・31.1%未満がC）。その年の同じ自信度のレースでの本命の勝率・3着内率と、アプリで記録した予想での実績を表示します。勝率の偏りやモデル間の食い違いなどより、本命が3着内に来るかをよく分けられた指標です。どの自信度でも単勝の回収率は100%を超えなかったので、当たりやすさの目安として使います。

もう1つの「予想全体の自信度」は、1〜3着の並び方がどれだけ絞れているか（上位3着の並びのエントロピーから、1 − エントロピー / log(n(n−1)(n−2))）で決めます。モデルの確率が調整されていれば、上位3着の並びの当てやすさの期待値そのものです。2025年の検証では、実際の当てやすさとの順位相関が中央0.375・地方0.435で、勝率の偏り・本命の勝率・オッズを使わない候補の組み合わせ学習のどれよりも高く、段階ごとの平均も実際の値とほぼ一致しました。境目は中央：0.195以上がA・0.118未満がC、地方：0.245以上がA・0.150未満がCです。

オッズの取得（中央は全券種を1回の通信で。地方は単勝・複勝・枠連・馬連・ワイド・馬単の5ページ。地方の3連複・3連単は軸馬ごとにページが分かれるので、画面から取得するときは、予想済みならモデルの確率が高い軸3頭の分だけ取ります。3連単は1着の確率が高い3頭、3連複は確率の高い60通りの3連複を一番多く含む3頭を軸にし、1頭につき1ページ・約10秒増えます。コマンドでは `--trio-axis 馬番` / `--trifecta-axis 馬番` で指定します）は画面の「オッズを取得」か、`keiba_scraper odds --region central --race-id 202605050811`（地方は `--region local`）。発走まで変わるので毎回新しく取得します。形式確認用に原本を保存するには `--sample-dir data\odds_samples` を付けます。

コマンドだけで使う場合: `keiba_app predict --region central --race-id 202505050811 --refresh`（1レースを予想して記録）、`keiba_app score`（照合）、`keiba_app nightly --only collect prepare`（段階を指定）。

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
