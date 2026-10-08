# keirin-ai-auto

GitHub Actionsで自動起動する競輪予想AIスターターです。

## 買い目更新（2026-10-06）

既存の着順別モデルとNEXUS表示を維持し、3連単の本線6/8/10/12点・穴0/6/8/10/12点を上限として自動切替します。強い軸は1着固定、2・3着候補を広げて生成後にEV順で圧縮。本線EV>=1.10、穴EV>=1.25、穴12点は荒れ指数60以上と十分なEV候補が必要です。重複を除き、上限まで無理に埋めず条件不足は見送ります。

暫定係数と判定条件は [BETTING_LOGIC.md](BETTING_LOGIC.md)。`outputs/latest_race_strategy.json` に指数内訳・フォーメーション・点数・見送り理由を保存します。回収率120%は未達成の目標。新戦略とモデルに対応する外部検証合格までは購入停止・影予想を維持します。

## できること

- 毎朝、自動で予想CSVを作成
- 毎晩、決済後に最新の過去データでモデル再学習
- 週1回、定期的にモデル再学習
- WINTICKETの出走表を自動取得
- WINTICKETの過去結果から学習用 `history.csv` を自動構築
- データが取れない場合はサンプルデータで動作確認
- outputs/latest_predictions.csv に予想結果を保存
- outputs/latest_bets.csv に実際に買う複数券種の候補と購入額を保存
- outputs/latest_bet_candidates.csv に検討した複数券種の候補を保存
- outputs/japanese_report.md に日本語の購入理由と損益結果を保存
- outputs/growth_log.csv に学習精度と損益の成長記録を追記
- outputs/trial_report.md に直近31日の実購入ベース損益を保存
- outputs/latest_backtest.csv に過去データでの損益検証を保存
- outputs/profit_backtest_report.md に過去実オッズ込みの損益バックテストを保存
- outputs/multi_bet_backtest_report.md に券種別の損益バックテストを保存
- outputs/index.html に簡易表示を保存

## 本物データを使う場合

GitHub Repository Settings > Secrets and variables > Actions で以下を設定してください。

- `KEIRIN_HISTORY_CSV_URL`
- `KEIRIN_TODAY_CSV_URL`

`KEIRIN_TODAY_CSV_URL` が空の場合は、WINTICKETの出走表ページから今日の出走表を自動取得します。自動取得を止めたい場合は Actions の環境変数で `AUTO_FETCH_TODAY_ENTRIES=0` を指定してください。

CSV形式は、最低限以下の列を想定しています。

### history.csv

```csv
race_id,date,venue,race_no,player_id,car_no,age,score,win_rate,place2_rate,place3_rate,back_count,style,recent_avg_finish,days_since_last_race,venue_win_rate,odds_win,finish_pos
```

### today_entries.csv

```csv
race_id,date,venue,race_no,player_id,car_no,age,score,win_rate,place2_rate,place3_rate,back_count,style,recent_avg_finish,days_since_last_race,venue_win_rate,odds_win
```

## 手動実行

```bash
pip install -r requirements.txt
python src/make_sample_data.py --if-missing
python src/ingest.py
python src/build_history.py --months-back 24 --max-races 10000 --workers 8 --sleep-sec 0 --progress-every 250
python src/train.py
python src/profit_backtest.py --min-train-dates 30 --min-prob 0.10 --min-expected-profit 1200 --max-odds 200 --max-bets-per-race 2 --max-bets-per-day 40 --base-stake 100 --max-stake 500
python src/multi_bet_backtest.py --min-train-dates 30 --retrain-every-days 7 --top-k 5 --min-prob 0.02 --min-expected-profit 100 --max-odds 300 --max-bets-per-race-type 2 --max-bets-per-day-type 40 --base-stake 100 --max-stake 500
python src/fetch_today_entries.py
python src/predict.py
```

買い目は、券種別バックテストで絞った条件をデフォルトにしています。

- `BET_BASE_STAKE_YEN`: 最低購入額。標準は `100`
- `BET_MAX_STAKE_YEN`: 1点あたりの購入上限。標準は `500`
- `BET_DAILY_BUDGET_YEN`: 1日あたりの購入予算上限。GitHub Actions は `10000` 円

```bash
BET_BASE_STAKE_YEN=100 BET_MAX_STAKE_YEN=500 python src/predict.py
```

購入額は買い目の確率と100円あたりの期待利益に応じて100円単位で配分し、当日の予算を超えません。締切前の複数スナップショット間でも、別レースに割り当て済みの金額を予約額として差し引きます。単勝オッズは取得元に単勝プールがある場合だけ表示・計算し、未取得なら「未取得」としてEV・払戻予想・単勝購入額は空欄/0にします。別券種オッズから単勝オッズを推定しません。精算時も公式払戻がない的中は実損益に含めません。

## 出力

`outputs/latest_predictions.csv` には以下の損益列が入ります。

- `stake_yen`: 1点あたりの賭け金
- `win_return_yen`: 当たった場合の払戻金
- `win_profit_yen`: 当たった場合の利益
- `loss_amount_yen`: 外れた場合の損失
- `expected_profit_yen`: 確率とオッズから見た期待利益

`outputs/backtest_summary.csv` は過去データのテスト期間で、実際に当たり外れを判定した損益サマリーです。

`outputs/latest_bets.csv` には、AIが実際に買う候補だけを出します。WINTICKETからオッズが取れた場合は、以下も入ります。

- `bet_type`: 券種
- `bet_label`: 日本語の券種名
- `buy`: 買い目
- `odds_used`: 判定に使ったオッズ
- `return_if_hit_yen`: 当たった場合の払戻金
- `expected_profit_yen`: AI確率とオッズから見た期待利益

`outputs/latest_bet_candidates.csv` は、AIが検討した全3連単候補です。

`outputs/purchase_plan.csv` は、`expected_profit_yen > 0` の買い目だけを抜き出した購入予定です。

`outputs/settled_bets.csv` と `outputs/settlement_summary.csv` は、レース結果取得後の損益集計です。

`outputs/japanese_report.md` は、購入条件、損益、主な買い目、選んだ理由を日本語でまとめたレポートです。

`outputs/growth_log.csv` は、学習時のAUC、本命的中率、結果精算後の損益を追記していく成長記録です。

`outputs/trial_report.md` は、直近31日の決済ログを実購入ベースで合算した1か月トライアル用レポートです。

`outputs/shadow_report.md` は、購入停止中に記録した影予想を公式払戻で精算した日別・券種別の成長記録です。`outputs/shadow_daily_summary.csv` に日別損益、`outputs/shadow_settled_bets.csv` に全買い目の勝敗を追記します。

朝の低流動性オッズを期待値判定に使わないため、`intraday-keirin-odds` は締切5〜40分前のレースだけを再取得します。旧方式と新方式の成績は `outputs/shadow_strategy_summary.csv` で分けて確認できます。

週次再学習モデルは `models/candidate_win_model.joblib` として保存します。外部検証を通過するまで `models/win_model.joblib` は自動で置き換えません。

`outputs/walk_forward_summary.csv` は、結果を見ずに日付順で予想し、結果が出たら次の日の学習に追加する実戦形式の検証です。

`outputs/profit_backtest_report.md` は、過去の実オッズを使った損益バックテストです。

`outputs/multi_bet_backtest_report.md` は、2車単・2車複・ワイド・3連複・3連単などを横並びで比較した損益バックテストです。

## 注意

これは予想補助ツールです。的中や利益を保証するものではありません。

## 厳密検証と購入ゲート

履歴作成では、落車・失格・途中棄権を含む全出走者を保持します。学習前に次の監査を実行し、出走者が1人でも欠けたレースがあれば失敗します。

```bash
python src/audit_history.py --fail-on-leakage
```

時間順の開発・校正・ホールドアウト検証は次で実行します。

```bash
python src/honest_backtest.py --rebuild-entry
```

`src/external_holdout.py` は、開発履歴とURLが重複しないレースを評価し、使用モデルのSHA-256を記録します。`outputs/external_holdout_overall.json` の `target_passed` が `true` になるまで、`outputs/latest_bets.csv` は0点です。未承認候補は `outputs/latest_shadow_bets.csv` にだけ保存されます。


### 買い目・回収率検証部

`src/ticket_return_department.py` は部署追加後の発走前に固定した本線・穴の全点数（最大24点）を公式払戻で検証します。予想ごとの全買い目と見送りを保持し、同一レースは最新の有効な一組を使います。旧予想の事後復元はしません。

会社更新時に `outputs/company/ticket_return_department.html` とJSONを生成し、本線・穴・合計の的中率、100円均等/既存配分での回収率、見送り率、月別成績を報告します。実購入の成績は別集計。公式払戻が未取得のレースは除外し、確定済み成績は日付が変わっても保持します。回収率120%の単純超過だけで購入を許可したりモデルを昇格させたりしません。


### 全レース・全予想部署の提出義務（2026-10-08）

取得できた各レースに対し、データ部・展開部・ライン部・リスク部・予想部・軍師・高配当戦略部がそれぞれ1着→2着→3着の暫定シナリオを出します。買い目の選定・購入可否とは独立しています。価格が不足しても着順予想を空欄にせず、年間実績が不足する選手はモデル補完と明示します。

- `outputs/company/all_department_predictions.json` / `.html`：レース×7部署の予想とデータ状態。予想が欠けた発走前のレースは公開時の異常として記録します。
- `outputs/company/all_department_prediction_ledger.jsonl`：時刻確認済みの発走前予想のみ保存。締切後に結果を見て予想を作りません。
- `outputs/company/all_department_results.json` / `.html`：公式着順と照合した部署ごとの1着・2着・3着・並び一致。実際の購入成績・回収率ではありません。
- `src/site_manager.py`：7部署の欠損を検知したときは再生成を試行。出走表未取得・締切済み・事前記録なしの場合は、予想を捏造せず不足と明記します。

高配当戦略部の着順仮説は、実際の100倍以上の車券を意味しません。100倍以上の実オッズ・期待値・購入許可は別に判定します。7部署の一致は将来の的中を保証しません。

### 累積履歴と直近1年の専門部署

全履歴を保持し、専門部署の参照期間は予想日の前日までの直近1年とします。年間成績・直近90日・ライン位置別成績・実測の決まり手/行動記録を選手別に集計し、データ部・展開部・ライン部・リスク部が独立した影予想を提出します。既存の学習済み本番モデルを未検証の1年モデルに置換しません。`company/annual_department_report.html`に件数、部署別の同一レース検証を表示します。

決まり手は公式結果のfactor、行動記録は公式結果に明示されたフラグだけを保存します。過去の欠損を脚質から補いません。履歴更新・日付変更で年間集計を更新し、モデルの過去成績集計は入力・履歴・処理コード一致を確認したキャッシュを再利用します。新しいオッズは毎回維持します。

実戦データは重い特徴量計算の前に保存します。定期起動で実行中の収集を取り消しません。公開の再計算で発走前証跡が消えないよう、保存済みの検証台帳を合流します。着順確率の校正と本番モデル対補正後の比較は発走前固定記録から集計し、自動的な購入許可・校正は行いません。

### 部署別の2・3着比較 v2（2026-10-08）

`company/annual_position_v2_report.html` に、通常の予測処理で締切40〜5分前に保存した同一時点の二者比較を表示します。旧v1台帳とリスク部の現行評価・買い目は保持します。単独の年間集計処理は履歴更新と照合に限定し、均等な入力確率から別の最新予想を作りません。既存の「初回予想」成績と「締切前最新予想」成績は参考表示のまま、v2へ混ぜません。

比較方式は共通選別の従来評価、①部署の2・3着評価保持、②順位別の条件付き市場支持、①＋②、①＋②＋専門展開、市場支持のみです。①②は同じ候補選別・従来評価から固定したリスク方針で比較し、正規化の尺度差が候補数に影響しません。②は1着固定後の2着、1・2着固定後の3着の市場支持を、それぞれ対応する評価と一度だけ幾何平均し、候補選別にも反映します。市場のみの方式は確率順の参考基準で、AIのEV条件を満たしたと偽りません。

- 主要比較は本線・穴それぞれ6点、1点100円。候補不足は比較不成立とし補充しません。リスク部との比較は現行の本線・穴それぞれの点数・金額へ合わせます。各最大12点、穴は取得時100倍以上です。
- 公開中の専門部署の評価保持修正も従来の選択点数を上限とし、点数を増やしません。専門展開は別の未校正仮説です。過去の完全な出走表から前日までの相手強度・級班・車立てを照合し、確認済みラインと過去に観測された行動に応じて1・2着の組合せ別の評価を変えます。不足データを作らず、利用不可の理由を保存します。
- 一部オッズの欠損・上限張り付きでも①の比較を残し、完全かつ同一時点のオッズが必要な②・市場比較を分けます。全方式一致を条件にせず、比較する二者ごとに対象レースと除外件数を表示します。
- 公式同着は複数の的中組合せ・各組合せの払戻で照合します。払戻未確認のレースを回収率の分母へ入れません。的中率とは分け、リスク部の穴の払戻寄与と最大1的中を除いた回収率も表示します。
- 実装・モデル・入力・オッズ・保存記録のハッシュを保存し、実装版別に集計します。最初の保存から56日間、各主要比較500レース・28開催日を事前条件とし、日単位の再標本化と24指標の多重比較補正で的中率・回収率を別に判定します。期間途中の有意判定、件数不足時の都合のよい延長、自動採用はしません。
- 全組合せの対数損失・Brierスコア・予測確率帯別実測も記録します。これらは将来の校正に向けた診断で、的中率改善やEVの正しさを保証するものではありません。

変更は `department_ticket_v2.py`、`department_context.py`、`department_experiment_v2.py` に隔離します。`betting_logic.py` と `high_payout_strategy.py` のリスク基準は変更せず、台帳の保存・合流と公開直前の整合検査にv2を含めます。再生成した過去予想を評価するバックテストは行いません。
