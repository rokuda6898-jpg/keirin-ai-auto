# 競輪AI 会社運用ダイジェスト

- 更新: 2026-10-08T14:25:26+09:00
- 中央戦史: 60,043レース
- 履歴キャッシュ: 60,043レース
- 正式な発走前予想検証: 512レース
- Top1的中率: 0.435547
- モデル監査: unmatched_cohort_gap
- ライブ入力ドリフト: baseline_unavailable
- モデル3連単Top10的中率: 0.15503875968992248
- 実商品3連単的中率: 0.07462686567164178
- 戦史クロスソース一致: 1.0
- 本番再現Top1: 未算出
- 本番再現3連単Top10: 未算出
- 本番再現最終券: 未算出
- 3連単専用AI純増pp: 未算出
- 司令塔AI純増pp: 未算出
- 実戦スナップショット学習済R: 91
- モデル入力完成スナップショットR: 91
- 実戦専用モデル状態: collecting
- 実戦専用モデル純増pp: 未算出
- 昇格候補: 0
- 全買い目回収率検証部: 保存 33R / 確定 31R（120%未検証）
- 直近1年の部署知識: 17328R / 2644選手（全履歴保持・部署別の影予想）
- 締切前時刻確認: 132R ／ 従来集計 512R（未確認分は実戦証拠に含めない）
- 第三者監査: amber

## 軍師提言
- P1: treat latest selected trifecta tickets as the primary product KPI; keep model Top10 coverage as a diagnostic KPI
- P2: optimize ticket logic against live 10-point hit rate as a separate KPI from Top1 accuracy
- P2: keep the current best challenger in shadow until the 300-race promotion floor is reached
- P2: continue prospective shadow testing of the best second-pick reversal rule; no live switch until promotion criteria pass
- P2: keep multi-head coverage for near-tie races; do not collapse to a single head

## 専務の周知
- 本番再現室: 簡易バックテストではなく現行処理順の時系列リプレイを基準にする。
- 3連単組合せ班: 1-2-3の順序付き組合せを直接採点し、Production10点と影対決する。
- 司令塔AI: レースごとに専門部署を選ぶが、本番変更権限は持たない。
- 実戦学習庫: 朝・40分前・20分前・10分前の本番モデル入力そのものを蓄積し、確定後のみ教師ラベルを付与する。
- 実戦専用モデル部: 500Rで影学習開始、1,500Rで外部審査候補、さらに300Rの前向き同一レース勝負を必須とする。
- モデル鮮度部: 直近重視チャレンジャーを影運用し、本番モデルの経年劣化を監視する。
- データ品質監査: 学習時との特徴量ドリフトと戦史クロスソース一致率を監視する。
- 券種監査: Top1とは別に3連単10点の実戦的中率を正式KPIとして監視する。
- モデル監査室: 本番と全チャレンジャーを同一確定レースで比較し、純増のない補正を昇格させない。
- 2番手逆転班: Top1→2番手入替候補を影運用し、300R以上の前向き検証まで本番変更しない。
- データ部: 累積戦史を保持し、新規確定結果を差分追加する。
- 展開部: 全戦史から圧縮した脚質・展開知識を維持する。
- ライン部: 全戦史のライン傾向と本番予想の失敗を分離して評価する。
- リスク部: 発走前固定予想だけを正式成績として監視する。

## 権限
- オーナーが最終決裁者。
- AI組織内ではCEOが最上位。軍師・専務・各部署・秘書・第三者監査はCEOを上書きしない。
- 第三者監査はCEOへ反証質問を返し、判断力を鍛えるが決裁権は持たない。
- 本番変更は検証を通過した候補だけを採用する。
