'use strict';
(async()=>{const q=s=>document.querySelector(s);try{
const local=['localhost','127.0.0.1'].includes(location.hostname);
const url=local?'../outputs/management_status.json':'https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/management_status.json';
const response=await fetch(url+'?t='+Math.floor(Date.now()/60000),{cache:'no-store'});if(!response.ok)throw Error();
const data=await response.json();const age=Date.now()-Date.parse(data.checked_at);const stale=!Number.isFinite(age)||age>45*60000;
q('#health').textContent=stale?'管理状況が45分以上更新されていません。':data.status==='healthy'?'今回の自動検査で異常は検出されませんでした。':`確認が必要な項目：${data.findings.length}件`;
q('#health').className=stale||data.status!=='healthy'?'error':'';q('#checked').textContent='最終確認：'+data.checked_at;
q('#email').textContent='メール通知：'+(data.notification_channel==='github_actions'?'GitHubの従来の処理失敗メールで通知（SMTP設定不要・配信先はGitHubの通知設定に従います）':!data.email_configured?'未設定（送信先・送信サービスの設定が必要）':data.email_status==='sent'?'送信サービスが受け付けました':data.email_status==='delivery_failed'?'送信失敗。設定と送信サービスを確認してください。':'設定済み。新しい異常・復旧時に通知します。');
for(const item of data.findings){const box=document.createElement('article');const h=document.createElement('h3');h.textContent=data.owners[item.owner];const p=document.createElement('p');p.textContent=item.message+(item.race_ids.length?' 対象 '+item.race_ids.length+'レース':'');box.append(h,p);q('#findings').append(box);}
const lines=[];for(const [key,v] of Object.entries(data.coverage||{}))lines.push((key==='company'?'会社':'独立')+`予想：${v.saved}/${v.total}レース作成、未作成${v.missing}レース`);
for(const [key,v] of Object.entries(data.matched_comparison||{}))lines.push((key==='company'?'会社':'独立')+`・共通対象：${v.hits}/${v.races}レース的中、投資${v.stake_yen}円、払戻${v.return_yen??'未確定'}円`+(v.largest_hit_return_share!==null?'、最大の1的中への払戻依存 '+(v.largest_hit_return_share*100).toFixed(1)+'%':''));
if(data.department_mean_ticket_overlap!==null&&data.department_mean_ticket_overlap!==undefined)lines.push('部署間の平均買い目重複率：'+(data.department_mean_ticket_overlap*100).toFixed(1)+'%');lines.push(data.comparison_scope||'');q('#metrics').textContent=lines.join('\n');
}catch(error){q('#health').textContent='管理状況を取得できません。正常と判断せず、時間をおいて再読込してください。';q('#health').className='error';}})();
